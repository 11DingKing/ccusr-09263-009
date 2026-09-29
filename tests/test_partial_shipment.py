"""材料分批发运：拆批追踪号、整单到货汇总、单批失败与独立重试。"""
from __future__ import annotations

import unittest
from datetime import datetime

from service_09252_008.domain.errors import StateError, ValidationError
from tests.helpers import SLOT_START, apply_payload, make_services, seed_catalog


def _book_and_ship_split(bookings, ids, *, dye=(2.0, 3.0)):
    """走通申请/报价/锁定，并把染料拆成若干批、布料整批发运。"""
    applied = bookings.apply(apply_payload(ids, "k-split-apply"))
    booking_id = applied["booking_id"]
    bookings.quote(booking_id)
    bookings.lock(booking_id, {"idempotency_key": "k-split-lock"})
    parts = [{"batch_id": ids["dye_batch_id"], "quantity": q} for q in dye]
    parts.append({"batch_id": ids["cloth_batch_id"], "quantity": 10.0})
    view = bookings.ship(booking_id, {"idempotency_key": "k-split-ship", "parts": parts})
    shipments = {s["material_id"]: s for s in view["shipments"]}
    return booking_id, shipments


class PartialShipmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog, self.bookings, self.clock, self.store = make_services()
        self.ids = seed_catalog(self.catalog)
        self.booking_id, self.shipments = _book_and_ship_split(self.bookings, self.ids)
        self.dye = self.shipments["dye"]
        self.cloth = self.shipments["cloth"]

    def test_each_part_has_persisted_tracking_number_and_status(self) -> None:
        self.assertEqual(self.dye["quantity"], 5.0)
        self.assertEqual(len(self.dye["parts"]), 2)
        tracking = [p["tracking_number"] for p in self.dye["parts"]]
        self.assertEqual(tracking, ["TRK-0001", "TRK-0002"])
        self.assertTrue(all(p["status"] == "IN_TRANSIT" for p in self.dye["parts"]))
        self.assertTrue(all(p["attempt"] == 1 for p in self.dye["parts"]))
        self.assertEqual([p["quantity"] for p in self.dye["parts"]], [2.0, 3.0])
        # 布料是第三批
        self.assertEqual(self.cloth["parts"][0]["tracking_number"], "TRK-0003")

    def test_split_quantities_must_cover_reservation(self) -> None:
        catalog, bookings, _, _ = make_services()
        ids = seed_catalog(catalog)
        applied = bookings.apply(apply_payload(ids, "k-bad-apply"))
        bid = applied["booking_id"]
        bookings.quote(bid)
        bookings.lock(bid, {"idempotency_key": "k-bad-lock"})
        with self.assertRaises(ValidationError):
            # 2.0 + 2.0 = 4.0 != 待发运 5.0，且漏发布料
            bookings.ship(
                bid,
                {
                    "idempotency_key": "k-bad-ship",
                    "parts": [
                        {"batch_id": ids["dye_batch_id"], "quantity": 2.0},
                        {"batch_id": ids["dye_batch_id"], "quantity": 2.0},
                    ],
                },
            )

    def test_failed_part_can_be_retried_independently(self) -> None:
        dye_parts = {p["part_id"]: p for p in self.dye["parts"]}
        p_first, p_second = self.dye["parts"][0], self.dye["parts"][1]

        # 第一批正常到货；第二批失败；布料整批到货 —— 失败不影响其他批次
        self.bookings.record_arrival(self.dye["shipment_id"], {"part_id": p_first["part_id"], "quantity": 2.0})
        self.bookings.record_arrival(
            self.cloth["shipment_id"], {"part_id": self.cloth["parts"][0]["part_id"], "quantity": 10.0}
        )
        view = self.bookings.report_part_failure(
            self.dye["shipment_id"],
            {"part_id": p_second["part_id"], "reason": "承运商丢件", "idempotency_key": "k-fail-1"},
        )

        dye_view = next(s for s in view["shipments"] if s["material_id"] == "dye")
        failed = next(p for p in dye_view["parts"] if p["part_id"] == p_second["part_id"])
        self.assertEqual(failed["status"], "FAILED")
        self.assertIsNone(failed["superseded_by"])
        # 整单汇总：染料到货 2.0、失败待重试 3.0，布料已齐
        summary = view["arrival_summary"]
        self.assertEqual(summary["open_shipment_ids"], [self.dye["shipment_id"]])
        self.assertFalse(summary["all_arrived"])
        totals = summary["totals"]
        self.assertEqual(totals["arrived_quantity"], 12.0)
        self.assertEqual(totals["failed_quantity"], 3.0)
        self.assertEqual(totals["in_transit_quantity"], 0.0)
        dye_row = next(r for r in summary["by_material"] if r["material_id"] == "dye")
        self.assertEqual((dye_row["arrived_quantity"], dye_row["failed_quantity"]), (2.0, 3.0))
        self.assertIn("shipment_part_failed", [e["type"] for e in view["events"]])

        # 失败批未重试成功前不可签到
        self.clock.set(datetime.fromisoformat(SLOT_START))
        with self.assertRaises(StateError):
            self.bookings.checkin(self.booking_id)

        # 仅对失败的那一批独立重试：获得新追踪号，attempt 递增，旧批保留审计链
        view = self.bookings.retry_part(
            self.dye["shipment_id"], {"part_id": p_second["part_id"], "idempotency_key": "k-retry-1"}
        )
        dye_view = next(s for s in view["shipments"] if s["material_id"] == "dye")
        old = next(p for p in dye_view["parts"] if p["part_id"] == p_second["part_id"])
        retry = next(p for p in dye_view["parts"] if p["part_id"] != p_first["part_id"] and p["part_id"] != old["part_id"])
        self.assertEqual(old["status"], "FAILED")
        self.assertEqual(old["superseded_by"], retry["part_id"])
        self.assertEqual(retry["quantity"], 3.0)
        self.assertEqual(retry["attempt"], 2)
        self.assertEqual(retry["tracking_number"], "TRK-0004")
        self.assertEqual(retry["status"], "IN_TRANSIT")
        # 重试后数量从“失败待重试”转为“在途”
        self.assertEqual(view["arrival_summary"]["totals"]["failed_quantity"], 0.0)
        self.assertEqual(view["arrival_summary"]["totals"]["in_transit_quantity"], 3.0)
        self.assertIn("shipment_part_retried", [e["type"] for e in view["events"]])

        # 重试批到货后整单汇总为全部到货，可签到
        view = self.bookings.record_arrival(self.dye["shipment_id"], {"part_id": retry["part_id"], "quantity": 3.0})
        dye_view = next(s for s in view["shipments"] if s["material_id"] == "dye")
        self.assertEqual(dye_view["status"], "ARRIVED")
        self.assertEqual(dye_view["arrived_quantity"], 5.0)
        self.assertTrue(view["arrival_summary"]["all_arrived"])
        self.assertEqual(view["arrival_summary"]["open_shipment_ids"], [])
        checked = self.bookings.checkin(self.booking_id)
        self.assertEqual(checked["status"], "CHECKED_IN")

    def test_retry_is_idempotent_and_only_resends_remainder_after_partial_arrival(self) -> None:
        p_first, p_second = self.dye["parts"][0], self.dye["parts"][1]
        # 第二批先到 1.0（部分到货），随后失败，仅重试剩余 2.0
        self.bookings.record_arrival(self.dye["shipment_id"], {"part_id": p_second["part_id"], "quantity": 1.0})
        self.bookings.report_part_failure(
            self.dye["shipment_id"], {"part_id": p_second["part_id"], "idempotency_key": "k-fail-2"}
        )
        first = self.bookings.retry_part(
            self.dye["shipment_id"], {"part_id": p_second["part_id"], "idempotency_key": "k-retry-2"}
        )
        replay = self.bookings.retry_part(
            self.dye["shipment_id"], {"part_id": p_second["part_id"], "idempotency_key": "k-retry-2"}
        )
        self.assertTrue(replay["idempotent_replay"])
        dye_first = next(s for s in first["shipments"] if s["material_id"] == "dye")
        dye_replay = next(s for s in replay["shipments"] if s["material_id"] == "dye")
        self.assertEqual(len(dye_first["parts"]), len(dye_replay["parts"]))
        retry_part = next(p for p in dye_first["parts"] if p["attempt"] == 2)
        self.assertEqual(retry_part["quantity"], 2.0)

        # 非失败批不可重试；失败批不可重复重试
        from service_09252_008.domain.errors import NotFoundError

        with self.assertRaises(StateError):
            self.bookings.retry_part(
                self.dye["shipment_id"], {"part_id": p_first["part_id"], "idempotency_key": "k-retry-x"}
            )
        with self.assertRaises(NotFoundError):
            self.bookings.retry_part(
                self.dye["shipment_id"], {"part_id": "prt_missing", "idempotency_key": "k-retry-y"}
            )

    def test_arrival_requires_part_selector_when_multiple_in_transit(self) -> None:
        with self.assertRaises(ValidationError):
            self.bookings.record_arrival(self.dye["shipment_id"], {"quantity": 1.0})
        # 指定追踪号同样可以定位批次
        tracking = self.dye["parts"][0]["tracking_number"]
        view = self.bookings.record_arrival(
            self.dye["shipment_id"], {"tracking_number": tracking, "quantity": 2.0}
        )
        part = next(
            p for s in view["shipments"] if s["material_id"] == "dye" for p in s["parts"]
            if p["tracking_number"] == tracking
        )
        self.assertEqual(part["status"], "ARRIVED")

    def test_failed_part_can_be_written_off_instead_of_retried(self) -> None:
        p_first, p_second = self.dye["parts"][0], self.dye["parts"][1]
        self.bookings.record_arrival(self.dye["shipment_id"], {"part_id": p_first["part_id"], "quantity": 2.0})
        self.bookings.report_part_failure(
            self.dye["shipment_id"], {"part_id": p_second["part_id"], "idempotency_key": "k-fail-4"}
        )
        # 确认灭失、不再补发：失败批直接登记在途损耗并关闭
        view = self.bookings.record_shipment_loss(
            self.dye["shipment_id"], {"part_id": p_second["part_id"], "quantity": 3.0}
        )
        dye_view = next(s for s in view["shipments"] if s["material_id"] == "dye")
        written_off = next(p for p in dye_view["parts"] if p["part_id"] == p_second["part_id"])
        self.assertEqual(written_off["status"], "CLOSED_WITH_LOSS")
        self.assertEqual(dye_view["status"], "CLOSED_WITH_LOSS")
        self.assertEqual((dye_view["arrived_quantity"], dye_view["lost_quantity"]), (2.0, 3.0))
        losses = [l for l in view["losses"] if l["reason"] == "in_transit_loss"]
        self.assertEqual(sum(l["quantity"] for l in losses), 3.0)

    def test_other_parts_continue_after_one_fails(self) -> None:
        p_second = self.dye["parts"][1]
        self.bookings.report_part_failure(
            self.dye["shipment_id"], {"part_id": p_second["part_id"], "idempotency_key": "k-fail-3"}
        )
        # 第一批仍可登记到货与在途损耗，互不干扰
        view = self.bookings.record_arrival(
            self.dye["shipment_id"], {"part_id": self.dye["parts"][0]["part_id"], "quantity": 2.0}
        )
        dye_view = next(s for s in view["shipments"] if s["material_id"] == "dye")
        self.assertEqual({p["part_id"]: p["status"] for p in dye_view["parts"]}[self.dye["parts"][0]["part_id"]],
                         "ARRIVED")
        # 布料发运单照常到货关闭
        cloth_part = self.cloth["parts"][0]
        view = self.bookings.record_arrival(
            self.cloth["shipment_id"], {"part_id": cloth_part["part_id"], "quantity": 10.0}
        )
        cloth_view = next(s for s in view["shipments"] if s["material_id"] == "cloth")
        self.assertEqual(cloth_view["status"], "ARRIVED")


if __name__ == "__main__":
    unittest.main()
