"""材料分批发运：追踪号持久化、单批失败独立重试、整单实际到货汇总。"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime

from service_09252_008.application.booking_service import BookingService
from service_09252_008.application.catalog_service import CatalogService
from service_09252_008.application.ports import ManualClock, UuidIdGenerator
from service_09252_008.domain.errors import (
    BusinessRuleError,
    IdempotencyConflict,
    StateError,
    ValidationError,
)
from service_09252_008.persistence.sqlite_store import SQLiteStore
from tests.helpers import NOW, SLOT_START, apply_payload, event_types, make_services, seed_catalog


class SplitShipmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog, self.bookings, self.clock, self.store = make_services()
        self.ids = seed_catalog(self.catalog)
        applied = self.bookings.apply(apply_payload(self.ids, "k-sp-apply"))
        self.booking_id = applied["booking_id"]
        self.bookings.quote(self.booking_id)
        self.bookings.lock(self.booking_id, {"idempotency_key": "k-sp-lock"})
        shipped = self.bookings.ship(self.booking_id, {"idempotency_key": "k-sp-ship"})
        self.shipments = {s["material_id"]: s for s in shipped["shipments"]}
        self.dye_shipment_id = self.shipments["dye"]["shipment_id"]  # 染料 5.0
        self.cloth_shipment_id = self.shipments["cloth"]["shipment_id"]  # 布料 10.0

    def _split_dye(self, quantities: list[float], key: str = "k-sp-split") -> dict:
        view = self.bookings.split_shipment(
            self.dye_shipment_id, {"idempotency_key": key, "quantities": quantities}
        )
        return next(s for s in view["shipments"] if s["shipment_id"] == self.dye_shipment_id)

    # ------------------------------------------------------------------
    # 拆批与追踪号
    # ------------------------------------------------------------------

    def test_split_assigns_unique_tracking_numbers_per_part(self) -> None:
        shipment = self._split_dye([2.0, 3.0])
        self.assertEqual(len(shipment["parts"]), 2)
        tracking = [p["tracking_no"] for p in shipment["parts"]]
        self.assertTrue(all(t.startswith("trk_") for t in tracking))
        self.assertEqual(len(set(tracking)), 2)
        for part, quantity in zip(shipment["parts"], (2.0, 3.0)):
            self.assertEqual(part["quantity"], quantity)
            self.assertEqual(part["status"], "IN_TRANSIT")
            self.assertEqual(part["attempt"], 1)
            self.assertEqual(part["arrived_quantity"], 0.0)
            self.assertEqual(part["previous_tracking"], [])
        # 拆批本身不改变到货汇总：5.0 染料仍全部在途
        view = self.bookings.get_booking(self.booking_id)
        self.assertEqual(view["arrival_summary"]["by_material"]["dye"]["in_transit"], 5.0)
        self.assertEqual(view["arrival_summary"]["totals"]["in_transit"], 15.0)

    def test_split_equal_parts_by_count(self) -> None:
        view = self.bookings.split_shipment(
            self.dye_shipment_id, {"idempotency_key": "k-sp-even", "part_count": 2}
        )
        shipment = next(s for s in view["shipments"] if s["shipment_id"] == self.dye_shipment_id)
        self.assertEqual([p["quantity"] for p in shipment["parts"]], [2.5, 2.5])

    def test_split_requires_quantities_or_part_count(self) -> None:
        with self.assertRaises(ValidationError):
            self.bookings.split_shipment(self.dye_shipment_id, {"idempotency_key": "k-sp-bad1"})

    def test_split_quantities_must_sum_to_shipment_quantity(self) -> None:
        with self.assertRaises(ValidationError):
            self.bookings.split_shipment(
                self.dye_shipment_id, {"idempotency_key": "k-sp-bad2", "quantities": [2.0, 2.0]}
            )

    def test_split_is_idempotent_and_cannot_repeat(self) -> None:
        payload = {"idempotency_key": "k-sp-once", "quantities": [2.0, 3.0]}
        first = self.bookings.split_shipment(self.dye_shipment_id, payload)
        replay = self.bookings.split_shipment(self.dye_shipment_id, payload)
        self.assertTrue(replay["idempotent_replay"])
        first_parts = [p["part_id"] for p in next(
            s for s in first["shipments"] if s["shipment_id"] == self.dye_shipment_id
        )["parts"]]
        replay_parts = [p["part_id"] for p in next(
            s for s in replay["shipments"] if s["shipment_id"] == self.dye_shipment_id
        )["parts"]]
        self.assertEqual(first_parts, replay_parts)
        # 无幂等键再次拆批被拒
        with self.assertRaises(StateError):
            self.bookings.split_shipment(
                self.dye_shipment_id, {"idempotency_key": "k-sp-again", "part_count": 2}
            )

    def test_split_requires_idempotency_key(self) -> None:
        with self.assertRaises(ValidationError):
            self.bookings.split_shipment(self.dye_shipment_id, {"part_count": 2})

    # ------------------------------------------------------------------
    # 核心：单批失败后的独立重试
    # ------------------------------------------------------------------

    def test_failed_part_retried_independently_with_new_tracking(self) -> None:
        shipment = self._split_dye([2.0, 3.0])
        part_one, part_two = shipment["parts"]
        old_tracking = part_two["tracking_no"]

        # 第一批正常到货；第二批失败
        self.bookings.record_part_arrival(self.dye_shipment_id, part_one["part_id"], {"quantity": 2.0})
        view = self.bookings.fail_part(
            self.dye_shipment_id, part_two["part_id"], {"reason": "运输车辆故障"}
        )
        dye = next(s for s in view["shipments"] if s["shipment_id"] == self.dye_shipment_id)
        p1, p2 = dye["parts"]
        self.assertEqual(p1["status"], "ARRIVED")  # 成功批次不受影响
        self.assertEqual(p2["status"], "FAILED")
        self.assertEqual(dye["status"], "PARTIALLY_ARRIVED")
        # 整单汇总：染料到货 2.0、失败待重试 3.0
        dye_row = view["arrival_summary"]["by_material"]["dye"]
        self.assertEqual((dye_row["arrived"], dye_row["failed"], dye_row["in_transit"]), (2.0, 3.0, 0.0))
        self.assertIn(self.dye_shipment_id, view["arrival_summary"]["open_shipments"])
        self.assertIn("shipment_part_failed", event_types(view))

        # 失败批次不能直接登记到货
        with self.assertRaises(StateError):
            self.bookings.record_part_arrival(self.dye_shipment_id, part_two["part_id"], {"quantity": 3.0})
        # 未失败的批次不能重试
        with self.assertRaises(StateError):
            self.bookings.retry_part(
                self.dye_shipment_id, part_one["part_id"], {"idempotency_key": "k-sp-retry-p1"}
            )

        # 独立重试：part_id 不变，生成新追踪号，旧号入履历，尝试次数 +1
        retry_payload = {"idempotency_key": "k-sp-retry-p2"}
        view = self.bookings.retry_part(self.dye_shipment_id, part_two["part_id"], retry_payload)
        dye = next(s for s in view["shipments"] if s["shipment_id"] == self.dye_shipment_id)
        retried = next(p for p in dye["parts"] if p["part_id"] == part_two["part_id"])
        self.assertEqual(retried["status"], "IN_TRANSIT")
        self.assertEqual(retried["attempt"], 2)
        self.assertNotEqual(retried["tracking_no"], old_tracking)
        self.assertTrue(retried["tracking_no"].startswith("trk_"))
        self.assertEqual(retried["previous_tracking"], [old_tracking])
        event = next(e for e in view["events"] if e["type"] == "shipment_part_retried")
        self.assertEqual(event["payload"]["previous_tracking_no"], old_tracking)

        # 重试幂等：同键重放不再生成第三个追踪号
        replay = self.bookings.retry_part(self.dye_shipment_id, part_two["part_id"], retry_payload)
        self.assertTrue(replay["idempotent_replay"])
        replayed = next(
            p
            for s in replay["shipments"]
            if s["shipment_id"] == self.dye_shipment_id
            for p in s["parts"]
            if p["part_id"] == part_two["part_id"]
        )
        self.assertEqual(replayed["tracking_no"], retried["tracking_no"])
        self.assertEqual(replayed["attempt"], 2)

        # 同键不同载荷 -> 冲突
        with self.assertRaises(IdempotencyConflict):
            self.bookings.retry_part(
                self.dye_shipment_id,
                part_two["part_id"],
                {"idempotency_key": "k-sp-retry-p2", "note": "different"},
            )

        # 重试批次到货后染料发运单结清；第一批自始至终保持 ARRIVED
        view = self.bookings.record_part_arrival(self.dye_shipment_id, part_two["part_id"], {"quantity": 3.0})
        dye = next(s for s in view["shipments"] if s["shipment_id"] == self.dye_shipment_id)
        self.assertEqual([p["status"] for p in dye["parts"]], ["ARRIVED", "ARRIVED"])
        self.assertEqual(dye["status"], "ARRIVED")
        self.assertEqual(dye["arrived_quantity"], 5.0)
        # 布料仍在途：整单尚未全部到货
        totals = view["arrival_summary"]["totals"]
        self.assertEqual(
            (totals["shipped"], totals["arrived"], totals["in_transit"], totals["failed"], totals["lost"]),
            (15.0, 5.0, 10.0, 0.0, 0.0),
        )
        self.assertFalse(view["arrival_summary"]["all_arrived"])
        self.assertIn(self.cloth_shipment_id, view["arrival_summary"]["open_shipments"])

        # 布料直发批次同步到货后整单全部到货，可以签到
        view = self.bookings.record_arrival(self.cloth_shipment_id, {"quantity": 10.0})
        totals = view["arrival_summary"]["totals"]
        self.assertEqual(
            (totals["shipped"], totals["arrived"], totals["in_transit"], totals["failed"], totals["lost"]),
            (15.0, 15.0, 0.0, 0.0, 0.0),
        )
        self.assertTrue(view["arrival_summary"]["all_arrived"])
        self.assertEqual(view["arrival_summary"]["open_shipments"], [])
        self.clock.set(datetime.fromisoformat(SLOT_START))
        checked = self.bookings.checkin(self.booking_id)
        self.assertEqual(checked["status"], "CHECKED_IN")

    def test_partial_arrival_then_failure_then_retry_only_remainder(self) -> None:
        """批次先到货一部分再失败：重试沿用部分到货量，仅补发余量。"""
        shipment = self._split_dye([2.0, 3.0], key="k-sp-pr")
        part_two = shipment["parts"][1]
        # 第二批先到 1.0，再失败
        self.bookings.record_part_arrival(self.dye_shipment_id, part_two["part_id"], {"quantity": 1.0})
        self.bookings.fail_part(self.dye_shipment_id, part_two["part_id"])
        view = self.bookings.retry_part(
            self.dye_shipment_id, part_two["part_id"], {"idempotency_key": "k-sp-pr-retry"}
        )
        part = next(
            p
            for s in view["shipments"]
            if s["shipment_id"] == self.dye_shipment_id
            for p in s["parts"]
            if p["part_id"] == part_two["part_id"]
        )
        self.assertEqual(part["status"], "PARTIALLY_ARRIVED")
        self.assertEqual(part["arrived_quantity"], 1.0)
        # 余量 2.0 到货即可关闭该批；超过余量被拒
        with self.assertRaises(ValidationError):
            self.bookings.record_part_arrival(self.dye_shipment_id, part_two["part_id"], {"quantity": 2.5})
        view = self.bookings.record_part_arrival(self.dye_shipment_id, part_two["part_id"], {"quantity": 2.0})
        part = next(
            p
            for s in view["shipments"]
            if s["shipment_id"] == self.dye_shipment_id
            for p in s["parts"]
            if p["part_id"] == part_two["part_id"]
        )
        self.assertEqual(part["status"], "ARRIVED")

    def test_other_part_arrival_while_one_failed_keeps_aggregation(self) -> None:
        """一个批次失败期间，其他批次仍可到货，汇总实时反映。"""
        shipment = self._split_dye([2.0, 3.0], key="k-sp-mix")
        part_one, part_two = shipment["parts"]
        self.bookings.fail_part(self.dye_shipment_id, part_one["part_id"])
        view = self.bookings.record_part_arrival(self.dye_shipment_id, part_two["part_id"], {"quantity": 3.0})
        dye_row = view["arrival_summary"]["by_material"]["dye"]
        self.assertEqual((dye_row["arrived"], dye_row["failed"]), (3.0, 2.0))
        # 重试第一批并到货
        self.bookings.retry_part(self.dye_shipment_id, part_one["part_id"], {"idempotency_key": "k-sp-mix-r"})
        view = self.bookings.record_part_arrival(self.dye_shipment_id, part_one["part_id"], {"quantity": 2.0})
        self.assertEqual(view["arrival_summary"]["by_material"]["dye"]["arrived"], 5.0)

    # ------------------------------------------------------------------
    # 按批损耗、拆批后的直发拒绝、取消联动
    # ------------------------------------------------------------------

    def test_split_shipment_rejects_direct_arrival_and_loss(self) -> None:
        self._split_dye([2.0, 3.0], key="k-sp-direct")
        with self.assertRaises(StateError):
            self.bookings.record_arrival(self.dye_shipment_id, {"quantity": 1.0})
        with self.assertRaises(StateError):
            self.bookings.record_shipment_loss(self.dye_shipment_id, {"quantity": 1.0})

    def test_part_loss_closes_part_and_shipment(self) -> None:
        shipment = self._split_dye([2.0, 3.0], key="k-sp-loss")
        part_one, part_two = shipment["parts"]
        self.bookings.record_part_arrival(self.dye_shipment_id, part_one["part_id"], {"quantity": 2.0})
        # 第二批全部在途灭失
        view = self.bookings.record_part_loss(self.dye_shipment_id, part_two["part_id"], {"quantity": 3.0})
        dye = next(s for s in view["shipments"] if s["shipment_id"] == self.dye_shipment_id)
        self.assertEqual([p["status"] for p in dye["parts"]], ["ARRIVED", "CLOSED_WITH_LOSS"])
        self.assertEqual(dye["status"], "CLOSED_WITH_LOSS")
        self.assertEqual(view["arrival_summary"]["by_material"]["dye"]["lost"], 3.0)
        self.assertTrue(any(l["reason"] == "in_transit_loss" and l["quantity"] == 3.0 for l in view["losses"]))
        # 染料实际到货仅 2.0 < 需要 5.0，即使发运单关闭也不可签到
        self.bookings.record_arrival(self.cloth_shipment_id, {"quantity": 10.0})
        self.clock.set(datetime.fromisoformat(SLOT_START))
        with self.assertRaises(BusinessRuleError) as ctx:
            self.bookings.checkin(self.booking_id)
        self.assertIn("dye", ctx.exception.details["shortages"])

    def test_cancel_closes_failed_and_in_transit_parts_as_loss(self) -> None:
        shipment = self._split_dye([2.0, 3.0], key="k-sp-cancel")
        part_one, part_two = shipment["parts"]
        self.bookings.record_part_arrival(self.dye_shipment_id, part_one["part_id"], {"quantity": 2.0})
        self.bookings.fail_part(self.dye_shipment_id, part_two["part_id"])
        cancelled = self.bookings.cancel(self.booking_id, {"reason": "课程取消"})
        dye = next(s for s in cancelled["shipments"] if s["shipment_id"] == self.dye_shipment_id)
        self.assertTrue(all(p["status"] == "CLOSED_WITH_LOSS" or p["status"] == "ARRIVED" for p in dye["parts"]))
        failed_part = next(p for p in dye["parts"] if p["part_id"] == part_two["part_id"])
        self.assertEqual(failed_part["status"], "CLOSED_WITH_LOSS")
        self.assertEqual(failed_part["lost_quantity"], 3.0)
        self.assertEqual(dye["status"], "CLOSED_WITH_LOSS")
        by_reason: dict[str, float] = {}
        for loss in cancelled["losses"]:
            by_reason[loss["reason"]] = by_reason.get(loss["reason"], 0.0) + loss["quantity"]
        # 染料 5.0（含已到货 2.0 与失败待重试 3.0）+ 布料 10.0，取消后全部记损耗
        self.assertEqual(by_reason["cancel_after_shipment"], 15.0)

    def test_unknown_part_raises_not_found(self) -> None:
        self._split_dye([2.0, 3.0], key="k-sp-nf")
        with self.assertRaises(Exception) as ctx:
            self.bookings.fail_part(self.dye_shipment_id, "prt_missing")
        self.assertEqual(ctx.exception.code, "not_found")


class SplitShipmentPersistenceTests(unittest.TestCase):
    """追踪号与批次状态必须落 SQLite，失败批次可在重启后独立重试。"""

    def test_failed_part_retried_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/booking.db"
            clock = ManualClock(NOW)

            # 第一次“进程”：申请→报价→锁定→发运→拆 3 批→第二批失败
            store = SQLiteStore(db_path)
            catalog = CatalogService(store, clock, UuidIdGenerator())
            bookings = BookingService(store, clock, UuidIdGenerator())
            ids = seed_catalog(catalog)
            applied = bookings.apply(apply_payload(ids, "k-sql-apply"))
            booking_id = applied["booking_id"]
            bookings.quote(booking_id)
            bookings.lock(booking_id, {"idempotency_key": "k-sql-lock"})
            shipped = bookings.ship(booking_id, {"idempotency_key": "k-sql-ship"})
            dye_shipment_id = next(s for s in shipped["shipments"] if s["material_id"] == "dye")["shipment_id"]
            split = bookings.split_shipment(
                dye_shipment_id, {"idempotency_key": "k-sql-split", "quantities": [2.0, 1.0, 2.0]}
            )
            parts = next(s for s in split["shipments"] if s["shipment_id"] == dye_shipment_id)["parts"]
            failed_part = parts[1]
            bookings.record_part_arrival(dye_shipment_id, parts[0]["part_id"], {"quantity": 2.0})
            failed = bookings.fail_part(dye_shipment_id, failed_part["part_id"], {"reason": "转运点积压"})
            self.assertEqual(
                next(
                    p
                    for s in failed["shipments"]
                    if s["shipment_id"] == dye_shipment_id
                    for p in s["parts"]
                    if p["part_id"] == failed_part["part_id"]
                )["status"],
                "FAILED",
            )
            old_tracking = failed_part["tracking_no"]
            store.close()

            # 模拟重启：新实例挂载同一数据库，追踪号、尝试次数与失败状态完整保留
            store2 = SQLiteStore(db_path)
            catalog2 = CatalogService(store2, clock, UuidIdGenerator())
            bookings2 = BookingService(store2, clock, UuidIdGenerator())
            view = bookings2.get_booking(booking_id)
            shipment = next(s for s in view["shipments"] if s["shipment_id"] == dye_shipment_id)
            self.assertTrue(shipment["parts"])
            by_id = {p["part_id"]: p for p in shipment["parts"]}
            self.assertEqual(set(by_id), {p["part_id"] for p in parts})
            self.assertEqual(by_id[failed_part["part_id"]]["tracking_no"], old_tracking)
            self.assertEqual(by_id[failed_part["part_id"]]["status"], "FAILED")
            self.assertEqual(by_id[failed_part["part_id"]]["attempt"], 1)
            self.assertEqual(by_id[parts[0]["part_id"]]["status"], "ARRIVED")
            # 整单汇总同样可从 SQLite 重算
            self.assertEqual(view["arrival_summary"]["by_material"]["dye"]["failed"], 1.0)

            # 重启后独立重试失败批次（新追踪号由 Python 端生成并再次持久化）
            retried = bookings2.retry_part(
                dye_shipment_id, failed_part["part_id"], {"idempotency_key": "k-sql-retry"}
            )
            part_view = next(
                p
                for s in retried["shipments"]
                if s["shipment_id"] == dye_shipment_id
                for p in s["parts"]
                if p["part_id"] == failed_part["part_id"]
            )
            self.assertEqual(part_view["attempt"], 2)
            self.assertNotEqual(part_view["tracking_no"], old_tracking)
            self.assertEqual(part_view["previous_tracking"], [old_tracking])
            # 布料也到货，签收前最后一步
            cloth_id = next(s for s in retried["shipments"] if s["material_id"] == "cloth")["shipment_id"]
            bookings2.record_arrival(cloth_id, {"quantity": 10.0})
            # 失败批重试后的 1.0 与第三批 2.0 到货
            bookings2.record_part_arrival(dye_shipment_id, failed_part["part_id"], {"quantity": 1.0})
            third = parts[2]
            bookings2.record_part_arrival(dye_shipment_id, third["part_id"], {"quantity": 2.0})
            final = bookings2.get_booking(booking_id)
            dye = next(s for s in final["shipments"] if s["shipment_id"] == dye_shipment_id)
            self.assertEqual(dye["status"], "ARRIVED")
            self.assertEqual(dye["arrived_quantity"], 5.0)
            self.assertTrue(final["arrival_summary"]["all_arrived"])
            store2.close()


if __name__ == "__main__":
    unittest.main()
