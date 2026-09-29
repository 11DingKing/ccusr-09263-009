"""分批发运的 SQLite 持久化：追踪号落库、重启后单批失败可独立重试、整单汇总。"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime

from service_09252_008.application.booking_service import BookingService
from service_09252_008.application.catalog_service import CatalogService
from service_09252_008.application.ports import (
    ManualClock,
    UuidIdGenerator,
    UuidTrackingNumberGenerator,
)
from service_09252_008.persistence.sqlite_store import SQLiteStore
from tests.helpers import NOW, SLOT_START, apply_payload, seed_catalog


class PartialShipmentSQLiteTests(unittest.TestCase):
    def test_tracking_numbers_generated_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(f"{tmp}/booking.db")
            clock = ManualClock(NOW)
            catalog = CatalogService(store, clock, UuidIdGenerator())
            bookings = BookingService(store, clock, UuidIdGenerator(), UuidTrackingNumberGenerator())
            ids = seed_catalog(catalog)
            applied = bookings.apply(apply_payload(ids, "k-sql-apply"))
            booking_id = applied["booking_id"]
            bookings.quote(booking_id)
            bookings.lock(booking_id, {"idempotency_key": "k-sql-lock"})
            view = bookings.ship(
                booking_id,
                {
                    "idempotency_key": "k-sql-ship",
                    "parts": [
                        {"batch_id": ids["dye_batch_id"], "quantity": 2.0},
                        {"batch_id": ids["dye_batch_id"], "quantity": 3.0},
                        {"batch_id": ids["cloth_batch_id"], "quantity": 10.0},
                    ],
                },
            )
            # 直接读 SQLite 原始记录，确认追踪号已持久化
            raw_tracking: list[str] = []
            for record in store.query("shipments"):
                self.assertIn("parts", record)
                raw_tracking.extend(p["tracking_number"] for p in record["parts"])
            self.assertEqual(len(raw_tracking), 3)
            self.assertEqual(len(set(raw_tracking)), 3)  # 每批追踪号唯一
            self.assertTrue(all(t.startswith("TRK-") for t in raw_tracking))
            self.assertEqual(view["arrival_summary"]["part_count"], 3)
            store.close()

    def test_failed_part_retried_independently_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/booking.db"
            clock = ManualClock(NOW)

            # 第一次“进程”：染料拆 2.0/3.0 两批，第二批失败
            store = SQLiteStore(db_path)
            catalog = CatalogService(store, clock, UuidIdGenerator())
            bookings = BookingService(store, clock, UuidIdGenerator(), UuidTrackingNumberGenerator())
            ids = seed_catalog(catalog)
            applied = bookings.apply(apply_payload(ids, "k-sql2-apply"))
            booking_id = applied["booking_id"]
            bookings.quote(booking_id)
            bookings.lock(booking_id, {"idempotency_key": "k-sql2-lock"})
            view = bookings.ship(
                booking_id,
                {
                    "idempotency_key": "k-sql2-ship",
                    "parts": [
                        {"batch_id": ids["dye_batch_id"], "quantity": 2.0},
                        {"batch_id": ids["dye_batch_id"], "quantity": 3.0},
                        {"batch_id": ids["cloth_batch_id"], "quantity": 10.0},
                    ],
                },
            )
            dye = next(s for s in view["shipments"] if s["material_id"] == "dye")
            cloth = next(s for s in view["shipments"] if s["material_id"] == "cloth")
            shipment_id = dye["shipment_id"]
            part_ok, part_failed = dye["parts"][0], dye["parts"][1]
            bookings.record_arrival(shipment_id, {"part_id": part_ok["part_id"], "quantity": 2.0})
            bookings.report_part_failure(
                shipment_id,
                {"part_id": part_failed["part_id"], "reason": "跨境段丢件", "idempotency_key": "k-sql2-fail"},
            )
            failed_tracking = part_failed["tracking_number"]
            failed_part_id = part_failed["part_id"]
            cloth_part_id = cloth["parts"][0]["part_id"]
            store.close()

            # 第二次“进程”：全新服务实例挂载同一 SQLite，失败状态存活
            store2 = SQLiteStore(db_path)
            catalog2 = CatalogService(store2, clock, UuidIdGenerator())
            bookings2 = BookingService(store2, clock, UuidIdGenerator(), UuidTrackingNumberGenerator())
            shipment_view = bookings2.get_shipment(shipment_id)
            failed = next(p for p in shipment_view["parts"] if p["part_id"] == failed_part_id)
            self.assertEqual(failed["status"], "FAILED")
            self.assertEqual(failed["tracking_number"], failed_tracking)  # 追踪号持久化
            self.assertEqual(failed["quantity"] - failed["arrived_quantity"] - failed["lost_quantity"], 3.0)
            self.assertEqual(shipment_view["arrival_summary"]["failed_quantity"], 3.0)

            # 对该失败批单独重试，不影响其他批次
            view = bookings2.retry_part(
                shipment_id, {"part_id": failed_part_id, "idempotency_key": "k-sql2-retry"}
            )
            dye2 = next(s for s in view["shipments"] if s["material_id"] == "dye")
            old = next(p for p in dye2["parts"] if p["part_id"] == failed_part_id)
            retried = next(p for p in dye2["parts"] if p["superseded_by"] is None and p["attempt"] == 2)
            self.assertEqual(old["status"], "FAILED")
            self.assertEqual(old["superseded_by"], retried["part_id"])
            self.assertEqual(retried["quantity"], 3.0)
            self.assertNotEqual(retried["tracking_number"], failed_tracking)  # 新追踪号
            self.assertEqual(retried["status"], "IN_TRANSIT")

            # 重试批与布料到货后，SQLite 汇总整单实际到货齐全
            bookings2.record_arrival(shipment_id, {"part_id": retried["part_id"], "quantity": 3.0})
            cloth_shipment_id = next(
                s["shipment_id"] for s in view["shipments"] if s["material_id"] == "cloth"
            )
            bookings2.record_arrival(cloth_shipment_id, {"part_id": cloth_part_id, "quantity": 10.0})
            final = bookings2.get_booking(booking_id)
            summary = final["arrival_summary"]
            self.assertTrue(summary["all_arrived"])
            self.assertEqual(summary["totals"]["arrived_quantity"], 15.0)
            self.assertEqual(summary["totals"]["failed_quantity"], 0.0)
            self.assertEqual(summary["open_shipment_ids"], [])
            clock.set(datetime.fromisoformat(SLOT_START))
            checked = bookings2.checkin(booking_id)
            self.assertEqual(checked["status"], "CHECKED_IN")
            store2.close()


if __name__ == "__main__":
    unittest.main()
