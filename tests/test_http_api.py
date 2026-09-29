"""HTTP 接口边界：路由、幂等头、错误映射与端到端流程。"""
from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

from service_09252_008.interfaces.http_api import create_server
from tests.helpers import make_services, seed_catalog


class HttpApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog, bookings, clock, store = make_services()
        cls.ids = seed_catalog(catalog)
        cls.server = create_server("127.0.0.1", 0, catalog, bookings)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _request(
        self, method: str, path: str, body: dict | None = None, headers: dict | None = None
    ) -> tuple[int, dict]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method, headers=headers or {}
        )
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_health(self) -> None:
        status, body = self._request("GET", "/health")
        self.assertEqual((status, body["status"]), (200, "ok"))

    def test_booking_flow_over_http_with_idempotency_header(self) -> None:
        apply_body = {
            "institution": "港城理工学院",
            "package_id": self.ids["package_id"],
            "mentor_id": self.ids["mentor_id"],
            "resource_id": self.ids["resource_id"],
            "window_id": self.ids["window_id"],
            "seats": 6,
            "slot_start": "2026-10-01T02:00:00+00:00",
            "slot_end": "2026-10-01T04:00:00+00:00",
        }
        status, applied = self._request(
            "POST", "/bookings", apply_body, headers={"Idempotency-Key": "http-apply-1"}
        )
        self.assertEqual(status, 201)
        self.assertEqual(applied["status"], "REQUESTED")
        booking_id = applied["booking_id"]

        # 相同幂等键重放：不产生新预约
        status, replay = self._request(
            "POST", "/bookings", apply_body, headers={"Idempotency-Key": "http-apply-1"}
        )
        self.assertEqual(status, 201)
        self.assertEqual(replay["booking_id"], booking_id)
        self.assertTrue(replay["idempotent_replay"])

        # 同一键不同载荷 -> 409
        status, conflict = self._request(
            "POST", "/bookings", {**apply_body, "seats": 8}, headers={"Idempotency-Key": "http-apply-1"}
        )
        self.assertEqual(status, 409)
        self.assertEqual(conflict["error"], "idempotency_conflict")

        status, quoted = self._request("POST", f"/bookings/{booking_id}/quote", {})
        self.assertEqual(status, 200)
        self.assertEqual(quoted["status"], "QUOTED")

        status, locked = self._request(
            "POST", f"/bookings/{booking_id}/lock", {}, headers={"Idempotency-Key": "http-lock-1"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(locked["status"], "LOCKED")

        status, fetched = self._request("GET", f"/bookings/{booking_id}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched["booking_id"], booking_id)
        self.assertEqual(len(fetched["reservations"]), 2)

    def test_error_mapping(self) -> None:
        status, body = self._request("GET", "/bookings/bkg_missing")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

        status, body = self._request("POST", "/bookings", {"institution": "x"})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "validation_error")

        status, body = self._request("POST", "/no-such-route", {})
        self.assertEqual(status, 404)

    def test_admin_recover_endpoint(self) -> None:
        status, body = self._request("POST", "/admin/recover", {})
        self.assertEqual(status, 200)
        self.assertIn("expired_locks", body)
        self.assertIn("expired_quotes", body)

    def test_split_shipment_fail_and_retry_over_http(self) -> None:
        apply_body = {
            "institution": "河西职业学院",
            "package_id": self.ids["package_id"],
            "mentor_id": self.ids["mentor_id"],
            "resource_id": self.ids["resource_id"],
            "window_id": self.ids["window_id"],
            "seats": 4,
            "slot_start": "2026-10-01T06:00:00+00:00",
            "slot_end": "2026-10-01T08:00:00+00:00",
        }
        status, applied = self._request(
            "POST", "/bookings", apply_body, headers={"Idempotency-Key": "http-split-apply"}
        )
        self.assertEqual(status, 201)
        booking_id = applied["booking_id"]
        self._request("POST", f"/bookings/{booking_id}/quote", {})
        self._request("POST", f"/bookings/{booking_id}/lock", {}, headers={"Idempotency-Key": "http-split-lock"})
        status, shipped = self._request(
            "POST", f"/bookings/{booking_id}/ship", {}, headers={"Idempotency-Key": "http-split-ship"}
        )
        self.assertEqual(status, 200)
        dye = next(s for s in shipped["shipments"] if s["material_id"] == "dye")  # 4 * 0.5 = 2.0
        shipment_id = dye["shipment_id"]

        status, split = self._request(
            "POST",
            f"/shipments/{shipment_id}/split",
            {"quantities": [1.0, 1.0]},
            headers={"Idempotency-Key": "http-split-do"},
        )
        self.assertEqual(status, 200)
        parts = next(s for s in split["shipments"] if s["shipment_id"] == shipment_id)["parts"]
        self.assertEqual([p["quantity"] for p in parts], [1.0, 1.0])
        self.assertTrue(all(p["tracking_no"].startswith("trk_") for p in parts))

        # 第一批到货，第二批失败
        status, _ = self._request(
            "POST", f"/shipments/{shipment_id}/parts/{parts[0]['part_id']}/arrivals", {"quantity": 1.0}
        )
        self.assertEqual(status, 200)
        status, failed = self._request(
            "POST",
            f"/shipments/{shipment_id}/parts/{parts[1]['part_id']}/fail",
            {"reason": "道路封闭"},
        )
        self.assertEqual(status, 200)
        part = next(
            p
            for s in failed["shipments"]
            if s["shipment_id"] == shipment_id
            for p in s["parts"]
            if p["part_id"] == parts[1]["part_id"]
        )
        self.assertEqual(part["status"], "FAILED")
        old_tracking = part["tracking_no"]
        self.assertEqual(failed["arrival_summary"]["by_material"]["dye"]["failed"], 1.0)

        # 失败批直接到货 -> 409 invalid_state
        status, body = self._request(
            "POST", f"/shipments/{shipment_id}/parts/{parts[1]['part_id']}/arrivals", {"quantity": 1.0}
        )
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "invalid_state")

        # 独立重试：新追踪号
        status, retried = self._request(
            "POST",
            f"/shipments/{shipment_id}/parts/{parts[1]['part_id']}/retry",
            {},
            headers={"Idempotency-Key": "http-split-retry"},
        )
        self.assertEqual(status, 200)
        part = next(
            p
            for s in retried["shipments"]
            if s["shipment_id"] == shipment_id
            for p in s["parts"]
            if p["part_id"] == parts[1]["part_id"]
        )
        self.assertEqual(part["attempt"], 2)
        self.assertNotEqual(part["tracking_no"], old_tracking)
        self.assertEqual(part["previous_tracking"], [old_tracking])

        # 余量到货
        status, arrived = self._request(
            "POST", f"/shipments/{shipment_id}/parts/{parts[1]['part_id']}/arrivals", {"quantity": 1.0}
        )
        self.assertEqual(status, 200)
        dye_view = next(s for s in arrived["shipments"] if s["shipment_id"] == shipment_id)
        self.assertEqual(dye_view["status"], "ARRIVED")


if __name__ == "__main__":
    unittest.main()
