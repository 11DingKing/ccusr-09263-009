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

    def test_split_shipment_failure_and_retry_over_http(self) -> None:
        apply_body = {
            "institution": "仓储职业学院",
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
        self.assertEqual(applied["status"], "REQUESTED")
        self._request("POST", f"/bookings/{booking_id}/quote", {})
        self._request("POST", f"/bookings/{booking_id}/lock", {}, headers={"Idempotency-Key": "http-split-lock"})

        # 染料 2.0 拆两批
        status, shipped = self._request(
            "POST",
            f"/bookings/{booking_id}/ship",
            {
                "parts": [
                    {"batch_id": self.ids["dye_batch_id"], "quantity": 1.0},
                    {"batch_id": self.ids["dye_batch_id"], "quantity": 1.0},
                    {"batch_id": self.ids["cloth_batch_id"], "quantity": 4.0},
                ]
            },
            headers={"Idempotency-Key": "http-split-ship"},
        )
        self.assertEqual(status, 200)
        dye = next(s for s in shipped["shipments"] if s["material_id"] == "dye")
        self.assertEqual(len(dye["parts"]), 2)
        bad_part, good_part = dye["parts"][0], dye["parts"][1]

        # 多批在途时不指定批次到货 -> 400
        status, body = self._request("POST", f"/shipments/{dye['shipment_id']}/arrivals", {"quantity": 1.0})
        self.assertEqual(status, 400)

        # 第一批失败，第二批正常到货
        status, failed = self._request(
            "POST",
            f"/shipments/{dye['shipment_id']}/failures",
            {"part_id": bad_part["part_id"], "reason": "丢件"},
            headers={"Idempotency-Key": "http-split-fail"},
        )
        self.assertEqual(status, 200)
        failed_part = next(
            p for s in failed["shipments"] if s["shipment_id"] == dye["shipment_id"] for p in s["parts"]
            if p["part_id"] == bad_part["part_id"]
        )
        self.assertEqual(failed_part["status"], "FAILED")

        # 单独重试失败批
        status, retried = self._request(
            "POST",
            f"/shipments/{dye['shipment_id']}/retries",
            {"part_id": bad_part["part_id"]},
            headers={"Idempotency-Key": "http-split-retry"},
        )
        self.assertEqual(status, 200)
        new_part = next(
            p for s in retried["shipments"] if s["shipment_id"] == dye["shipment_id"] for p in s["parts"]
            if p["attempt"] == 2
        )
        self.assertEqual(new_part["quantity"], 1.0)
        self.assertNotEqual(new_part["tracking_number"], bad_part["tracking_number"])

        # 单批发运单查询
        status, shipment_view = self._request("GET", f"/shipments/{dye['shipment_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(shipment_view["arrival_summary"]["part_count"], 3)


if __name__ == "__main__":
    unittest.main()
