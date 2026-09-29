"""预约核心服务：申请、报价、锁定、发运、到货、签到、结算、取消与超时恢复。

事务与并发约定：
- 每个用例在单个 ``store.transaction()`` 内完成“读-判-写”，
  由存储层的可重入事务保证原子性，从而支持并发锁定；
- 变更类用例支持幂等键：相同键重放返回首次结果，不同载荷复用键则冲突；
- 所有状态迁移都会追加领域事件，便于审计与测试断言。
"""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from typing import Any, Callable

from ..domain.errors import (
    BookingImmutableError,
    BusinessRuleError,
    ConflictError,
    IdempotencyConflict,
    NotFoundError,
    StateError,
    ValidationError,
)
from ..domain.models import (
    ACTIVE_PART_STATUSES,
    CANCELLABLE_STATUSES,
    LOSS_CANCEL_AFTER_SHIPMENT,
    LOSS_DAMAGED_IN_USE,
    LOSS_IN_TRANSIT,
    LOSS_NON_RETURNABLE_LEFTOVER,
    QTY_EPS,
    RESOURCE_HOLDING_STATUSES,
    WINDOW_OCCUPYING_STATUSES,
    Booking,
    BookingStatus,
    DomainEvent,
    LossRecord,
    MaterialBatch,
    MaterialReservation,
    Mentor,
    PlannedAllocation,
    Quote,
    ReceptionWindow,
    Settlement,
    Shipment,
    ShipmentPart,
    ShipmentPartStatus,
    ShipmentStatus,
    WorkshopResource,
    dt_to_str,
)
from ..domain.rules import (
    ensure_mentor_qualified,
    ensure_resource_fit,
    ensure_slot_shape,
    ensure_window_fit,
    find_resource_conflict,
    plan_material_allocation,
    safety_ceiling,
)
from ..persistence.store import Store
from .catalog_service import (
    COLLECTION_BATCHES,
    COLLECTION_MENTORS,
    COLLECTION_PACKAGES,
    COLLECTION_RESOURCES,
    COLLECTION_WINDOWS,
)
from .ports import (
    Clock,
    IdGenerator,
    TrackingNumberGenerator,
    UuidTrackingNumberGenerator,
)

COLLECTION_BOOKINGS = "bookings"
COLLECTION_RESERVATIONS = "material_reservations"
COLLECTION_SHIPMENTS = "shipments"
COLLECTION_LOSSES = "material_losses"
COLLECTION_SETTLEMENTS = "settlements"
COLLECTION_EVENTS = "events"
COLLECTION_IDEMPOTENCY = "idempotency_keys"

DEFAULT_LOCK_TTL_SECONDS = 1800
DEFAULT_QUOTE_TTL_SECONDS = 86400
MIN_LOCK_TTL_SECONDS = 60
MAX_LOCK_TTL_SECONDS = 86400


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


class BookingService:
    """预约用例编排。"""

    def __init__(
        self,
        store: Store,
        clock: Clock,
        ids: IdGenerator,
        tracking_numbers: TrackingNumberGenerator | None = None,
        *,
        lock_ttl_seconds: int = DEFAULT_LOCK_TTL_SECONDS,
        quote_ttl_seconds: int = DEFAULT_QUOTE_TTL_SECONDS,
    ) -> None:
        self._store = store
        self._clock = clock
        self._ids = ids
        self._tracking_numbers = tracking_numbers or UuidTrackingNumberGenerator()
        self._lock_ttl = lock_ttl_seconds
        self._quote_ttl = quote_ttl_seconds

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, booking_id: str | None, payload: dict[str, Any]) -> None:
        event = DomainEvent(
            event_id=self._ids.new_id("evt"),
            type=event_type,
            booking_id=booking_id,
            payload=payload,
            created_at=self._clock.now(),
        )
        self._store.put(COLLECTION_EVENTS, event.event_id, event.to_dict())

    def _idempotent(
        self,
        endpoint: str,
        key: str | None,
        payload: dict[str, Any],
        fn: Callable[[], dict[str, Any]],
        *,
        required: bool,
    ) -> dict[str, Any]:
        """幂等执行：键命中且载荷一致则重放首次结果。"""
        if key is None:
            if required:
                raise ValidationError("idempotency_key is required for this operation")
            with self._store.transaction():
                return fn()
        fingerprint = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        with self._store.transaction():
            existing = self._store.get(COLLECTION_IDEMPOTENCY, key)
            if existing is not None:
                if existing["endpoint"] != endpoint or existing["request_hash"] != fingerprint:
                    raise IdempotencyConflict(
                        "idempotency key was already used with a different request",
                        details={"key": key, "endpoint": endpoint},
                    )
                return {**existing["response"], "idempotent_replay": True}
            result = fn()
            self._store.put(
                COLLECTION_IDEMPOTENCY,
                key,
                {
                    "key": key,
                    "endpoint": endpoint,
                    "request_hash": fingerprint,
                    "response": result,
                    "created_at": dt_to_str(self._clock.now()),
                },
            )
            return result

    # ------------------------------------------------------------------
    # 读取辅助
    # ------------------------------------------------------------------

    def _load_booking(self, booking_id: str) -> Booking:
        record = self._store.get(COLLECTION_BOOKINGS, booking_id)
        if record is None:
            raise NotFoundError(f"booking not found: {booking_id}", details={"booking_id": booking_id})
        return Booking.from_dict(record)

    def _save_booking(self, booking: Booking) -> None:
        booking.version += 1
        booking.updated_at = self._clock.now()
        self._store.put(COLLECTION_BOOKINGS, booking.booking_id, booking.to_dict())

    def _load_package(self, package_id: str):
        from ..domain.models import CoursePackage

        record = self._store.get(COLLECTION_PACKAGES, package_id)
        if record is None:
            raise NotFoundError(f"package not found: {package_id}", details={"package_id": package_id})
        return CoursePackage.from_dict(record)

    def _load_mentor(self, mentor_id: str) -> Mentor:
        record = self._store.get(COLLECTION_MENTORS, mentor_id)
        if record is None:
            raise NotFoundError(f"mentor not found: {mentor_id}", details={"mentor_id": mentor_id})
        return Mentor.from_dict(record)

    def _load_resource(self, resource_id: str) -> WorkshopResource:
        record = self._store.get(COLLECTION_RESOURCES, resource_id)
        if record is None:
            raise NotFoundError(f"resource not found: {resource_id}", details={"resource_id": resource_id})
        return WorkshopResource.from_dict(record)

    def _load_window(self, window_id: str) -> ReceptionWindow:
        record = self._store.get(COLLECTION_WINDOWS, window_id)
        if record is None:
            raise NotFoundError(f"window not found: {window_id}", details={"window_id": window_id})
        return ReceptionWindow.from_dict(record)

    def _load_batch(self, batch_id: str) -> MaterialBatch:
        record = self._store.get(COLLECTION_BATCHES, batch_id)
        if record is None:
            raise NotFoundError(f"material batch not found: {batch_id}", details={"batch_id": batch_id})
        return MaterialBatch.from_dict(record)

    def _save_batch(self, batch: MaterialBatch) -> None:
        self._store.put(COLLECTION_BATCHES, batch.batch_id, batch.to_dict())

    def _reservations_of(self, booking_id: str) -> list[MaterialReservation]:
        return [
            MaterialReservation.from_dict(r)
            for r in self._store.query(COLLECTION_RESERVATIONS, booking_id=booking_id)
        ]

    def _shipments_of(self, booking_id: str) -> list[Shipment]:
        return [Shipment.from_dict(s) for s in self._store.query(COLLECTION_SHIPMENTS, booking_id=booking_id)]

    def _load_shipment(self, shipment_id: str) -> Shipment:
        record = self._store.get(COLLECTION_SHIPMENTS, shipment_id)
        if record is None:
            raise NotFoundError(
                f"shipment not found: {shipment_id}", details={"shipment_id": shipment_id}
            )
        return Shipment.from_dict(record)

    def _save_shipment(self, shipment: Shipment) -> None:
        shipment.roll_up()
        self._store.put(COLLECTION_SHIPMENTS, shipment.shipment_id, shipment.to_dict())

    def _reservation_for(self, booking_id: str, batch_id: str) -> MaterialReservation | None:
        for reservation in self._reservations_of(booking_id):
            if reservation.batch_id == batch_id:
                return reservation
        return None

    def _save_reservation(self, reservation: MaterialReservation) -> None:
        self._store.put(COLLECTION_RESERVATIONS, reservation.reservation_id, reservation.to_dict())

    def _window_bookings(self, window_id: str) -> list[Booking]:
        return [Booking.from_dict(b) for b in self._store.query(COLLECTION_BOOKINGS, window_id=window_id)]

    def _all_bookings(self) -> list[Booking]:
        return [Booking.from_dict(b) for b in self._store.query(COLLECTION_BOOKINGS)]

    # ------------------------------------------------------------------
    # 申请
    # ------------------------------------------------------------------

    def apply(self, request: dict[str, Any]) -> dict[str, Any]:
        """申请预约：校验前置培训、容量、安全、运输周期并生成预约方案。"""
        key = request.get("idempotency_key")
        return self._idempotent("apply", key, request, lambda: self._apply(request), required=True)

    def _apply(self, request: dict[str, Any]) -> dict[str, Any]:
        from ..domain.models import dt_from_str

        now = self._clock.now()
        for field in ("package_id", "mentor_id", "resource_id", "window_id"):
            if not isinstance(request.get(field), str) or not request[field].strip():
                raise ValidationError(f"field {field} must be a non-empty string", details={"field": field})
        package = self._load_package(request["package_id"].strip())
        mentor = self._load_mentor(request["mentor_id"].strip())
        resource = self._load_resource(request["resource_id"].strip())
        window = self._load_window(request["window_id"].strip())
        institution = request.get("institution")
        if not isinstance(institution, str) or not institution.strip():
            raise ValidationError("field institution must be a non-empty string")
        seats = request.get("seats")
        if isinstance(seats, bool) or not isinstance(seats, int) or seats < 1:
            raise ValidationError("field seats must be a positive integer")
        if seats > package.max_seats:
            raise BusinessRuleError(
                "seats exceed package maximum",
                details={"max_seats": package.max_seats, "seats": seats},
            )
        try:
            slot_start = dt_from_str(request.get("slot_start"))
            slot_end = dt_from_str(request.get("slot_end"))
        except (ValueError, TypeError) as exc:
            raise ValidationError(f"invalid slot: {exc}") from exc
        if slot_start <= now:
            raise ValidationError("slot_start must be in the future")

        ensure_slot_shape(package, slot_start, slot_end)
        ensure_window_fit(window, slot_start, slot_end)
        ensure_mentor_qualified(mentor, package, slot_end)
        ensure_resource_fit(resource, seats)

        batches = [MaterialBatch.from_dict(b) for b in self._store.query(COLLECTION_BATCHES)]
        plan = plan_material_allocation(
            package,
            seats,
            batches,
            now=now,
            slot_start=slot_start,
            max_safety=safety_ceiling(resource, window),
        )

        # 容量与互斥：窗口满或互斥资源被持有时进入候补
        candidate = Booking(
            booking_id=self._ids.new_id("bkg"),
            institution=institution.strip(),
            package_id=package.package_id,
            mentor_id=mentor.mentor_id,
            resource_id=resource.resource_id,
            window_id=window.window_id,
            seats=seats,
            slot_start=slot_start,
            slot_end=slot_end,
            status=BookingStatus.REQUESTED,
            created_at=now,
            updated_at=now,
            material_plan=plan,
        )
        others = [(b, self._load_resource(b.resource_id)) for b in self._window_bookings(window.window_id)]
        active = [b for b, _ in others if b.status in WINDOW_OCCUPYING_STATUSES]
        waitlist_reason: str | None = None
        if len(active) >= window.capacity:
            waitlist_reason = "window_capacity_full"
        elif find_resource_conflict(candidate, resource, others, RESOURCE_HOLDING_STATUSES) is not None:
            waitlist_reason = "resource_mutex_blocked"
        if waitlist_reason:
            candidate.status = BookingStatus.WAITLISTED
            candidate.waitlist_reason = waitlist_reason
        self._store.put(COLLECTION_BOOKINGS, candidate.booking_id, candidate.to_dict())
        self._emit(
            "booking_waitlisted" if waitlist_reason else "booking_applied",
            candidate.booking_id,
            {
                "institution": candidate.institution,
                "seats": seats,
                "waitlist_reason": waitlist_reason,
            },
        )
        return self._booking_view(candidate)

    # ------------------------------------------------------------------
    # 报价
    # ------------------------------------------------------------------

    def quote(self, booking_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request or {}
        key = request.get("idempotency_key")
        return self._idempotent(
            "quote", key, {"booking_id": booking_id, **request}, lambda: self._quote(booking_id), required=False
        )

    def _quote(self, booking_id: str) -> dict[str, Any]:
        booking = self._load_booking(booking_id)
        if booking.status != BookingStatus.REQUESTED:
            raise StateError(
                "only a REQUESTED booking can be quoted",
                details={"booking_id": booking_id, "status": booking.status.value},
            )
        package = self._load_package(booking.package_id)
        mentor = self._load_mentor(booking.mentor_id)
        resource = self._load_resource(booking.resource_id)
        minutes = package.duration_minutes
        mentor_fee = (mentor.hourly_fee_cents * minutes + 59) // 60
        venue_fee = (resource.hourly_fee_cents * minutes + 59) // 60
        material_fee = 0
        for alloc in booking.material_plan:
            batch = self._load_batch(alloc.batch_id)
            material_fee += int(round(batch.unit_cost_cents * alloc.quantity))
        quote = Quote(
            quote_id=self._ids.new_id("quo"),
            mentor_fee_cents=mentor_fee,
            venue_fee_cents=venue_fee,
            material_fee_cents=material_fee,
            total_cents=mentor_fee + venue_fee + material_fee,
            currency="CNY",
            expires_at=self._clock.now() + timedelta(seconds=self._quote_ttl),
        )
        booking.quote = quote
        booking.status = BookingStatus.QUOTED
        self._save_booking(booking)
        self._emit("quote_issued", booking.booking_id, {"total_cents": quote.total_cents})
        return self._booking_view(booking)

    # ------------------------------------------------------------------
    # 锁定（并发敏感，幂等键防重复占位）
    # ------------------------------------------------------------------

    def lock(self, booking_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request or {}
        key = request.get("idempotency_key")
        return self._idempotent(
            "lock",
            key,
            {"booking_id": booking_id, **request},
            lambda: self._lock(booking_id, request),
            required=True,
        )

    def _lock(self, booking_id: str, request: dict[str, Any]) -> dict[str, Any]:
        now = self._clock.now()
        booking = self._load_booking(booking_id)
        if booking.status != BookingStatus.QUOTED:
            raise StateError(
                "only a QUOTED booking can be locked",
                details={"booking_id": booking_id, "status": booking.status.value},
            )
        if booking.quote is None or booking.quote.expires_at <= now:
            raise StateError("quote has expired; request a new quote", details={"booking_id": booking_id})
        ttl = request.get("ttl_seconds", self._lock_ttl)
        if isinstance(ttl, bool) or not isinstance(ttl, int) or not (MIN_LOCK_TTL_SECONDS <= ttl <= MAX_LOCK_TTL_SECONDS):
            raise ValidationError(
                "ttl_seconds out of range",
                details={"min": MIN_LOCK_TTL_SECONDS, "max": MAX_LOCK_TTL_SECONDS},
            )

        # 互斥资源最终判定：同事务内复查，保证并发锁定只有一个成功者
        resource = self._load_resource(booking.resource_id)
        others = [
            (b, self._load_resource(b.resource_id))
            for b in self._all_bookings()
            if b.booking_id != booking.booking_id
        ]
        conflict = find_resource_conflict(booking, resource, others, RESOURCE_HOLDING_STATUSES)
        if conflict is not None:
            raise ConflictError(
                "resource is held by a conflicting booking",
                details={"conflict_booking_id": conflict.booking_id, "resource_id": resource.resource_id},
            )

        # 依据当前库存重新生成分配计划并预占
        package = self._load_package(booking.package_id)
        window = self._load_window(booking.window_id)
        batches = [MaterialBatch.from_dict(b) for b in self._store.query(COLLECTION_BATCHES)]
        plan = plan_material_allocation(
            package,
            booking.seats,
            batches,
            now=now,
            slot_start=booking.slot_start,
            max_safety=safety_ceiling(resource, window),
        )
        by_id = {b.batch_id: b for b in batches}
        for alloc in plan:
            batch = by_id[alloc.batch_id]
            batch.available_quantity = round(batch.available_quantity - alloc.quantity, 6)
            self._save_batch(batch)
            reservation = MaterialReservation(
                reservation_id=self._ids.new_id("rsv"),
                booking_id=booking.booking_id,
                batch_id=alloc.batch_id,
                material_id=alloc.material_id,
                quantity_reserved=alloc.quantity,
            )
            self._store.put(COLLECTION_RESERVATIONS, reservation.reservation_id, reservation.to_dict())

        booking.material_plan = plan
        booking.status = BookingStatus.LOCKED
        booking.lock_expires_at = now + timedelta(seconds=ttl)
        self._save_booking(booking)
        self._emit(
            "booking_locked",
            booking.booking_id,
            {"lock_expires_at": dt_to_str(booking.lock_expires_at)},
        )
        return self._booking_view(booking)

    # ------------------------------------------------------------------
    # 改期（已发运不可移动）
    # ------------------------------------------------------------------

    def reschedule(self, booking_id: str, request: dict[str, Any]) -> dict[str, Any]:
        key = request.get("idempotency_key")
        return self._idempotent(
            "reschedule",
            key,
            {"booking_id": booking_id, **request},
            lambda: self._reschedule(booking_id, request),
            required=False,
        )

    def _reschedule(self, booking_id: str, request: dict[str, Any]) -> dict[str, Any]:
        from ..domain.models import dt_from_str

        now = self._clock.now()
        booking = self._load_booking(booking_id)
        if booking.status in (BookingStatus.SHIPPED, BookingStatus.CHECKED_IN, BookingStatus.SETTLED):
            raise BookingImmutableError(
                "booking cannot be moved after materials have shipped",
                details={"booking_id": booking_id, "status": booking.status.value},
            )
        if booking.status in (BookingStatus.CANCELLED, BookingStatus.EXPIRED, BookingStatus.WAITLISTED):
            raise StateError(
                "booking in current status cannot be rescheduled",
                details={"booking_id": booking_id, "status": booking.status.value},
            )
        try:
            slot_start = dt_from_str(request.get("slot_start"))
            slot_end = dt_from_str(request.get("slot_end"))
        except (ValueError, TypeError) as exc:
            raise ValidationError(f"invalid slot: {exc}") from exc
        if slot_start <= now:
            raise ValidationError("slot_start must be in the future")

        package = self._load_package(booking.package_id)
        mentor = self._load_mentor(booking.mentor_id)
        resource = self._load_resource(booking.resource_id)
        window = self._load_window(booking.window_id)
        ensure_slot_shape(package, slot_start, slot_end)
        ensure_window_fit(window, slot_start, slot_end)
        ensure_mentor_qualified(mentor, package, slot_end)
        batches = [MaterialBatch.from_dict(b) for b in self._store.query(COLLECTION_BATCHES)]
        plan = plan_material_allocation(
            package,
            booking.seats,
            batches,
            now=now,
            slot_start=slot_start,
            max_safety=safety_ceiling(resource, window),
        )

        # 已锁定的先释放原预占，回到待报价重新走流程
        if booking.status == BookingStatus.LOCKED:
            self._release_reservations(booking)
        booking.slot_start = slot_start
        booking.slot_end = slot_end
        booking.material_plan = plan
        booking.quote = None
        booking.lock_expires_at = None
        booking.status = BookingStatus.REQUESTED
        self._save_booking(booking)
        self._emit(
            "booking_rescheduled",
            booking.booking_id,
            {"slot_start": dt_to_str(slot_start), "slot_end": dt_to_str(slot_end)},
        )
        return self._booking_view(booking)

    # ------------------------------------------------------------------
    # 发运与到货
    # ------------------------------------------------------------------

    def ship(self, booking_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request or {}
        key = request.get("idempotency_key")
        return self._idempotent(
            "ship", key, {"booking_id": booking_id, **request}, lambda: self._ship(booking_id, request), required=True
        )

    def _ship(self, booking_id: str, request: dict[str, Any]) -> dict[str, Any]:
        now = self._clock.now()
        booking = self._load_booking(booking_id)
        if booking.status != BookingStatus.LOCKED:
            raise StateError(
                "only a LOCKED booking can be shipped",
                details={"booking_id": booking_id, "status": booking.status.value},
            )
        if booking.lock_expires_at is None or booking.lock_expires_at <= now:
            raise StateError("lock has expired; recover timeouts before shipping", details={"booking_id": booking_id})

        reservations = [r for r in self._reservations_of(booking_id) if r.outstanding_reserved > QTY_EPS]
        if not reservations:
            raise StateError("nothing to ship for this booking", details={"booking_id": booking_id})
        splits = self._build_ship_splits(request.get("parts"), reservations)

        shipments: list[Shipment] = []
        for reservation in reservations:
            batch = self._load_batch(reservation.batch_id)
            shipment_id = self._ids.new_id("shp")
            quantities = splits.get(reservation.batch_id, [reservation.outstanding_reserved])
            parts: list[ShipmentPart] = []
            for quantity in quantities:
                parts.append(
                    ShipmentPart(
                        part_id=self._ids.new_id("prt"),
                        tracking_number=self._tracking_numbers.new_tracking_number(shipment_id, 1),
                        quantity=round(quantity, 6),
                        shipped_at=now,
                        eta=now + timedelta(seconds=batch.lead_time_seconds),
                    )
                )
            total = round(sum(quantities), 6)
            shipment = Shipment(
                shipment_id=shipment_id,
                booking_id=booking_id,
                batch_id=batch.batch_id,
                material_id=batch.material_id,
                quantity=total,
                shipped_at=now,
                eta=now + timedelta(seconds=batch.lead_time_seconds),
                parts=parts,
            )
            self._save_shipment(shipment)
            reservation.quantity_shipped = round(reservation.quantity_shipped + total, 6)
            self._save_reservation(reservation)
            shipments.append(shipment)
        booking.status = BookingStatus.SHIPPED
        booking.lock_expires_at = None
        self._save_booking(booking)
        self._emit(
            "materials_shipped",
            booking_id,
            {
                "shipment_ids": [s.shipment_id for s in shipments],
                "parts": [
                    {"shipment_id": s.shipment_id, "part_id": p.part_id, "tracking_number": p.tracking_number}
                    for s in shipments
                    for p in s.parts
                ],
            },
        )
        return self._booking_view(booking)

    def _build_ship_splits(
        self,
        raw_parts: Any,
        reservations: list[MaterialReservation],
    ) -> dict[str, list[float]]:
        """解析并校验拆批载荷：每个批次可拆成若干正数数量，合计须等于待发运量。"""
        outstanding = {r.batch_id: round(r.outstanding_reserved, 6) for r in reservations}
        if raw_parts is None:
            return {batch_id: [quantity] for batch_id, quantity in outstanding.items()}
        if not isinstance(raw_parts, list) or not raw_parts:
            raise ValidationError("field parts must be a non-empty list of {batch_id, quantity}")
        splits: dict[str, list[float]] = {}
        for item in raw_parts:
            if not isinstance(item, dict):
                raise ValidationError("each shipping part must be an object {batch_id, quantity}")
            batch_id = item.get("batch_id")
            if not isinstance(batch_id, str) or not batch_id.strip():
                raise ValidationError("each shipping part requires a non-empty batch_id")
            quantity = item.get("quantity")
            if isinstance(quantity, bool) or not isinstance(quantity, (int, float)) or float(quantity) <= 0:
                raise ValidationError("each shipping part quantity must be a positive number")
            if batch_id not in outstanding:
                raise ValidationError(
                    "part references a batch without outstanding reservation",
                    details={"batch_id": batch_id},
                )
            splits.setdefault(batch_id, []).append(float(quantity))
        for batch_id, quantities in splits.items():
            total = round(sum(quantities), 6)
            if abs(total - outstanding[batch_id]) > QTY_EPS:
                raise ValidationError(
                    "part quantities must sum to the outstanding reserved quantity",
                    details={"batch_id": batch_id, "expected": outstanding[batch_id], "actual": total},
                )
        missing = [batch_id for batch_id in outstanding if batch_id not in splits]
        if missing:
            raise ValidationError(
                "parts must cover every batch with outstanding reservation",
                details={"missing_batch_ids": missing},
            )
        return splits

    def record_arrival(self, shipment_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """登记某个追踪批次的到货，支持部分到货；其余批次不受影响。"""
        key = request.get("idempotency_key")
        return self._idempotent(
            "record_arrival",
            key,
            {"shipment_id": shipment_id, **request},
            lambda: self._record_arrival(shipment_id, request),
            required=False,
        )

    def _select_active_part(self, shipment: Shipment, request: dict[str, Any]) -> ShipmentPart:
        part_id = request.get("part_id")
        tracking = request.get("tracking_number")
        if part_id is not None or tracking is not None:
            part = shipment.find_part(part_id=part_id, tracking_number=tracking)
            if part is None:
                raise NotFoundError(
                    "shipping part not found on this shipment",
                    details={"shipment_id": shipment.shipment_id, "part_id": part_id, "tracking_number": tracking},
                )
            if part.status not in ACTIVE_PART_STATUSES:
                raise StateError(
                    "shipping part is not accepting arrivals",
                    details={"part_id": part.part_id, "status": part.status.value},
                )
            return part
        active = shipment.active_parts()
        if not active:
            raise StateError(
                "no shipping part is in transit for this shipment",
                details={"shipment_id": shipment.shipment_id},
            )
        if len(active) > 1:
            raise ValidationError(
                "shipment has multiple in-transit parts; specify part_id or tracking_number",
                details={"part_ids": [p.part_id for p in active]},
            )
        return active[0]

    def _record_arrival(self, shipment_id: str, request: dict[str, Any]) -> dict[str, Any]:
        shipment = self._load_shipment(shipment_id)
        part = self._select_active_part(shipment, request)
        quantity = request.get("quantity")
        if isinstance(quantity, bool) or not isinstance(quantity, (int, float)) or float(quantity) <= 0:
            raise ValidationError("field quantity must be a positive number")
        quantity = float(quantity)
        if quantity > part.remaining + QTY_EPS:
            raise ValidationError(
                "arrival quantity exceeds this part's remainder",
                details={"part_id": part.part_id, "remaining": part.remaining, "quantity": quantity},
            )
        part.arrived_quantity = round(part.arrived_quantity + quantity, 6)
        part.status = (
            ShipmentPartStatus.ARRIVED if part.remaining <= QTY_EPS else ShipmentPartStatus.PARTIALLY_ARRIVED
        )
        self._save_shipment(shipment)
        reservation = self._reservation_for(shipment.booking_id, shipment.batch_id)
        if reservation is not None:
            reservation.quantity_arrived = round(reservation.quantity_arrived + quantity, 6)
            self._save_reservation(reservation)
        self._emit(
            "shipment_partially_arrived"
            if shipment.status == ShipmentStatus.PARTIALLY_ARRIVED
            else "shipment_arrived",
            shipment.booking_id,
            {
                "shipment_id": shipment_id,
                "part_id": part.part_id,
                "tracking_number": part.tracking_number,
                "quantity": quantity,
            },
        )
        return self._booking_view(self._load_booking(shipment.booking_id))

    def report_part_failure(self, shipment_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """仓储人员上报某个追踪批次发运失败（丢件/退件），等待单独重试。"""
        key = request.get("idempotency_key")
        return self._idempotent(
            "report_part_failure",
            key,
            {"shipment_id": shipment_id, **request},
            lambda: self._report_part_failure(shipment_id, request),
            required=True,
        )

    def _report_part_failure(self, shipment_id: str, request: dict[str, Any]) -> dict[str, Any]:
        shipment = self._load_shipment(shipment_id)
        part = self._select_part_for_retry(shipment, request)
        if part.status not in ACTIVE_PART_STATUSES:
            raise StateError(
                "only an in-transit part can be marked failed",
                details={"part_id": part.part_id, "status": part.status.value},
            )
        part.status = ShipmentPartStatus.FAILED
        self._save_shipment(shipment)
        self._emit(
            "shipment_part_failed",
            shipment.booking_id,
            {
                "shipment_id": shipment_id,
                "part_id": part.part_id,
                "tracking_number": part.tracking_number,
                "remaining": part.remaining,
                "reason": request.get("reason"),
            },
        )
        return self._booking_view(self._load_booking(shipment.booking_id))

    def _select_part_for_loss(self, shipment: Shipment, request: dict[str, Any]) -> ShipmentPart:
        """损耗可登记到在途批，或显式登记到失败批（确认灭失、不再补发）。"""
        part_id = request.get("part_id")
        tracking = request.get("tracking_number")
        if part_id is not None or tracking is not None:
            part = shipment.find_part(part_id=part_id, tracking_number=tracking)
            if part is None:
                raise NotFoundError(
                    "shipping part not found on this shipment",
                    details={"shipment_id": shipment.shipment_id, "part_id": part_id, "tracking_number": tracking},
                )
            if part.status not in ACTIVE_PART_STATUSES and part.status != ShipmentPartStatus.FAILED:
                raise StateError(
                    "shipping part is already closed",
                    details={"part_id": part.part_id, "status": part.status.value},
                )
            if part.superseded_by is not None:
                raise StateError(
                    "shipping part was superseded by a retry; record loss on the retry part",
                    details={"part_id": part.part_id, "superseded_by": part.superseded_by},
                )
            return part
        return self._select_active_part(shipment, request)

    def _select_part_for_retry(self, shipment: Shipment, request: dict[str, Any]) -> ShipmentPart:
        part_id = request.get("part_id")
        tracking = request.get("tracking_number")
        if part_id is None and tracking is None:
            raise ValidationError("part_id or tracking_number is required")
        part = shipment.find_part(part_id=part_id, tracking_number=tracking)
        if part is None:
            raise NotFoundError(
                "shipping part not found on this shipment",
                details={"shipment_id": shipment.shipment_id, "part_id": part_id, "tracking_number": tracking},
            )
        return part

    def retry_part(self, shipment_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """对单个失败批次独立重试：旧批次保留审计痕迹，新批次获得新追踪号。

        失败批次若已有部分到货，仅重试其剩余量；整批失败则整量重发。
        """
        key = request.get("idempotency_key")
        return self._idempotent(
            "retry_part",
            key,
            {"shipment_id": shipment_id, **request},
            lambda: self._retry_part(shipment_id, request),
            required=True,
        )

    def _retry_part(self, shipment_id: str, request: dict[str, Any]) -> dict[str, Any]:
        now = self._clock.now()
        shipment = self._load_shipment(shipment_id)
        failed = self._select_part_for_retry(shipment, request)
        if failed.status != ShipmentPartStatus.FAILED:
            raise StateError(
                "only a FAILED part can be retried",
                details={"part_id": failed.part_id, "status": failed.status.value},
            )
        retry_quantity = round(failed.remaining, 6)
        if retry_quantity <= QTY_EPS:
            raise StateError(
                "failed part has no quantity left to retry",
                details={"part_id": failed.part_id},
            )
        batch = self._load_batch(shipment.batch_id)
        attempt = max(p.attempt for p in shipment.parts) + 1
        new_part = ShipmentPart(
            part_id=self._ids.new_id("prt"),
            tracking_number=self._tracking_numbers.new_tracking_number(shipment_id, attempt),
            quantity=retry_quantity,
            shipped_at=now,
            eta=now + timedelta(seconds=batch.lead_time_seconds),
            attempt=attempt,
        )
        failed.superseded_by = new_part.part_id
        shipment.parts.append(new_part)
        self._save_shipment(shipment)
        self._emit(
            "shipment_part_retried",
            shipment.booking_id,
            {
                "shipment_id": shipment_id,
                "failed_part_id": failed.part_id,
                "failed_tracking_number": failed.tracking_number,
                "new_part_id": new_part.part_id,
                "new_tracking_number": new_part.tracking_number,
                "quantity": retry_quantity,
                "attempt": attempt,
            },
        )
        return self._booking_view(self._load_booking(shipment.booking_id))

    def record_shipment_loss(self, shipment_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """登记某个追踪批次的在途损耗；剩余全部灭失时关闭该批与发运单。"""
        key = request.get("idempotency_key")
        return self._idempotent(
            "record_shipment_loss",
            key,
            {"shipment_id": shipment_id, **request},
            lambda: self._record_shipment_loss(shipment_id, request),
            required=False,
        )

    def _record_shipment_loss(self, shipment_id: str, request: dict[str, Any]) -> dict[str, Any]:
        shipment = self._load_shipment(shipment_id)
        part = self._select_part_for_loss(shipment, request)
        quantity = request.get("quantity", part.remaining)
        if isinstance(quantity, bool) or not isinstance(quantity, (int, float)) or float(quantity) <= 0:
            raise ValidationError("field quantity must be a positive number")
        quantity = float(quantity)
        if quantity > part.remaining + QTY_EPS:
            raise ValidationError(
                "loss quantity exceeds this part's remainder",
                details={"part_id": part.part_id, "remaining": part.remaining, "quantity": quantity},
            )
        part.lost_quantity = round(part.lost_quantity + quantity, 6)
        if part.remaining <= QTY_EPS:
            part.status = ShipmentPartStatus.CLOSED_WITH_LOSS
        self._save_shipment(shipment)
        reservation = self._reservation_for(shipment.booking_id, shipment.batch_id)
        if reservation is not None:
            reservation.quantity_lost = round(reservation.quantity_lost + quantity, 6)
            self._save_reservation(reservation)
        self._record_loss(
            booking_id=shipment.booking_id,
            batch_id=shipment.batch_id,
            material_id=shipment.material_id,
            quantity=quantity,
            reason=LOSS_IN_TRANSIT,
        )
        return self._booking_view(self._load_booking(shipment.booking_id))

    # ------------------------------------------------------------------
    # 签到与结算
    # ------------------------------------------------------------------

    def checkin(self, booking_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request or {}
        key = request.get("idempotency_key")
        return self._idempotent(
            "checkin", key, {"booking_id": booking_id, **request}, lambda: self._checkin(booking_id), required=False
        )

    def _checkin(self, booking_id: str) -> dict[str, Any]:
        now = self._clock.now()
        booking = self._load_booking(booking_id)
        if booking.status != BookingStatus.SHIPPED:
            raise StateError(
                "only a SHIPPED booking can be checked in",
                details={"booking_id": booking_id, "status": booking.status.value},
            )
        if now < booking.slot_start:
            raise StateError(
                "check-in is not allowed before the slot starts",
                details={"slot_start": dt_to_str(booking.slot_start), "now": dt_to_str(now)},
            )
        shipments = self._shipments_of(booking_id)
        open_shipments = [s.shipment_id for s in shipments if not s.closed]
        if open_shipments:
            raise StateError(
                "cannot check in while shipments are still in transit",
                details={"open_shipments": open_shipments},
            )
        package = self._load_package(booking.package_id)
        arrived_by_material: dict[str, float] = {}
        for reservation in self._reservations_of(booking_id):
            arrived_by_material[reservation.material_id] = arrived_by_material.get(reservation.material_id, 0.0) + (
                reservation.quantity_arrived
            )
        shortages = {
            req.material_id: package.required_quantity(req.material_id, booking.seats)
            for req in package.materials
            if arrived_by_material.get(req.material_id, 0.0) + QTY_EPS
            < package.required_quantity(req.material_id, booking.seats)
        }
        if shortages:
            raise BusinessRuleError(
                "arrived materials are insufficient for the booked seats",
                details={"shortages": shortages},
            )
        booking.status = BookingStatus.CHECKED_IN
        self._save_booking(booking)
        self._emit("booking_checked_in", booking_id, {})
        return self._booking_view(booking)

    def settle(self, booking_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request or {}
        key = request.get("idempotency_key")
        return self._idempotent(
            "settle",
            key,
            {"booking_id": booking_id, **request},
            lambda: self._settle(booking_id, request),
            required=False,
        )

    def _settle(self, booking_id: str, request: dict[str, Any]) -> dict[str, Any]:
        now = self._clock.now()
        booking = self._load_booking(booking_id)
        if booking.status != BookingStatus.CHECKED_IN:
            raise StateError(
                "only a CHECKED_IN booking can be settled",
                details={"booking_id": booking_id, "status": booking.status.value},
            )
        attendance = request.get("actual_attendance")
        if isinstance(attendance, bool) or not isinstance(attendance, int) or not (0 <= attendance <= booking.seats):
            raise ValidationError(
                "actual_attendance must be an integer between 0 and booked seats",
                details={"seats": booking.seats},
            )
        damaged_raw = request.get("damaged", {})
        if not isinstance(damaged_raw, dict):
            raise ValidationError("field damaged must be an object of material_id -> quantity")
        damaged_remaining = {str(k): float(v) for k, v in damaged_raw.items()}
        for material_id, qty in damaged_remaining.items():
            if qty < 0:
                raise ValidationError("damaged quantities must be non-negative", details={"material_id": material_id})

        package = self._load_package(booking.package_id)
        mentor = self._load_mentor(booking.mentor_id)
        resource = self._load_resource(booking.resource_id)
        ratio = attendance / booking.seats if booking.seats else 0.0
        material_fee = 0
        loss_fee = 0
        for reservation in self._reservations_of(booking_id):
            batch = self._load_batch(reservation.batch_id)
            consumed = round(reservation.quantity_reserved * ratio, 6)
            if consumed > reservation.quantity_arrived + QTY_EPS:
                raise BusinessRuleError(
                    "actual attendance exceeds what arrived materials can serve",
                    details={"material_id": reservation.material_id, "arrived": reservation.quantity_arrived},
                )
            # 同种材料可能跨多个批次，损坏量按批次依次分摊
            damaged_qty = round(
                min(damaged_remaining.get(reservation.material_id, 0.0), reservation.quantity_arrived - consumed),
                6,
            )
            damaged_qty = max(damaged_qty, 0.0)
            damaged_remaining[reservation.material_id] = round(
                damaged_remaining.get(reservation.material_id, 0.0) - damaged_qty, 6
            )
            leftover = round(reservation.quantity_arrived - consumed - damaged_qty, 6)
            if leftover < -QTY_EPS:
                raise ValidationError(
                    "damaged quantity exceeds available leftover",
                    details={"material_id": reservation.material_id},
                )
            leftover = max(leftover, 0.0)
            reservation.quantity_consumed = round(reservation.quantity_consumed + consumed, 6)
            material_fee += int(round(batch.unit_cost_cents * consumed))
            if damaged_qty > QTY_EPS:
                reservation.quantity_lost = round(reservation.quantity_lost + damaged_qty, 6)
                loss_fee += int(round(batch.unit_cost_cents * damaged_qty))
                self._record_loss(
                    booking_id=booking_id,
                    batch_id=batch.batch_id,
                    material_id=batch.material_id,
                    quantity=damaged_qty,
                    reason=LOSS_DAMAGED_IN_USE,
                )
            if leftover > QTY_EPS:
                if batch.cross_border:
                    # 跨境余料退回不经济，记损耗
                    reservation.quantity_lost = round(reservation.quantity_lost + leftover, 6)
                    loss_fee += int(round(batch.unit_cost_cents * leftover))
                    self._record_loss(
                        booking_id=booking_id,
                        batch_id=batch.batch_id,
                        material_id=batch.material_id,
                        quantity=leftover,
                        reason=LOSS_NON_RETURNABLE_LEFTOVER,
                    )
                else:
                    reservation.quantity_returned = round(reservation.quantity_returned + leftover, 6)
                    batch.available_quantity = round(batch.available_quantity + leftover, 6)
                    self._save_batch(batch)
            self._store.put(COLLECTION_RESERVATIONS, reservation.reservation_id, reservation.to_dict())

        unallocated_damage = {m: q for m, q in damaged_remaining.items() if q > QTY_EPS}
        if unallocated_damage:
            raise ValidationError(
                "damaged quantities reference materials beyond this booking",
                details={"unallocated": unallocated_damage},
            )

        minutes = package.duration_minutes
        mentor_fee = (mentor.hourly_fee_cents * minutes + 59) // 60
        venue_fee = (resource.hourly_fee_cents * minutes + 59) // 60
        settlement = Settlement(
            settlement_id=self._ids.new_id("stl"),
            booking_id=booking_id,
            actual_attendance=attendance,
            mentor_fee_cents=mentor_fee,
            venue_fee_cents=venue_fee,
            material_fee_cents=material_fee,
            loss_fee_cents=loss_fee,
            total_cents=mentor_fee + venue_fee + material_fee + loss_fee,
            currency="CNY",
            settled_at=now,
        )
        self._store.put(COLLECTION_SETTLEMENTS, settlement.settlement_id, settlement.to_dict())
        booking.status = BookingStatus.SETTLED
        self._save_booking(booking)
        self._emit("booking_settled", booking_id, {"total_cents": settlement.total_cents})
        return self._booking_view(booking)

    # ------------------------------------------------------------------
    # 取消与候补释放
    # ------------------------------------------------------------------

    def cancel(self, booking_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request or {}
        key = request.get("idempotency_key")
        return self._idempotent(
            "cancel",
            key,
            {"booking_id": booking_id, **request},
            lambda: self._cancel(booking_id, request),
            required=False,
        )

    def _cancel(self, booking_id: str, request: dict[str, Any]) -> dict[str, Any]:
        booking = self._load_booking(booking_id)
        if booking.status not in CANCELLABLE_STATUSES:
            raise StateError(
                "booking in current status cannot be cancelled",
                details={"booking_id": booking_id, "status": booking.status.value},
            )
        reason = request.get("reason")
        if booking.status == BookingStatus.LOCKED:
            self._release_reservations(booking)
        elif booking.status == BookingStatus.SHIPPED:
            self._write_off_shipped_materials(booking)
        booking.status = BookingStatus.CANCELLED
        booking.lock_expires_at = None
        self._save_booking(booking)
        self._emit("booking_cancelled", booking_id, {"reason": reason})
        # 容量/互斥资源可能已释放，按规则尝试晋级候补（无候补时为 no-op）
        self._promote_waitlist(booking.window_id)
        return self._booking_view(booking)

    def _release_reservations(self, booking: Booking) -> None:
        """释放未发运的预占库存。"""
        for reservation in self._reservations_of(booking.booking_id):
            outstanding = reservation.outstanding_reserved
            if outstanding <= QTY_EPS:
                continue
            batch = self._load_batch(reservation.batch_id)
            batch.available_quantity = round(batch.available_quantity + outstanding, 6)
            self._save_batch(batch)
            reservation.quantity_released = round(reservation.quantity_released + outstanding, 6)
            self._store.put(COLLECTION_RESERVATIONS, reservation.reservation_id, reservation.to_dict())

    def _write_off_shipped_materials(self, booking: Booking) -> None:
        """发运后取消：已发运（在途+已到货）材料全部记损耗。"""
        for reservation in self._reservations_of(booking.booking_id):
            unshipped = reservation.outstanding_reserved
            if unshipped > QTY_EPS:
                batch = self._load_batch(reservation.batch_id)
                batch.available_quantity = round(batch.available_quantity + unshipped, 6)
                self._save_batch(batch)
                reservation.quantity_released = round(reservation.quantity_released + unshipped, 6)
            shipped_uncounted = round(
                reservation.quantity_shipped - reservation.quantity_lost - reservation.quantity_consumed,
                6,
            )
            if shipped_uncounted > QTY_EPS:
                reservation.quantity_lost = round(reservation.quantity_lost + shipped_uncounted, 6)
                self._record_loss(
                    booking_id=booking.booking_id,
                    batch_id=reservation.batch_id,
                    material_id=reservation.material_id,
                    quantity=shipped_uncounted,
                    reason=LOSS_CANCEL_AFTER_SHIPMENT,
                )
            self._store.put(COLLECTION_RESERVATIONS, reservation.reservation_id, reservation.to_dict())
        # 关闭仍未终结的分批：在途批与未被重试取代的失败批，其剩余记损耗；
        # 已被新批次取代（superseded_by）的失败批，剩余量由后继批次承担，不重复计。
        for shipment in self._shipments_of(booking.booking_id):
            changed = False
            for part in shipment.parts:
                if part.terminal or part.superseded_by is not None:
                    continue
                remaining = part.remaining
                if remaining > QTY_EPS:
                    part.lost_quantity = round(part.lost_quantity + remaining, 6)
                part.status = ShipmentPartStatus.CLOSED_WITH_LOSS
                changed = True
            if changed:
                self._save_shipment(shipment)

    def _promote_waitlist(self, window_id: str) -> None:
        """按申请先后顺序释放候补：容量与互斥均满足者晋级为 REQUESTED。"""
        window = self._load_window(window_id)
        waiting = sorted(
            (b for b in self._window_bookings(window_id) if b.status == BookingStatus.WAITLISTED),
            key=lambda b: (b.created_at, b.booking_id),
        )
        if not waiting:
            return
        for candidate in waiting:
            others = [
                (b, self._load_resource(b.resource_id))
                for b in self._window_bookings(window_id)
                if b.booking_id != candidate.booking_id
            ]
            active = [b for b, _ in others if b.status in WINDOW_OCCUPYING_STATUSES]
            if len(active) >= window.capacity:
                continue
            resource = self._load_resource(candidate.resource_id)
            if find_resource_conflict(candidate, resource, others, RESOURCE_HOLDING_STATUSES) is not None:
                continue
            candidate.status = BookingStatus.REQUESTED
            candidate.waitlist_reason = None
            self._save_booking(candidate)
            self._emit("waitlist_promoted", candidate.booking_id, {"window_id": window_id})

    def _record_loss(self, *, booking_id: str | None, batch_id: str, material_id: str, quantity: float, reason: str) -> None:
        loss = LossRecord(
            loss_id=self._ids.new_id("los"),
            booking_id=booking_id,
            batch_id=batch_id,
            material_id=material_id,
            quantity=round(quantity, 6),
            reason=reason,
            recorded_at=self._clock.now(),
        )
        self._store.put(COLLECTION_LOSSES, loss.loss_id, loss.to_dict())
        self._emit(
            "material_loss_recorded",
            booking_id,
            {"loss_id": loss.loss_id, "batch_id": batch_id, "quantity": loss.quantity, "reason": reason},
        )

    # ------------------------------------------------------------------
    # 超时恢复（服务重启后调用）
    # ------------------------------------------------------------------

    def recover(self) -> dict[str, Any]:
        """恢复超时任务：过期锁定释放库存并晋级候补，过期报价退回待报价。"""
        now = self._clock.now()
        expired_locks: list[str] = []
        expired_quotes: list[str] = []
        with self._store.transaction():
            affected_windows: set[str] = set()
            for booking in self._all_bookings():
                if (
                    booking.status == BookingStatus.LOCKED
                    and booking.lock_expires_at is not None
                    and booking.lock_expires_at <= now
                ):
                    self._release_reservations(booking)
                    booking.status = BookingStatus.EXPIRED
                    booking.lock_expires_at = None
                    self._save_booking(booking)
                    self._emit("lock_expired", booking.booking_id, {})
                    expired_locks.append(booking.booking_id)
                    affected_windows.add(booking.window_id)
                elif (
                    booking.status == BookingStatus.QUOTED
                    and booking.quote is not None
                    and booking.quote.expires_at <= now
                ):
                    booking.quote = None
                    booking.status = BookingStatus.REQUESTED
                    self._save_booking(booking)
                    self._emit("quote_expired", booking.booking_id, {})
                    expired_quotes.append(booking.booking_id)
            for window_id in affected_windows:
                self._promote_waitlist(window_id)
        return {
            "expired_locks": expired_locks,
            "expired_quotes": expired_quotes,
            "recovered_at": dt_to_str(now),
        }

    # ------------------------------------------------------------------
    # 查询视图
    # ------------------------------------------------------------------

    def get_booking(self, booking_id: str) -> dict[str, Any]:
        booking = self._load_booking(booking_id)
        return self._booking_view(booking)

    def get_shipment(self, shipment_id: str) -> dict[str, Any]:
        """单批发运单视图：各追踪批次状态与汇总到货。"""
        shipment = self._load_shipment(shipment_id)
        return self._shipment_view(shipment)

    def list_bookings(self, **filters: Any) -> list[dict[str, Any]]:
        return [self._booking_view(Booking.from_dict(b)) for b in self._store.query(COLLECTION_BOOKINGS, **filters)]

    def _shipment_view(self, shipment: Shipment) -> dict[str, Any]:
        view = shipment.to_dict()
        view["arrival_summary"] = {
            "quantity": shipment.quantity,
            "arrived_quantity": shipment.arrived_quantity,
            "lost_quantity": shipment.lost_quantity,
            "in_transit_quantity": round(
                sum(
                    p.remaining
                    for p in shipment.parts
                    if p.status in ACTIVE_PART_STATUSES
                ),
                6,
            ),
            "failed_quantity": round(
                sum(
                    p.remaining
                    for p in shipment.parts
                    if p.status == ShipmentPartStatus.FAILED and p.superseded_by is None
                ),
                6,
            ),
            "part_count": len(shipment.parts),
        }
        return view

    def _booking_view(self, booking: Booking) -> dict[str, Any]:
        view = booking.to_dict()
        view["reservations"] = [r.to_dict() for r in self._reservations_of(booking.booking_id)]
        shipments = self._shipments_of(booking.booking_id)
        view["shipments"] = [self._shipment_view(s) for s in shipments]
        view["arrival_summary"] = self._arrival_summary_from(shipments)
        settlements = self._store.query(COLLECTION_SETTLEMENTS, booking_id=booking.booking_id)
        view["settlement"] = settlements[0] if settlements else None
        losses = self._store.query(COLLECTION_LOSSES, booking_id=booking.booking_id)
        view["losses"] = losses
        events = [e for e in self._store.query(COLLECTION_EVENTS, booking_id=booking.booking_id)]
        events.sort(key=lambda e: (e["created_at"], e["event_id"]))
        view["events"] = events
        if booking.status == BookingStatus.WAITLISTED:
            ahead = [
                b
                for b in self._window_bookings(booking.window_id)
                if b.status == BookingStatus.WAITLISTED
                and (b.created_at, b.booking_id) < (booking.created_at, booking.booking_id)
            ]
            view["waitlist_position"] = len(ahead) + 1
        return view

    def _arrival_summary_from(self, shipments: list[Shipment]) -> dict[str, Any]:
        summary = {
            "shipment_count": len(shipments),
            "part_count": sum(len(s.parts) for s in shipments),
            "open_shipment_ids": [s.shipment_id for s in shipments if not s.closed],
            "by_material": [],
            "totals": {},
        }
        by_material: dict[str, dict[str, Any]] = {}
        totals = {
            "shipped_quantity": 0.0,
            "arrived_quantity": 0.0,
            "lost_quantity": 0.0,
            "in_transit_quantity": 0.0,
            "failed_quantity": 0.0,
        }
        for shipment in shipments:
            row = by_material.setdefault(
                shipment.material_id,
                {"material_id": shipment.material_id, "shipped_quantity": 0.0, "arrived_quantity": 0.0,
                 "lost_quantity": 0.0, "in_transit_quantity": 0.0, "failed_quantity": 0.0},
            )
            in_transit = round(
                sum(p.remaining for p in shipment.parts if p.status in ACTIVE_PART_STATUSES), 6
            )
            failed = round(
                sum(
                    p.remaining
                    for p in shipment.parts
                    if p.status == ShipmentPartStatus.FAILED and p.superseded_by is None
                ),
                6,
            )
            deltas = {
                "shipped_quantity": shipment.quantity,
                "arrived_quantity": shipment.arrived_quantity,
                "lost_quantity": shipment.lost_quantity,
                "in_transit_quantity": in_transit,
                "failed_quantity": failed,
            }
            for key, value in deltas.items():
                row[key] = round(row[key] + value, 6)
                totals[key] = round(totals[key] + value, 6)
        summary["by_material"] = [by_material[key] for key in sorted(by_material)]
        summary["totals"] = totals
        summary["all_arrived"] = (
            bool(shipments) and not summary["open_shipment_ids"] and totals["lost_quantity"] <= QTY_EPS
        )
        return summary
