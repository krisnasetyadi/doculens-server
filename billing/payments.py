# billing/payments.py
"""Stripe Checkout (test mode, one-time payment — MS-90) and the payments table."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import stripe
from fastapi import HTTPException

import app_db
import db
from billing.allocation import build_subscription_usage
from billing.plans import PLAN_PRICES, get_latest_plan_window
from config import config
from models import (
    CheckoutSessionResponse,
    CreateCheckoutSessionRequest,
    PaymentRecord,
    PaymentResponse,
)
from router.auth import UserRecord

logger = logging.getLogger(__name__)

stripe.api_key = config.stripe_secret_key


def _as_record(row) -> PaymentRecord:
    return PaymentRecord(
        payment_id=row["payment_id"],
        plan_id=row["plan_id"],
        amount=row["amount"],
        currency=row["currency"],
        status=row["status"],
        created_at=app_db.ts(row.get("created_at")),
    )


def create_checkout_session(body: CreateCheckoutSessionRequest, user: Optional[UserRecord]):
    user_id = user.user_id if user else "guest"
    plan = PLAN_PRICES.get(body.plan_id)
    if not plan:
        raise HTTPException(status_code=400, detail="Unknown plan")

    if not config.stripe_secret_key:
        raise HTTPException(status_code=503, detail="Payments are not configured")

    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    payment_id = f"pay_{uuid.uuid4().hex}"

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "idr",
                        "product_data": {"name": f"DocuLens {plan['name']} plan"},
                        "unit_amount": plan["amount"],
                    },
                    "quantity": 1,
                }
            ],
            success_url=(
                f"{config.frontend_url}/payment/result"
                "?status=success&session_id={CHECKOUT_SESSION_ID}"
            ),
            cancel_url=f"{config.frontend_url}/payment/result?status=cancelled",
            metadata={"user_id": user_id, "plan_id": body.plan_id, "payment_id": payment_id},
        )
    except Exception as exc:
        logger.error("payment: checkout session creation failed: %s", exc)
        conn.close()
        raise HTTPException(status_code=502, detail="Could not start checkout")

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO payments (payment_id, user_id, plan_id, amount, currency, status, stripe_checkout_session_id)
                VALUES (%s, %s, %s, %s, 'idr', 'pending', %s)
                """,
                (payment_id, user_id, body.plan_id, plan["amount"], session.id),
            )
    except Exception as exc:
        logger.error("payment: failed to record pending payment: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to start checkout")
    finally:
        conn.close()

    return CheckoutSessionResponse(checkout_url=session.url, payment_id=payment_id)


def get_payment_by_session(session_id: str):
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM payments WHERE stripe_checkout_session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Payment not found")

    return PaymentResponse(payment=_as_record(row))


def cancel_subscription(admin: UserRecord):
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        window = get_latest_plan_window(conn, admin.user_id)
        if not window or window.status != "active" or window.payment_id is None:
            raise HTTPException(status_code=400, detail="No paid subscription to cancel")
        if window.cancel_at_period_end:
            raise HTTPException(status_code=400, detail="Subscription is already set to cancel")

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE payments SET cancelled_at = now() WHERE payment_id = %s",
                (window.payment_id,),
            )
        subscription = build_subscription_usage(conn, admin.user_id)
    finally:
        conn.close()
    return subscription


def resume_subscription(admin: UserRecord):
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        window = get_latest_plan_window(conn, admin.user_id)
        if not window or window.status != "active":
            raise HTTPException(status_code=400, detail="Subscription period has already ended")
        if not window.cancel_at_period_end:
            raise HTTPException(status_code=400, detail="Subscription isn't set to cancel")

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE payments SET cancelled_at = NULL WHERE payment_id = %s",
                (window.payment_id,),
            )
        subscription = build_subscription_usage(conn, admin.user_id)
    finally:
        conn.close()
    return subscription


def handle_webhook(payload: bytes, sig_header: Optional[str]) -> dict:
    if not config.stripe_webhook_secret:
        # Not configured (e.g. local dev without `stripe listen` yet) — ack
        # rather than 500, since Stripe retries failed webhooks aggressively.
        logger.warning("payment: webhook received but STRIPE_WEBHOOK_SECRET is not set")
        return {"received": True}

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, config.stripe_webhook_secret)
    except Exception as exc:
        logger.warning("payment: webhook signature verification failed: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    conn = db.get_conn("payment")
    if not conn:
        # Let Stripe retry rather than silently losing the event.
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        event_type = event["type"]
        # stripe-python 15.x's typed objects (e.g. Session) support [] item
        # access but not .get() — .to_dict() gives a plain dict so both the
        # .get() calls and the ["id"] access below behave as expected.
        data = event["data"]["object"].to_dict()

        # Only these two are handled: `checkout.session.completed` is the
        # reliable, documented success signal for mode="payment" Checkout;
        # `checkout.session.expired` covers an abandoned/timed-out session.
        # payment_intent.payment_failed is deliberately not handled — we
        # don't have a stripe_payment_intent_id on file to match against
        # until a session actually completes, so it can't reliably find the
        # right row anyway.
        if event_type == "checkout.session.completed":
            new_status = "succeeded" if data.get("payment_status") == "paid" else "failed"
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE payments SET status = %s, stripe_payment_intent_id = %s
                    WHERE stripe_checkout_session_id = %s
                    """,
                    (new_status, data.get("payment_intent"), data["id"]),
                )
        elif event_type == "checkout.session.expired":
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE payments SET status = 'cancelled' WHERE stripe_checkout_session_id = %s",
                    (data["id"],),
                )
    except Exception as exc:
        logger.error("payment: webhook handling failed: %s", exc)
    finally:
        conn.close()

    return {"received": True}
