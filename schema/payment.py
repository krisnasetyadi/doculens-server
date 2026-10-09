# schema/payment.py
"""Tables owned by router/payment.py (moved here from it by MS-657)."""

import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)

_tables_ensured = False
_usage_tables_ensured = False


def ensure(conn) -> None:
    global _tables_ensured
    if _tables_ensured:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS payments (
                    id                          BIGSERIAL PRIMARY KEY,
                    payment_id                  TEXT        NOT NULL UNIQUE,
                    user_id                     TEXT        NOT NULL,
                    plan_id                     TEXT        NOT NULL,
                    amount                      INTEGER     NOT NULL,
                    currency                    TEXT        NOT NULL DEFAULT 'idr',
                    status                      TEXT        NOT NULL DEFAULT 'pending'
                                                CHECK (status IN ('pending', 'succeeded', 'failed', 'cancelled')),
                    stripe_checkout_session_id  TEXT        UNIQUE,
                    stripe_payment_intent_id    TEXT,
                    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS idx_payments_user_created
                    ON payments (user_id, created_at DESC);

                -- Cancel-at-period-end (MS-248 follow-up): set on a
                -- SUCCEEDED payment when its admin cancels the subscription
                -- period it started. Distinct from the `status` column
                -- above (which only ever reaches 'cancelled' pre-success,
                -- via the Stripe webhook's checkout.session.expired case) —
                -- this instead marks a period that was paid for and is
                -- still running, just flagged not to be treated as
                -- renewable/resumable-by-default once it ends.
                ALTER TABLE payments
                    ADD COLUMN IF NOT EXISTS cancelled_at TIMESTAMPTZ;

                CREATE OR REPLACE FUNCTION _set_payments_updated_at()
                RETURNS TRIGGER LANGUAGE plpgsql AS $$
                BEGIN
                    NEW.updated_at = now();
                    RETURN NEW;
                END;
                $$;

                DROP TRIGGER IF EXISTS trg_payments_updated_at ON payments;
                CREATE TRIGGER trg_payments_updated_at
                    BEFORE UPDATE ON payments
                    FOR EACH ROW EXECUTE FUNCTION _set_payments_updated_at();
                """
            )
        _tables_ensured = True
    except Exception as exc:
        logger.error("payment: ensure table failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to initialize payments schema")


def ensure_usage(conn) -> None:
    """token_usage (append-only consumption ledger) + token_allocations (the
    admin-assigned per-member cap, one row per member) — same auto-create
    convention as ensure() above."""
    global _usage_tables_ensured
    if _usage_tables_ensured:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id             BIGSERIAL   PRIMARY KEY,
                    user_id        TEXT        NOT NULL,
                    admin_user_id  TEXT        NOT NULL,
                    tokens         INTEGER     NOT NULL,
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS idx_token_usage_admin_created
                    ON token_usage (admin_user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_token_usage_user_created
                    ON token_usage (user_id, created_at);

                -- MS-247 "Efficient Mode" — nullable/defaulted so existing
                -- rows and every caller that doesn't pass these stay
                -- unaffected.
                ALTER TABLE token_usage ADD COLUMN IF NOT EXISTS efficient_mode BOOLEAN NOT NULL DEFAULT false;
                ALTER TABLE token_usage ADD COLUMN IF NOT EXISTS raw_tokens_est INTEGER;
                ALTER TABLE token_usage ADD COLUMN IF NOT EXISTS final_tokens_est INTEGER;

                CREATE TABLE IF NOT EXISTS token_allocations (
                    id               BIGSERIAL   PRIMARY KEY,
                    admin_user_id    TEXT        NOT NULL,
                    user_id          TEXT        NOT NULL UNIQUE,
                    allocated_tokens INTEGER     NOT NULL DEFAULT 0,
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS idx_token_allocations_admin
                    ON token_allocations (admin_user_id);

                -- MS-418: a row remains an allocation from the workspace
                -- pool. These optional fields activate rolling per-user
                -- daily/weekly limits; allocated_tokens becomes that user's
                -- monthly cap only after the admin activates the quotas.
                ALTER TABLE token_allocations ADD COLUMN IF NOT EXISTS daily_token_quota INTEGER;
                ALTER TABLE token_allocations ADD COLUMN IF NOT EXISTS weekly_token_quota INTEGER;
                ALTER TABLE token_allocations ADD COLUMN IF NOT EXISTS quota_anchor_at TIMESTAMPTZ;

                CREATE OR REPLACE FUNCTION _set_token_allocations_updated_at()
                RETURNS TRIGGER LANGUAGE plpgsql AS $$
                BEGIN
                    NEW.updated_at = now();
                    RETURN NEW;
                END;
                $$;

                DROP TRIGGER IF EXISTS trg_token_allocations_updated_at ON token_allocations;
                CREATE TRIGGER trg_token_allocations_updated_at
                    BEFORE UPDATE ON token_allocations
                    FOR EACH ROW EXECUTE FUNCTION _set_token_allocations_updated_at();

                -- "Request more tokens" (MS-248 follow-up) — a member who
                -- hit their admin-assigned cap can ask for more; the admin
                -- sees pending ones in the Billing tab and actually raises
                -- the cap via the existing allocation editor, then dismisses
                -- the request. In-app only for now (polling, no real push).
                CREATE TABLE IF NOT EXISTS token_requests (
                    id             BIGSERIAL   PRIMARY KEY,
                    request_id     TEXT        NOT NULL UNIQUE,
                    user_id        TEXT        NOT NULL,
                    admin_user_id  TEXT        NOT NULL,
                    message        TEXT,
                    status         TEXT        NOT NULL DEFAULT 'pending'
                                   CHECK (status IN ('pending', 'resolved')),
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                    resolved_at    TIMESTAMPTZ
                );

                CREATE INDEX IF NOT EXISTS idx_token_requests_admin_status
                    ON token_requests (admin_user_id, status, created_at DESC);

                -- MS-402: per-workspace "Default Token Allocation". No row
                -- means the admin hasn't set one yet and
                -- config.default_member_token_allocation applies.
                CREATE TABLE IF NOT EXISTS workspace_settings (
                    admin_user_id             TEXT        PRIMARY KEY,
                    default_member_allocation INTEGER     NOT NULL,
                    updated_at                TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
        _usage_tables_ensured = True
    except Exception as exc:
        logger.error("payment: ensure usage tables failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to initialize token-usage schema")
