"""MS-402: pure token-allocation rules in router/payment.py.

Run: python -m unittest test_token_allocation
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# router.auth refuses to import without one; these tests never issue tokens.
os.environ.setdefault("JWT_SECRET", "test-only-secret")

from datetime import datetime, timedelta, timezone

from router.payment import (
    SUBSCRIPTION_PERIOD_DAYS,
    _free_window,
    clamp_to_pool,
    effective_allocations,
    exceeds_cap,
)


class ExceedsCapTest(unittest.TestCase):
    def test_blocks_once_cap_reached(self):
        self.assertTrue(exceeds_cap(used=5_000, cap=5_000, reserve=0))

    def test_allows_under_cap_without_reserve(self):
        self.assertFalse(exceeds_cap(used=4_999, cap=5_000, reserve=0))

    def test_blocks_when_remaining_cant_fit_reserve(self):
        # The overshoot this ticket is about: 10 tokens left, a query costs ~2k.
        self.assertTrue(exceeds_cap(used=4_990, cap=5_000, reserve=2_000))

    def test_allows_when_reserve_fits_exactly(self):
        self.assertFalse(exceeds_cap(used=3_000, cap=5_000, reserve=2_000))

    def test_cap_below_reserve_still_allows_first_query(self):
        # Reserve is limited to cap // 2 = 500, so a 1,000 cap isn't dead on arrival.
        self.assertFalse(exceeds_cap(used=0, cap=1_000, reserve=2_000))
        self.assertFalse(exceeds_cap(used=500, cap=1_000, reserve=2_000))
        self.assertTrue(exceeds_cap(used=501, cap=1_000, reserve=2_000))

    def test_cap_equal_to_reserve_allows_more_than_one_query(self):
        self.assertFalse(exceeds_cap(used=800, cap=2_000, reserve=2_000))

    def test_zero_cap_always_blocks(self):
        self.assertTrue(exceeds_cap(used=0, cap=0, reserve=0))

    def test_negative_reserve_treated_as_zero(self):
        self.assertFalse(exceeds_cap(used=4_999, cap=5_000, reserve=-100))


class ClampToPoolTest(unittest.TestCase):
    def test_grants_request_that_fits(self):
        self.assertEqual(clamp_to_pool(5_000, token_limit=60_000, allocated_elsewhere=10_000), (5_000, False))

    def test_clamps_to_what_is_left(self):
        self.assertEqual(clamp_to_pool(5_000, token_limit=60_000, allocated_elsewhere=58_000), (2_000, True))

    def test_over_allocated_pool_grants_zero(self):
        self.assertEqual(clamp_to_pool(5_000, token_limit=60_000, allocated_elsewhere=70_000), (0, True))


class FreeWindowTest(unittest.TestCase):
    """Expired paid plan drops to a Free window anchored at its expiry."""

    def test_window_contains_now_and_starts_on_a_period_boundary(self):
        now = datetime.now(timezone.utc)
        anchor = now - timedelta(days=SUBSCRIPTION_PERIOD_DAYS * 2 + 3)
        window = _free_window(anchor)
        self.assertLessEqual(window.period_start, now)
        self.assertLess(now, window.period_end)
        self.assertEqual((window.period_start - anchor) % timedelta(days=SUBSCRIPTION_PERIOD_DAYS), timedelta(0))

    def test_uses_free_quota_and_is_active(self):
        window = _free_window(datetime.now(timezone.utc) - timedelta(days=2))
        self.assertEqual(window.plan["name"], "Free")
        self.assertEqual(window.status, "active")
        self.assertIsNone(window.payment_id)


class EffectiveAllocationsTest(unittest.TestCase):
    def test_member_without_row_gets_default(self):
        result = effective_allocations({}, ["m1"], default_allocation=5_000)
        self.assertEqual(result, {"m1": (5_000, True)})

    def test_explicit_row_wins_over_default(self):
        result = effective_allocations({"m1": 1_000}, ["m1"], default_allocation=5_000)
        self.assertEqual(result, {"m1": (1_000, False)})

    def test_explicit_zero_is_not_replaced_by_default(self):
        result = effective_allocations({"m1": 0}, ["m1"], default_allocation=5_000)
        self.assertEqual(result, {"m1": (0, False)})

    def test_admin_row_kept_even_though_not_a_member(self):
        result = effective_allocations({"admin": 3_000}, ["m1"], default_allocation=5_000)
        self.assertEqual(result, {"admin": (3_000, False), "m1": (5_000, True)})


if __name__ == "__main__":
    unittest.main()
