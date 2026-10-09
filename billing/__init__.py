# billing/__init__.py
"""
Plans, token metering and payments (moved out of router/payment.py by MS-657).

  plans        plan prices/quotas and the subscription period a workspace is in
  ledger       token_usage: logging consumption, sums, the per-user rate limit
  quotas       daily/weekly/monthly windows for a member's allocation
  allocation   per-member allocations carved out of the workspace pool
  inflight     workspace locks and in-flight reservations around LLM calls
  enforcement  the checks run before an LLM call
  payments     Stripe Checkout and the payments table
  token_requests  members asking their admin for more tokens
"""
