# models.py
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import List, Literal, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime

class ChatPlatform(str, Enum):
    WHATSAPP = "whatsapp"
    TEAMS = "teams"
    SLACK = "slack"
    TELEGRAM = "telegram"
    GENERIC = "generic"

class UploadResponse(BaseModel):
    collection_id: str
    file_count: int
    status: str
    file_names: Optional[List[str]] = None
    title: Optional[str] = None


class UploadFromUrlRequest(BaseModel):
    url: str
    title: Optional[str] = None


class UploadFromUrlsRequest(BaseModel):
    urls: List[str]
    title: Optional[str] = None


class DriveFolderRequest(BaseModel):
    url: str
    recursive: Optional[bool] = True
    max_depth: Optional[int] = 5


class DriveFolderItem(BaseModel):
    id: str
    name: str
    url: str
    item_type: str


class DriveFolderItemsResponse(BaseModel):
    folder_id: str
    files: List[DriveFolderItem]
    folders: List[DriveFolderItem]
    count: int


class CreatePublicLinkRequest(BaseModel):
    title: Optional[str] = None
    url: str
    item_urls: Optional[List[str]] = None


class SetPublicLinkActiveRequest(BaseModel):
    link_id: str
    active: bool


class PublicLinkItem(BaseModel):
    id: str
    name: str
    url: str
    item_type: str


class PublicLinkSource(BaseModel):
    link_id: str
    workspace_id: Optional[str] = None
    title: str
    url: str
    status: str
    item_count: int
    created_at: Union[str, datetime]
    items: List[PublicLinkItem] = []


class PublicLinksResponse(BaseModel):
    links: List[PublicLinkSource]
    count: int


class CreateDatabaseConnectionRequest(BaseModel):
    label: Optional[str] = None
    url: str


class SetDatabaseConnectionActiveRequest(BaseModel):
    connection_id: str
    active: bool


class DbColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False


class DbTableInfo(BaseModel):
    name: str
    row_count: Optional[int] = None
    columns: List[DbColumnInfo] = []


class DatabaseConnectionSource(BaseModel):
    connection_id: str
    workspace_id: Optional[str] = None
    label: str
    url: str
    status: str
    table_count: int
    created_at: Union[str, datetime]
    tables: List[DbTableInfo] = []


class DatabaseConnectionsResponse(BaseModel):
    connections: List[DatabaseConnectionSource]
    count: int


class CreateCheckoutSessionRequest(BaseModel):
    plan_id: str


class CheckoutSessionResponse(BaseModel):
    checkout_url: str
    payment_id: str


class PaymentRecord(BaseModel):
    payment_id: str
    plan_id: str
    amount: int
    currency: str
    status: str
    created_at: Union[str, datetime]


class PaymentResponse(BaseModel):
    payment: PaymentRecord


# ===================== TOKEN USAGE & ALLOCATION (MS-248) =====================
# Workspace-level subscription usage plus, on top of it, per-member token
# allocations that an admin distributes out of the workspace's token_limit
# (see services/billing/allocation.py for how these are computed/persisted).

class SubscriptionUsage(BaseModel):
    plan_name: str
    subscription_status: str  # active | expired | none
    token_limit: int
    token_used: int
    token_remaining: int
    period_start: Union[str, datetime]
    period_end: Union[str, datetime]
    next_reset_date: Optional[Union[str, datetime]] = None
    # True once the admin has cancelled — access still runs until period_end
    # (already paid for), it just won't be treated as renewable after that.
    cancel_at_period_end: bool = False
    # False for the synthetic Free-tier plan (nobody's paid — see
    # _get_latest_plan_window's fallback branch) — there's no purchase on
    # file to cancel, so the frontend should hide cancel/resume for it.
    is_paid: bool = True


class TokenQuotaTierUsage(BaseModel):
    interval: str  # daily | weekly | monthly
    token_limit: int
    token_used: int
    token_remaining: int
    period_start: Union[str, datetime]
    next_reset_date: Union[str, datetime]
    blocked: bool


class MemberTokenUsage(BaseModel):
    user_id: str
    email: str
    allocated_tokens: int
    used_tokens: int
    remaining_tokens: int
    usage_percent: float
    # MS-402: True when this member has no explicit token_allocations row and
    # allocated_tokens is the workspace's Default Token Allocation — the
    # value that's actually enforced for them, not an explicit admin pick.
    is_default_allocation: bool = False
    # None for legacy allocations that still follow the workspace plan period.
    quota_anchor_at: Optional[Union[str, datetime]] = None
    quota_tiers: List[TokenQuotaTierUsage] = Field(default_factory=list)


class MyMemberUsageResponse(BaseModel):
    usage: Optional[MemberTokenUsage] = None
    # Whether this user's workspace plan includes Compliance Gap Check
    # (services.billing.plans.GAP_CHECK_PLAN_IDS) — chat-ui hides the feature when False
    # instead of letting the user hit the 403 from enforce_gap_check_plan.
    gap_check_available: bool = False


class MembersUsageResponse(BaseModel):
    subscription: Optional[SubscriptionUsage] = None
    members: List[MemberTokenUsage]
    unallocated_tokens: int
    # MS-402: the pool allocations are carved from and enforced against --
    # differs from subscription.token_limit once a paid plan has expired
    # and the workspace has dropped back to the Free quota.
    pool_token_limit: int = 0
    pool_plan_name: Optional[str] = None


class UpdateMemberAllocationRequest(BaseModel):
    user_id: str
    allocated_tokens: int = Field(..., ge=0, strict=True)
    daily_token_quota: Optional[int] = Field(default=None, ge=0, strict=True)
    weekly_token_quota: Optional[int] = Field(default=None, ge=0, strict=True)


class UpdateMemberAllocationResponse(BaseModel):
    member: MemberTokenUsage
    unallocated_tokens: int


# MS-402: workspace-level "Default Token Allocation" — what a newly created
# team member (or an existing one with no explicit allocation) is capped at.
class WorkspaceTokenSettings(BaseModel):
    default_member_allocation: int


class UpdateWorkspaceTokenSettingsRequest(BaseModel):
    default_member_allocation: int = Field(..., ge=0)


# MS-504: what Settings > Storage and the Files-tab meter show. Limits come
# from the workspace plan, so members see the same numbers as their admin.
class StorageUsageResponse(BaseModel):
    plan_name: str
    used_bytes: int
    limit_bytes: int
    remaining_bytes: int
    usage_percent: float
    max_file_bytes: int
    max_batch_files: int
    # True once no new upload can fit, so the UI need not infer it from
    # remaining_bytes == 0.
    blocked: bool


# Flat, plan-independent safety-net rate limit (same for every user) — see
# services/billing/ledger.py::get_rate_limit_status. Separate from SubscriptionUsage
# / MemberTokenUsage above, which track the per-plan/per-member allocation.
class RateLimitStatus(BaseModel):
    used_tokens: int
    cap_tokens: int
    window_hours: float
    blocked: bool
    reset_at: Optional[Union[str, datetime]] = None


# MS-247 "Efficient Mode" — aggregate stats for the Settings mini-dashboard.
class EfficientModeStats(BaseModel):
    queries_tested: int
    avg_reduction_pct: float
    total_raw_tokens_est: int
    total_final_tokens_est: int
    total_tokens_saved_est: int


# "Request more tokens" (MS-248 follow-up) — in-app only (polled, not real
# push). A member who hit their admin-assigned cap can ask for more; the
# admin sees pending ones in the Billing tab, raises the cap themselves via
# the existing allocation editor, then dismisses the request.
class CreateTokenRequestRequest(BaseModel):
    message: Optional[str] = None


class TokenRequestRecord(BaseModel):
    request_id: str
    user_id: str
    email: str
    message: Optional[str] = None
    status: str  # pending | resolved
    created_at: Union[str, datetime]


class TokenRequestResponse(BaseModel):
    request: TokenRequestRecord


class TokenRequestsResponse(BaseModel):
    requests: List[TokenRequestRecord]
    pending_count: int


class CollectionInfo(BaseModel):
    collection_id: str
    document_count: int
    created_at: Union[str, datetime]
    file_names: List[str]
    title: Optional[str] = None
    status: str = "active"
    owner_id: Optional[str] = None
    folder_id: Optional[str] = None


class SetPdfCollectionActiveRequest(BaseModel):
    collection_id: str
    active: bool


# ===================== SOURCE FOLDERS (MS-274) =====================
# Folders group `collections` rows (kind='pdf' or 'chat') for the Files tab
# only. Owner-scoped like collections themselves — there is no "workspace"
# table to attach to (see services/storage/supabase.py::_team_admin_id for why).

class Folder(BaseModel):
    folder_id: str
    name: str
    owner_id: str
    parent_folder_id: Optional[str] = None
    created_at: Union[str, datetime]
    updated_at: Union[str, datetime]


class FolderCreate(BaseModel):
    name: str
    parent_folder_id: Optional[str] = None


class FolderRename(BaseModel):
    name: str
    parent_folder_id: Optional[str] = None


class MoveToFolderRequest(BaseModel):
    collection_id: str
    folder_id: Optional[str] = None  # null = unassign back to root


class DatabaseResult(BaseModel):
    table: str
    data: List[Dict[str, Any]]
    record_count: int
    avg_relevance_score: Optional[float] = None  # Average score dari search results


    # answer: str
    # pdf_sources: List[str]
    # db_results: Dict[str, DatabaseResult]  # FIXED: Dict bukan List
    # processing_time: float
    # search_terms: List[str]

# ===================== CHAT MODELS =====================

class ChatMessage(BaseModel):
    """Single chat message parsed from export file"""
    message_id: Optional[str] = None
    sender: str
    timestamp: datetime
    content: str
    platform: ChatPlatform = ChatPlatform.WHATSAPP
    thread_id: Optional[str] = None
    conversation_id: Optional[str] = None
    raw_line: Optional[str] = None  # Original line for debugging


class ChatCollection(BaseModel):
    """Metadata for an uploaded chat collection"""
    collection_id: str
    platform: ChatPlatform
    file_name: str
    message_count: int
    date_range: Optional[Dict[str, str]] = None  # {"start": ..., "end": ...}
    participants: List[str]
    created_at: datetime
    status: str = "active"


class SetChatCollectionActiveRequest(BaseModel):
    collection_id: str
    active: bool


# ===================== TELEGRAM MODELS =====================
# Live connection (Telethon/MTProto login), not a file upload — see
# services/telegram.py. A connection holds one encrypted user session; a
# connection can have multiple "selected chats", each of which becomes its
# own searchable ChatCollection (platform=telegram) once synced.

# How far back a synced chat is kept: the last month, 3 months, year, or everything.
TelegramHistoryRange = Literal["1m", "3m", "1y", "all"]


class TelegramDialog(BaseModel):
    dialog_id: str
    title: str
    type: str  # "user" | "group" | "channel"
    participants_count: Optional[int] = None


class TelegramDialogsResponse(BaseModel):
    dialogs: List[TelegramDialog]
    count: int


class TelegramSelectedChat(BaseModel):
    dialog_id: str
    title: str
    type: str
    chat_collection_id: Optional[str] = None
    message_count: Optional[int] = None
    history_range: TelegramHistoryRange = "all"
    status: str = "active"
    last_synced_at: Optional[Union[str, datetime]] = None


class TelegramConnectionSource(BaseModel):
    connection_id: str
    label: str
    phone_masked: str
    status: str
    created_at: Union[str, datetime]
    selected_chats: List[TelegramSelectedChat] = []


class TelegramConnectionsResponse(BaseModel):
    connections: List[TelegramConnectionSource]
    count: int
    # False until TELEGRAM_API_ID / TELEGRAM_API_HASH are set, so the client can
    # say "not configured" before a sign-in starts instead of failing midway.
    app_configured: bool = False


class TelegramChatRangeRequest(BaseModel):
    history_range: TelegramHistoryRange


class TelegramChatRemovedResponse(BaseModel):
    status: str
    message: str
    connection_id: str
    dialog_id: str


class TelegramConnectStartRequest(BaseModel):
    # api_id/api_hash identify the Telegram *application* making the request.
    # Leave both out to use DocuLens's own registration (TELEGRAM_API_ID /
    # TELEGRAM_API_HASH); sending them still works for clients older than MS-911.
    api_id: Optional[int] = None
    api_hash: Optional[str] = None
    phone: str
    label: Optional[str] = None


class TelegramConnectStartResponse(BaseModel):
    flow_id: str
    phone: str


class TelegramConnectVerifyRequest(BaseModel):
    flow_id: str
    code: str
    password: Optional[str] = None  # only needed if the account has 2FA enabled


class TelegramConnectVerifyResponse(BaseModel):
    status: str  # "connected" | "password_required"
    connection: Optional[TelegramConnectionSource] = None


class TelegramQrStartRequest(BaseModel):
    label: Optional[str] = None


class TelegramQrCancelRequest(BaseModel):
    flow_id: str


class TelegramQrLoginState(BaseModel):
    """Where a QR sign-in stands. The client polls it and draws `qr_url` (a
    tg://login link Telegram issued) as the QR code; it changes whenever the
    token is renewed. "password_required" continues on connect/verify with the
    same flow_id, like the code flow."""
    flow_id: str
    status: str  # "waiting" | "password_required" | "connected" | "expired" | "error"
    qr_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    connection: Optional[TelegramConnectionSource] = None
    error: Optional[str] = None


class TelegramSyncRequest(BaseModel):
    dialog_ids: List[str]
    # dialog_id -> range, for the chats the user just picked. A chat left out
    # keeps the range it was synced with, or "all" if it was never synced.
    ranges: Dict[str, TelegramHistoryRange] = {}
    # Kept for older clients; ignored. The history range decides how much is read.
    message_limit: Optional[int] = None


class TelegramSyncResult(BaseModel):
    dialog_id: str
    title: str
    chat_collection_id: str
    message_count: int
    status: str  # "success" | "error"
    error: Optional[str] = None


class TelegramSyncResponse(BaseModel):
    results: List[TelegramSyncResult]


class SetTelegramConnectionActiveRequest(BaseModel):
    connection_id: str
    active: bool


class ChatUploadResponse(BaseModel):
    """Response after uploading chat file"""
    collection_id: str
    file_name: str
    platform: str
    message_count: int
    participants: List[str]
    date_range: Optional[Dict[str, str]] = None
    status: str


# ===================== SKILL / GAP-ANALYSIS MODELS =====================
# Generic "Reference Framework Gap Analysis" capability. skill_id selects
# behavior; ISO 27001 is just the first framework_name used with
# "compliance_gap_check" — nothing here is ISO-specific.

class GapAnalysisRequest(BaseModel):
    skill_id: str  # "compliance_gap_check" | "scenario_regulatory_impact"
    reference_collection_ids: List[str]  # array from day one (Opsi A: multi-framework per run)
    framework_name: str  # free-label, e.g. "ISO 27001" or "Ketentuan PPh Pinjaman vs Modal"
    target_collection_ids: List[str] = []  # required for compliance_gap_check — one guideline vs N files, verdict per file
    scenario_input: Optional[str] = None  # used by scenario_regulatory_impact instead of a target collection


class GapAnalysisItem(BaseModel):
    label: str  # control/klausul name (Skill 1) or option name (Skill 2)
    status: str  # "met" | "partial" | "not_met" | "unknown"
    evidence: Optional[str] = None
    source_citation: Optional[str] = None
    recommendation: Optional[str] = None
    target_collection_id: Optional[str] = None  # which target collection/file this item was checked against


class GapAnalysisRun(BaseModel):
    run_id: str
    skill_id: str
    framework_name: str
    reference_collection_ids: List[str]
    target_collection_ids: List[str] = []
    scenario_input: Optional[str] = None
    status: str = "completed"
    created_at: Union[str, datetime] = ""


class GapAnalysisResponse(BaseModel):
    run: GapAnalysisRun
    items: List[GapAnalysisItem]
    summary: Dict[str, int]  # counts per status, e.g. {"met": 60, "partial": 18, "not_met": 15, "unknown": 0}
    disclaimer: Optional[str] = None


# ===================== SKILLS (MS-251) =====================
# A skill is an uploaded instruction file: `instruction` holds the body of the
# .md, the rest is its frontmatter. Who may use it is decided by `scope` alone
# — "personal" is the uploader's own, "team" is an admin's and reaches every
# member that admin created (see storage.list_skills_for_user).

SKILL_SCOPES = ("personal", "team")


def _validate_scope(v: str) -> str:
    if v not in SKILL_SCOPES:
        raise ValueError(f"scope must be one of {SKILL_SCOPES}")
    return v


def _normalize_command(v: str) -> str:
    """Store commands in the one shape the "/" menu matches on, so a skill
    uploaded as "audit" is still reachable as "/audit"."""
    v = v.strip()
    if not v.lstrip("/"):
        raise ValueError("slash_command must not be empty")
    return "/" + v.lstrip("/")


def _require_text(v: str, field: str) -> str:
    if not v or not v.strip():
        raise ValueError(f"{field} must not be empty")
    return v.strip()


class SkillCreate(BaseModel):
    name: str
    slash_command: str
    instruction: str          # body of the uploaded .md
    description: str = ""
    scope: str = "personal"   # "team" is rejected for non-admins in services/skills.py

    @field_validator("scope")
    @classmethod
    def _scope(cls, v: str) -> str:
        return _validate_scope(v)

    @field_validator("slash_command")
    @classmethod
    def _command(cls, v: str) -> str:
        return _normalize_command(v)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _require_text(v, "name")

    @field_validator("instruction")
    @classmethod
    def _instruction(cls, v: str) -> str:
        return _require_text(v, "instruction")


class SkillUpdate(BaseModel):
    """Partial update — every field optional, omitted ones are left alone."""
    name: Optional[str] = None
    slash_command: Optional[str] = None
    instruction: Optional[str] = None
    description: Optional[str] = None
    scope: Optional[str] = None

    @field_validator("scope")
    @classmethod
    def _scope(cls, v: Optional[str]) -> Optional[str]:
        return _validate_scope(v) if v is not None else v

    @field_validator("slash_command")
    @classmethod
    def _command(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_command(v) if v is not None else v

    @field_validator("name")
    @classmethod
    def _name(cls, v: Optional[str]) -> Optional[str]:
        return _require_text(v, "name") if v is not None else v

    @field_validator("instruction")
    @classmethod
    def _instruction(cls, v: Optional[str]) -> Optional[str]:
        return _require_text(v, "instruction") if v is not None else v


class SkillResponse(BaseModel):
    skill_id: str
    name: str
    slash_command: str
    description: str = ""
    instruction: str
    scope: str
    owner_id: str
    created_at: Union[str, datetime] = ""
    updated_at: Union[str, datetime] = ""



# ---------------------------------------------------------------------------
# Chat sessions (router/sessions.py)
# ---------------------------------------------------------------------------

class StoredMessage(BaseModel):
    id: str
    role: str          # "user" | "assistant"
    content: str
    model_used: Optional[str] = None
    created_at: str    # ISO

class UpsertSessionRequest(BaseModel):
    session_id: Optional[str] = None   # if None -> create new
    # Optional past creation — omitted, the existing title (auto-derived or
    # since renamed via PUT /sessions/{id}) is left untouched (MS-253).
    title: Optional[str] = None
    messages: List[StoredMessage]
    pdf_collections: Optional[List[str]] = []
    chat_collections: Optional[List[str]] = []

class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[StoredMessage]
    pdf_collections: List[str]
    chat_collections: List[str]
    # MS-237: only meaningful on the paginated GET below. POST (create/
    # update) always returns the full list it was given, so has_more is
    # correctly False there too — nothing more to page in.
    has_more: bool = False
    next_cursor: Optional[str] = None
    # Total user-authored messages in the session (loaded or not) — lets the
    # client draw one navigation marker per question, including ones it
    # hasn't fetched yet.
    total_user_turns: int = 0

class SessionSummary(BaseModel):
    session_id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str
    pdf_collections: List[str]
    chat_collections: List[str]
    # MS-417: set only when `q` matched inside a message and not in the title —
    # a title match is shown by highlighting the title itself. Plain text, so
    # the client can highlight by splitting on the query.
    matched_snippet: Optional[str] = None
    matched_message_id: Optional[str] = None

class SessionQuestion(BaseModel):
    """One entry in the navigation index — a question the user asked, with
    just enough text to recognise it in a list."""
    turn: int          # 1-based, counted from the start of the session
    message_id: str
    preview: str


class SessionQuestionsResponse(BaseModel):
    session_id: str
    total: int         # every question in the session, even beyond the cap below
    questions: List[SessionQuestion]


class RenameSessionRequest(BaseModel):
    title: str


# ---------------------------------------------------------------------------
# Auth (router/auth.py)
# ---------------------------------------------------------------------------

PASSWORD_MAX_BYTES = 72  # bcrypt hard limit

def _validate_password_length(v: str) -> str:
    if len(v.encode("utf-8")) > PASSWORD_MAX_BYTES:
        raise ValueError(f"Password must be at most {PASSWORD_MAX_BYTES} bytes long")
    return v


EMAIL_MAX_LENGTH = 50
NEW_MEMBER_PASSWORD_MIN_LENGTH = 8
NEW_MEMBER_PASSWORD_MAX_LENGTH = 50


def _validate_new_member_email(v: str) -> str:
    if len(v) > EMAIL_MAX_LENGTH:
        raise ValueError(f"Email must be at most {EMAIL_MAX_LENGTH} characters")
    return v


def _validate_new_member_password(v: str) -> str:
    if len(v) < NEW_MEMBER_PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {NEW_MEMBER_PASSWORD_MIN_LENGTH} characters")
    if len(v) > NEW_MEMBER_PASSWORD_MAX_LENGTH:
        raise ValueError(f"Password must be at most {NEW_MEMBER_PASSWORD_MAX_LENGTH} characters")
    return _validate_password_length(v)


NAME_MAX_LENGTH = 100


def _validate_name(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    v = v.strip()
    if not v:
        raise ValueError("Name cannot be empty")
    if len(v) > NAME_MAX_LENGTH:
        raise ValueError(f"Name must be at most {NAME_MAX_LENGTH} characters")
    return v


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        return _validate_password_length(v)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        return _validate_name(v)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    role: str
    name: Optional[str] = None

class UserRecord(BaseModel):
    user_id: str
    email: str
    role: str
    is_active: bool
    name: Optional[str] = None
    avatar_url: Optional[str] = None

class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    remove_avatar: bool = False  # explicit signal to clear avatar_url — None on avatar_url just means "leave as-is"

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        return _validate_name(v)

    @field_validator("avatar_url")
    @classmethod
    def validate_avatar_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not v.startswith("data:image/"):
            raise ValueError("Avatar must be an image data URL")
        if len(v) > 500_000:
            raise ValueError("Avatar image is too large")
        return v

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        return _validate_password_length(v)

class AdminResetRequest(BaseModel):
    user_id: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        return _validate_password_length(v)


class AdminCreateUserRequest(BaseModel):
    email: EmailStr
    password: str
    # MS-402: optional custom token cap; omitted -> the workspace's Default
    # Token Allocation (services/billing/allocation.py::assign_initial_allocation).
    allocated_tokens: Optional[int] = Field(default=None, ge=0)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return _validate_new_member_email(v)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        return _validate_new_member_password(v)


class AdminUserStatusRequest(BaseModel):
    user_id: str
    active: bool


class AdminUpdateUserRequest(BaseModel):
    name: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        return _validate_name(v) if v is not None else v


class TeamMember(BaseModel):
    user_id: str
    email: str
    role: str
    is_active: bool
    name: Optional[str] = None
    created_at: datetime
    # MS-402: only set on the create response, so the admin UI can land on
    # this member's Token Settings with the value that was actually granted
    # (allocation_clamped = reduced to fit what was left of the pool).
    allocated_tokens: Optional[int] = None
    allocation_clamped: bool = False


class TeamMembersResponse(BaseModel):
    members: list[TeamMember]
    max_sub_users: int


# ---------------------------------------------------------------------------
# Agnostic query (router/agnostic.py, services/agnostic.py)
# ---------------------------------------------------------------------------

class MemoryTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class AgnosticQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

    # Kept for schema compatibility with HF Space / old clients — ignored locally.
    source: Optional[str] = Field(None)

    # MS-237: which session this question belongs to (currently unused
    # server-side — the client sends `memory` directly instead of us looking
    # it up — kept so the contract already has it once that moves server-side).
    session_id: Optional[str] = None
    # Previous 5 messages so a follow-up like "ringkas semua di atas" has
    # something to resolve against. Re-clamped below regardless of what the
    # client sends.
    memory: Optional[List[MemoryTurn]] = None

    # MS-252: one-shot skill invocation — the uploaded Skill (see
    # router/skills.py) whose `instruction` should shape THIS answer only.
    # Resolved + authorization-checked server-side via
    # storage.get_skill_for_user; an unresolved id is ignored, never an
    # error (the client already cleared its own armed state either way).
    skill_id: Optional[str] = None

    include_pdf_results:  Optional[bool] = True
    # Legacy: queries THIS app's own fixed database (app management data, not
    # a user-connected source). Kept for backward compatibility only.
    include_db_results:   Optional[bool] = False
    include_chat_results: Optional[bool] = False
    include_public_links: Optional[bool] = False
    # Queries the user's own external database connection(s) set up in
    # Sources > Database — the actual "database as a knowledge source" path.
    include_external_db:  Optional[bool] = False
    llm_provider:         Optional[str]  = None
    llm_model:            Optional[str]  = None
    source_mode:          Optional[str]  = None

    # MS-247: opt-in context-compression experiment ("Efficient Mode" /
    # caveman-inspired). Defaults to False so every existing caller is
    # byte-for-byte unaffected — see services/rag/efficient_mode/compressor.py.
    efficient_mode:       Optional[bool] = False

    # Optional collection selectors (defaults to "all")
    pdf_collection_ids:      Optional[List[str]] = None
    chat_collection_ids:     Optional[List[str]] = None
    public_link_ids:         Optional[List[str]] = None
    external_db_connection_ids: Optional[List[str]] = None


class PdfSourceDetail(BaseModel):
    file_name:        str
    collection_id:    str
    page:             Optional[int]   = None
    relevance_score:  Optional[float] = None
    content_preview:  Optional[str]   = None
    file_url:         Optional[str]   = None
    page_url:         Optional[str]   = None
    search_text:      Optional[str]   = None


class AgnosticQueryResponse(BaseModel):
    answer:               str
    model_used:           str
    pdf_sources:          List[str]
    pdf_sources_detailed: List[PdfSourceDetail]
    db_results:           Dict[str, Any]
    chat_results:         List[Any]
    processing_time:      float
    search_terms:         List[str]
    target_tables:        List[str]
    source_type:          str = "Indexed Collections"
    retrieved_count:      int = 0
    # MS-247: populated only when efficient_mode was requested and the
    # answer actually went through the LLM branch (absent for the
    # system/meta-help/no-source short-circuits above, and for the
    # low-confidence direct-extraction branch, none of which build a
    # prompt/context to measure).
    efficiency:           Optional[Dict[str, Any]] = None



# ===================== ROUTE RESPONSES (MS-657) =====================
# Bodies these routes already returned as plain dicts, declared so OpenAPI
# documents them and a field that goes missing fails instead of vanishing.

class MessageResponse(BaseModel):
    message: str


class UserActiveResponse(BaseModel):
    status: str
    user_id: str
    active: bool


class UserDeletedResponse(BaseModel):
    status: str
    user_id: str


class CollectionActiveResponse(BaseModel):
    status: str
    collection_id: str
    active: bool


class CollectionFolderResponse(BaseModel):
    status: str
    collection_id: str
    folder_id: Optional[str] = None


class StatusMessageResponse(BaseModel):
    status: str
    message: str


class CollectionDeletedResponse(BaseModel):
    status: str
    collection_id: str


class TextLine(BaseModel):
    line_number: int
    content: str


class TextContentResponse(BaseModel):
    collection_id: str
    file_name: str
    total_lines: int
    offset: int
    limit: int
    has_more: bool
    lines: List[TextLine]


class CollectionFile(BaseModel):
    """created_at comes from the database row; size_bytes/modified_at from disk."""
    file_name: str
    url: str
    created_at: Optional[Union[str, datetime]] = None
    size_bytes: Optional[int] = None
    modified_at: Optional[str] = None


class CollectionFilesResponse(BaseModel):
    collection_id: str
    file_count: int
    files: List[CollectionFile]


class ChatCollectionsResponse(BaseModel):
    collections: List[Dict[str, Any]]
    count: int = 0


class ChatPreviewResponse(BaseModel):
    collection_id: str
    file_name: str
    content_preview: str
    truncated: bool
    max_chars: int


class ChatMessagesResponse(BaseModel):
    """WhatsApp sources fill `messages`; other text sources fill `lines`."""
    collection_id: str
    file_name: str
    offset: int
    limit: int
    subtype: str
    platform: str
    total: int
    has_more: bool
    messages: List[Dict[str, Any]]
    lines: List[TextLine]


class RunDeletedResponse(BaseModel):
    run_id: str
    deleted: bool


class FolderDeletedResponse(BaseModel):
    folder_id: str
    deleted: bool


class SkillDeletedResponse(BaseModel):
    skill_id: str
    deleted: bool


class SessionDeletedResponse(BaseModel):
    status: str
    session_id: str


class ConnectionActiveResponse(BaseModel):
    status: str
    connection_id: str
    active: bool


class ConnectionDeletedResponse(BaseModel):
    status: str
    message: str
    connection_id: str


class PublicLinkActiveResponse(BaseModel):
    status: str
    link_id: str
    active: bool


class PublicLinkDeletedResponse(BaseModel):
    status: str
    message: str
    link_id: str


class AvailableModelsResponse(BaseModel):
    default_provider: str
    default_model: str
    available_models: Dict[str, List[str]]
    usage_hint: str
