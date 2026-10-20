"""Shared runtime constants. Secrets and model selection stay in environment variables."""
from zoneinfo import ZoneInfo

SG = ZoneInfo("Asia/Singapore")
LLM_TIMEOUT_SECONDS = 20

CONTEXT_IDLE_MINUTES = 30
MAX_REQUEST_CHARS = 1000
MAX_FILTER_CHARS = 160
MAX_QUESTION_CHARS = 240
