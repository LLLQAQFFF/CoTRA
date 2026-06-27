# Example local override for the CoTRA judge (DeepSeek-V4-pro at temperature 0).
#
# Usage:
#   cp config/llm_config.example.py llm_config.local.py
#   # then edit llm_config.local.py and set LLM_API_KEY
#
# llm_config.local.py is git-ignored and overrides llm_config.py.
# Any of these can also be set via environment variables (LLM_API_KEY,
# LLM_BASE_URL, LLM_JUDGE_MODEL, ...); see src/llm_judge/config.py.

LLM_API_KEY = "<your-api-key>"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_JUDGE_MODEL = "deepseek-v4-pro"

LLM_REASONING_EFFORT = "high"
LLM_EXTRA_BODY = {"thinking": {"type": "enabled"}}

MAX_TOKENS_ACTION = 2048
MAX_TOKENS_TRAJECTORY = 4096
CONCURRENCY_PER_TRAJECTORY = 4
ENCODER_REQUEST_TIMEOUT_SECONDS = 90.0
ENCODER_RETRY_MAX_ATTEMPTS = 2

# Fill in your provider's list price for cost accounting (optional).
PRICING_CURRENCY = "USD"
PRICING_PER_M_TOKENS = {
    "deepseek-v4-pro": {"input_cached": 0.0, "input": 0.0, "output": 0.0},
}
