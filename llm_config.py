# CoTRA LLM configuration (defaults).
#
# This file holds NON-SECRET defaults that are safe to commit. Put your API key
# in `llm_config.local.py` (git-ignored) or in the LLM_API_KEY environment
# variable. Values here can also be overridden by environment variables
# (see src/llm_judge/config.py).

# --- API endpoint ---
# Leave the key empty here; supply it via llm_config.local.py or LLM_API_KEY.
LLM_API_KEY = ""
LLM_BASE_URL = "https://api.deepseek.com"
LLM_JUDGE_MODEL = "deepseek-v4-pro"
LLM_REASONING_EFFORT = "high"
LLM_EXTRA_BODY = {"thinking": {"type": "enabled"}}

# --- decoding / runtime ---
MAX_TOKENS_ACTION = 2048
MAX_TOKENS_TRAJECTORY = 4096
CONCURRENCY_PER_TRAJECTORY = 4
ENCODER_REQUEST_TIMEOUT_SECONDS = 90.0
ENCODER_RETRY_MAX_ATTEMPTS = 2

# --- pricing (for cost accounting only) ---
# The effective per-token price depends on the provider/endpoint you use.
# Fill in your provider's list price; currency is whatever you set here.
PRICING_CURRENCY = "USD"
PRICING_PER_M_TOKENS = {
    # "deepseek-v4-pro": {"input_cached": 0.0, "input": 0.0, "output": 0.0},
}
