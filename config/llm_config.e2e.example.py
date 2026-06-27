# Example config for the B4 end-to-end judge baseline.
#
# The end-to-end baseline may run against a different endpoint than the main
# pipeline. Copy and edit, then point the runner at it:
#   cp config/llm_config.e2e.example.py llm_config.e2e.py
#   LLM_CONFIG_EXTRA=llm_config.e2e.py PYTHONPATH=src \
#       python -m encoder_judge.cli prelabel <template> --baseline e2e-llm ...
#
# llm_config.e2e.py is git-ignored.

LLM_API_KEY = "<your-api-key>"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_JUDGE_MODEL = "deepseek-v4-pro"
LLM_REASONING_EFFORT = None
LLM_EXTRA_BODY = None

# End-to-end judging packs the whole (shortened) trajectory into one call and
# needs a large output budget.
MAX_TOKENS_E2E = 16384
E2E_REQUEST_TIMEOUT_SECONDS = 600.0
E2E_RETRY_MAX_ATTEMPTS = 6

ENCODER_REQUEST_TIMEOUT_SECONDS = 120.0
ENCODER_RETRY_MAX_ATTEMPTS = 6

PRICING_CURRENCY = "USD"
PRICING_PER_M_TOKENS = {
    "deepseek-v4-pro": {"input_cached": 0.0, "input": 0.0, "output": 0.0},
}
