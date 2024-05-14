"""
This file just provides constants that are shared across project
"""
SSL_KEYFILE = "/etc/letsencrypt/live/teamcare.argonautai.site/privkey.pem"
SSL_CERTFILE = "/etc/letsencrypt/live/teamcare.argonautai.site/fullchain.pem"
FAVICON_PATH = "/mnt/resources/argo-logo-compact.webp"
USERS_FILE = "/mnt/resources/users.json"
LLM_YAML_FILE = "gen_ai/llm.yaml"

PROCESSED_FILES_DIR = "/mnt/resources/uhg/main_folder"

LOGS_DIRECTORY = "gen_ai/logs"
VECTOR_STORE = "gen_ai/vector_store"

MAX_OUTPUT_TOKENS = 4000
MAX_CONTEXT_SIZE = 1025000
RETRIEVER_SCORE_THRESHOLD = 2
MEMORY_STORE_HOST = "redis.t2xservice.internal"
