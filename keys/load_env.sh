#!/usr/bin/env bash
# Source this file to export credentials into the current shell:
#   source keys/load_env.sh
KEYS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_TOKEN="$(<"$KEYS_DIR/hugging_face_token")"
export ANTHROPIC_API_KEY="$(<"$KEYS_DIR/claude_API_key.txt")"
export OPENAI_API_KEY="$(<"$KEYS_DIR/openai_API_key.txt")"
export GOOGLE_APPLICATION_CREDENTIALS="$KEYS_DIR/vertex_service_account.json"
