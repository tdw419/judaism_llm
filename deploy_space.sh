#!/usr/bin/env bash
# Deploy the Judaism LLM RAG Space to Hugging Face.
# Prerequisite: HF account with PRO (Gradio Spaces on cpu-basic require PRO)
#   or accept Static Space limitations (see README notes).
# Usage: bash deploy_space.sh
set -euo pipefail

SPACE_ID="tdw419/judaism-llm-rag"
DIR="$(cd "$(dirname "$0")" && pwd)/space_app"

python3 - <<EOF
from huggingface_hub import HfApi
api = HfApi()
res = api.create_repo(
    repo_id="${SPACE_ID}",
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
)
print("Space repo ready:", res)
EOF

python3 - <<EOF
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="${DIR}",
    repo_id="${SPACE_ID}",
    repo_type="space",
    commit_message="Deploy Sefaria RAG demo: hybrid retrieval + extractive generation",
)
print("App uploaded")
EOF

echo
echo "Deployed. Monitor the build at:"
echo "  https://huggingface.co/spaces/${SPACE_ID}"
