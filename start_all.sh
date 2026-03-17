#!/usr/bin/env bash

# Start all external services (Docker), build indexes if needed,
# and launch the multi-hop RAG pipeline ready to receive queries.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

RUN_INGESTION=1
FORCE_INGESTION=0
USER_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --skip-ingest|--skip-ingestion)
      RUN_INGESTION=0
      ;;
    --force-ingest|--force-ingestion)
      FORCE_INGESTION=1
      ;;
    *)
      USER_ARGS+=("$arg")
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "✗ docker command not found. Please install Docker and try again." >&2
  exit 1
fi

ensure_container() {
  local name="$1"; shift
  # Remaining arguments form the docker run command
  if docker ps -a --format '{{.Names}}' | grep -qw "$name"; then
    local state
    state="$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || echo false)"
    if [ "$state" != "true" ]; then
      echo "Starting existing container: $name"
      docker start "$name" >/dev/null
    else
      echo "Container already running: $name"
    fi
  else
    echo "Creating and starting container: $name"
    "$@"
  fi
}

# GROBID (required for PDF → TEI/text extraction). No Postgres/Qdrant needed
# because the pipeline now uses local FAISS indexes only.
ensure_container grobid \
  docker run -d --name grobid \
    -p 8070:8070 -p 8071:8071 \
    lfoppiano/grobid:0.8.0

# Give services a moment to initialize
sleep 5

# Activate virtualenv if present
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

FAISS_DIR="${FAISS_INDEX_DIR:-$PROJECT_ROOT/faiss_store}"
META_PATH="$FAISS_DIR/metadata.json"

if [ "$RUN_INGESTION" -eq 0 ]; then
  echo "Skipping ingestion pipeline (per --skip-ingest flag)."
elif [ -f "$META_PATH" ] && [ "$FORCE_INGESTION" -eq 0 ]; then
  echo "FAISS index already exists at $FAISS_DIR; skipping ingestion."
  echo "Use --force-ingestion to rebuild indexes."
else
  echo "Running ingestion pipeline to build/update indexes..."

  python3 -m Extractor.TextExtraction
  python3 -m Extractor.SeparateContentReferences
  python3 -m Extractor.convert_into_text
  python3 -m Extractor.convert_into_json
  python3 -m Extractor.chunking
  python3 -m Extractor.generate_vectors

  echo "Ingestion pipeline completed."
fi

# Finally, start the multi-hop RAG pipeline.
python3 multihop_rag.py "${USER_ARGS[@]}"
