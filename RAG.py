# Load environment variables from .env file (if present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on actual env vars

from rag_utils.metrics import stage_timer, log_stage

import os

# Default LLM configuration.
# These can be overridden via environment variables to support different providers
# including local Ollama (OpenAI-compatible) servers.
#
# For Ollama, a typical setup is:
#   export LLM_BASE_URL="http://localhost:11434/v1"
#   export LLM_MODEL="llama3.1:8b-instruct"   # or any installed Ollama model
#   (no API key required)

LLM_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
# Big, high-quality model for normal runs
LLM_MODEL_DEFAULT = os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
# Smaller, cheaper/faster model for "fast" runs
LLM_MODEL_FAST_DEFAULT = os.getenv("LLM_MODEL_FAST", "meta-llama/llama-3.1-8b-instruct")
LLM_TIMEOUT_DEFAULT_S = int(os.getenv("LLM_TIMEOUT_S", "60"))

# Shared run logging (for step-by-step visibility)
RUN_VERBOSE_DEFAULT = True
RUN_LOG_PATH: str | None = None


def _run_log(msg: str, *, verbose: bool | None = None, log_path: str | None = None) -> None:
    """Log a message to console and optionally append to a run log file."""
    v = RUN_VERBOSE_DEFAULT if verbose is None else bool(verbose)
    lp = RUN_LOG_PATH if log_path is None else log_path
    if v:
        try:
            print(msg)
        except Exception:
            pass
    if lp:
        try:
            with open(lp, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n")
        except Exception:
            pass

# Stronger-than-user instructions: sent as a SYSTEM message.
# This cannot *guarantee* the remote model won't use latent prior knowledge,
# but it improves compliance vs user-only prompts.
LLM_SYSTEM_PROMPT_EVIDENCE_ONLY = (
    "You are an evidence-bound scientific assistant. "
    "You must use ONLY the evidence explicitly provided in the user message. "
    "Do NOT use prior knowledge, common knowledge, or assumptions. "
    "Do NOT invent citations, papers, datasets, or results. "
    "If evidence is insufficient, say it is insufficient. "
    "Output must follow the requested JSON schema exactly."
)

# API Key — optional for local providers like Ollama.
# For hosted providers (OpenRouter, OpenAI, etc.) load from env.
LLM_API_KEY_HARDCODED = None  # kept for backwards compat; .env is preferred

# Fallback to environment variables if hardcoded key is empty
LLM_API_KEY_DEFAULT = LLM_API_KEY_HARDCODED or os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY") or ""

# Function to call the LLM API with a prompt and return the output
import json
import urllib.request
import urllib.error


def _truncate_text(text: str, max_chars: int) -> str:
    s = str(text or "")
    if len(s) <= int(max_chars):
        return s
    return s[: int(max_chars)] + "..."


def _safe_json_loads(s: str) -> dict | list | None:
    try:
        return json.loads(s)
    except Exception:
        return None


def _parse_json_from_llm(text: str) -> dict | list | None:
    """Parse JSON from an LLM response that may include ``` fences or extra text."""
    s = str(text or "").strip()
    if not s:
        return None

    # 1) Direct parse
    obj = _safe_json_loads(s)
    if obj is not None:
        return obj

    # 2) Try fenced code blocks
    if "```" in s:
        parts = s.split("```")
        for part in parts:
            candidate = part.strip()
            if not candidate:
                continue
            # Drop optional language tag line like "json\n"
            if "\n" in candidate:
                first, rest = candidate.split("\n", 1)
                if first.strip().lower() in {"json", "javascript", "js"}:
                    candidate = rest.strip()
            obj2 = _safe_json_loads(candidate)
            if obj2 is not None:
                return obj2

    # 3) Extract a JSON-looking substring
    start_candidates = [i for i in [s.find("{"), s.find("[")] if i != -1]
    if start_candidates:
        start = min(start_candidates)
        end = max(s.rfind("}"), s.rfind("]"))
        if end != -1 and end > start:
            sub = s[start : end + 1].strip()
            obj3 = _safe_json_loads(sub)
            if obj3 is not None:
                return obj3
    return None

# Prompt Injection
def call_llm(
    prompt,
    model: str | None = None,
    api_key=LLM_API_KEY_DEFAULT,
    base_url=LLM_URL,
    timeout=LLM_TIMEOUT_DEFAULT_S,
    *,
    system_prompt: str = LLM_SYSTEM_PROMPT_EVIDENCE_ONLY,
    temperature: float = 0.0,
    fast: bool = False,
):
    """Send a prompt to the LLM API and return the model's output.

    Notes:
    - We use a SYSTEM message to more strongly enforce evidence-only behavior.
    - `temperature=0` reduces randomness and usually improves schema compliance.
    """
    # Select model: if caller didn't override model, choose based on `fast` flag.
    if model is None:
        model = LLM_MODEL_FAST_DEFAULT if fast else LLM_MODEL_DEFAULT

    base_url = str(base_url or "").rstrip("/")
    api_key = str(api_key or "")
    is_local = base_url.startswith("http://localhost") or base_url.startswith("http://127.0.0.1")

    # Hosted providers require an API key; local providers like Ollama do not.
    if (not is_local) and not api_key.strip():
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY or LLM_API_KEY in your environment, "
            "or set LLM_BASE_URL to a local provider like Ollama."
        )

    headers = {
        "Content-Type": "application/json",
    }
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key}"
    sys = str(system_prompt or "").strip()
    messages = []
    if sys:
        messages.append({"role": "system", "content": sys})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
    }
    url = (base_url.rstrip("/") + "/chat/completions")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"LLM HTTPError {getattr(e, 'code', '')}: {body[:2000]}") from e
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}") from e

    result = json.loads(raw)
    # Extract the output text from the response
    return result["choices"][0]["message"]["content"]

import aiohttp
import asyncio

async def async_call_llm(
    prompt,
    model: str | None = None,
    api_key=LLM_API_KEY_DEFAULT,
    base_url=LLM_URL,
    timeout=LLM_TIMEOUT_DEFAULT_S,
    *,
    system_prompt: str = LLM_SYSTEM_PROMPT_EVIDENCE_ONLY,
    temperature: float = 0.0,
    fast: bool = False,
):
    """(Async) Send a prompt to the LLM API and return the model's output."""
    base_url = str(base_url or "").rstrip("/")
    api_key = str(api_key or "")
    is_local = base_url.startswith("http://localhost") or base_url.startswith("http://127.0.0.1")

    if (not is_local) and not api_key.strip():
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY or LLM_API_KEY in your environment, "
            "or set LLM_BASE_URL to a local provider like Ollama."
        )

    headers = {
        "Content-Type": "application/json",
    }
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key}"
    sys = str(system_prompt or "").strip()
    # Select model: if caller didn't override model, choose based on `fast` flag.
    if model is None:
        model = LLM_MODEL_FAST_DEFAULT if fast else LLM_MODEL_DEFAULT
    messages = []
    if sys:
        messages.append({"role": "system", "content": sys})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
    }
    url = (base_url.rstrip("/") + "/chat/completions")
    
    timeout_client = aiohttp.ClientTimeout(total=float(timeout))
    async with aiohttp.ClientSession(timeout=timeout_client) as session:
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"LLM HTTPError {resp.status}: {body[:2000]}")
                result = await resp.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}") from e

# print(query_llm("Hello, how are you?"))

# Query Refinement for RAG
def refine_query(user_question: str) -> dict:
    """
    Refines a raw user question into a context-aware research query
    suitable for high-quality retrieval in a multi-hop RAG system.

    Returns STRICT JSON-like dict:
    - refined_question: string
    - intent: string
    - constraints: list[str]
    - excluded_topics: list[str]
    - search_hint: string
    """

    prompt = f"""
You are a research question refinement agent.

ROLE:
Transform a raw user question into a precise, retrieval-optimized research query.
This step prepares the question for literature search only.

HARD RULES (STRICT):
- Do NOT answer the question.
- Do NOT propose, suggest, or imply solutions.
- Do NOT include recommendations, examples, or fixes.
- Do NOT include speculative or evaluative language.
- Preserve scientific neutrality.
- Make implicit intent and constraints explicit.
- Respond ONLY with valid JSON.
- Output MUST strictly follow the schema below.
- No markdown, no explanations, no extra text.

FIELD DEFINITIONS (READ CAREFULLY):

1) refined_question:
   - A single, well-formed research question.
    - Must expand ambiguity but remain neutral.
    - MUST be retrieval-friendly: include key technical terms that are likely to appear in papers.
    - Where helpful, include common synonyms/acronyms inline (e.g., "semi-supervised learning (SSL)").
   - Must NOT contain solution language (e.g., "use", "apply", "best method").

2) intent:
   - A precise statement of what the user wants to understand.
   - Must explicitly reflect:
     • the failure/pain point,
     • the domain (e.g., NLP, IR, ML),
     • the scope of inquiry.
   - Avoid vague phrasing like "investigate approaches" or "explore methods".

3) constraints:
   - Normalize user constraints into short, actionable phrases.
   - Include limits on:
     • model size,
     • resources,
     • scope,
     • assumptions.
   - Each constraint must be enforceable during retrieval or judging.

4) excluded_topics:
   - MUST exclude DISALLOWED SOLUTION MECHANISMS (not vague labels) EXPLAIN IT IN DETAIL.
   - Exclusions must close common loopholes.
   - If the user disallows large / huge / foundation models, exclusions MUST block:
       (a) solving primarily by switching to a larger pre-trained or foundation model,
       (b) fine-tuning or transfer learning from large pre-trained/foundation models as the main fix,
       (c) scaling parameter count instead of changing representation, architecture, or training signal.
   - Write exclusions as concrete mechanisms, not buzzwords.

5) search_hint:
    - Comma-separated retrieval keywords/phrases (aim for 12–25 terms) CONSISTENT with constraints and excluded_topics.
    - Include synonyms, abbreviations, and closely related indexing terms that improve recall.
    - Do NOT include terms that imply disallowed mechanisms.
    - Keep it short and retrieval-oriented (methods/terms), not sentences.

USER QUESTION:
{user_question}

OUTPUT SCHEMA (STRICT — FOLLOW EXACTLY):
{{
    "refined_question": "<string>",
    "intent": "<string>",
    "constraints": ["<string>", "<string>", "..."],
    "excluded_topics": ["<string>", "<string>", "..."],
    "search_hint": "<comma-separated keywords>"
}}
"""


    _run_log("[refine] Starting question refinement")
    _run_log(f"[refine] user_question (first 200 chars): {str(user_question or '')[:200].replace(chr(10),' ')}")

    # LLM Call
    response = call_llm(prompt)
    # print("LLM Response:", response)

    # Saving Response to TEMP folder
    import os
    import datetime
    response_json = _parse_json_from_llm(response)
    if response_json is None:
        _run_log("[refine] ERROR: LLM response was not valid JSON")
        print("Error: LLM response is not valid JSON.")
        print("Raw response:", response)
        return None

    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "TEMP"))
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"refined_query_{timestamp}.json"
    file_path = os.path.abspath(os.path.join(temp_dir, filename))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(response_json, f, ensure_ascii=False, indent=2)

    _run_log(f"[refine] Saved refined JSON: {file_path}")

    # Safety check: ensure file exists (helps catch odd environment/path issues)
    if not os.path.exists(file_path):
        try:
            os.makedirs(temp_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(response_json, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Keep the path so callers can reuse the same JSON later.
    if isinstance(response_json, dict):
        response_json["_saved_path"] = file_path

    return response_json


def retrieve_chunks_for_question(
    user_question: str,
    *,
    refined: dict | None = None,
    refined_json_path: str | None = None,
    top_n: int = 10,
    stage1_k: int = 100,
    stage2_k: int = 20,
    faiss_index_dir: str | None = None,
    collection_name: str = "research_papers",
):
    """Refine a user question, then retrieve top-N chunks using the refined query.

    Retrieval query strategy:
    - Use refined_question as primary query.
    - If search_hint exists, append it to help retrieval (still compatible with constraints).
    """
    if refined is None and refined_json_path:
        with open(refined_json_path, "r", encoding="utf-8") as f:
            refined = json.load(f)

    if refined is None:
        refined = refine_query(user_question)

    if not refined:
        return [], None

    refined_q = str(refined.get("refined_question") or "").strip() or str(user_question)
    hint = str(refined.get("search_hint") or "").strip()

    retrieval_query = refined_q
    if hint:
        retrieval_query = f"{refined_q}\n\nSearch hints: {hint}"

    chunks = retrieve_top_chunks(
        retrieval_query,
        top_n=top_n,
        stage1_k=stage1_k,
        stage2_k=stage2_k,
        faiss_index_dir=faiss_index_dir,
        collection_name=collection_name,
    )
    return chunks, refined

def retrieve_top_chunks(
    query: str,
    *,
    top_n: int = 10,
    stage1_k: int = 100,
    stage2_k: int = 20,
    exclude_paper_ids: set[str] | None = None,
    exclude_chunk_ids: set[str] | None = None,
    faiss_index_dir: str | None = None,
    collection_name: str = "research_papers",
):
    """Two-stage retrieval for your RAG pipeline.

    Stage 1 (recall):
    - Embed query with all-MiniLM-L6-v2 and retrieve `stage1_k` candidates from the FAISS `vector_1` index.

    Stage 2 (semantic refinement / rerank):
    - Embed query with all-mpnet-base-v2 and rerank ONLY within stage-1 candidates using the FAISS `vector_2` index.

    Returns:
    - List[dict]: top-N chunk rows from Postgres with added scores:
      `score_stage1`, `score_stage2`, `score`.
    """
    q = (query or "").strip()
    if not q:
        return []

    _run_log(f"[retrieve] Query (first 200 chars): {q[:200].replace(chr(10),' ')}")
    _run_log(f"[retrieve] Params: top_n={top_n} stage1_k={stage1_k} stage2_k={stage2_k}")

    try:
        top_n = int(top_n)
        stage1_k = int(stage1_k)
        stage2_k = int(stage2_k)
    except Exception:
        raise ValueError("top_n/stage1_k/stage2_k must be integers")

    if top_n <= 0:
        return []
    if stage1_k <= 0 or stage2_k <= 0:
        raise ValueError("stage1_k and stage2_k must be positive")
    if stage2_k < top_n:
        stage2_k = top_n
    if stage1_k < stage2_k:
        stage1_k = stage2_k

    from Extractor.database import DatabaseManager
    import numpy as np

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers") from e

    db = DatabaseManager(index_dir=faiss_index_dir, collection_name=collection_name)
    try:
        db.connect_faiss()
    except Exception as e:
        raise RuntimeError(
            f"FAISS index could not be loaded at {db.index_dir}. "
            "Ensure FAISS_INDEX_DIR points to a valid index."
        ) from e

    model_small = SentenceTransformer("BAAI/bge-small-en-v1.5")
    model_big = SentenceTransformer("all-mpnet-base-v2")
    from sentence_transformers import CrossEncoder
    model_cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    excluded_papers = exclude_paper_ids or set()
    excluded_chunks = exclude_chunk_ids or set()

    # Stage 1: recall with vector_1
    with stage_timer("retrieval_stage1", extra={"stage1_k": int(stage1_k)}):
        vec_small = model_small.encode(q, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        stage1_hits = db.search_similar(
            vec_small,
            vector_name="vector_1",
            limit=stage1_k,
            exclude_chunk_ids=excluded_chunks,
            exclude_paper_ids=excluded_papers,
        )
    _run_log(f"[retrieve] Stage1 returned {len(stage1_hits)} points")
    stage1_ids: list[str] = []
    score_stage1: dict[str, float] = {}
    for hit in stage1_hits:
        cid = str(hit.get("chunk_id") or "").strip()
        if not cid:
            continue
        stage1_ids.append(cid)
        score_stage1[cid] = float(hit.get("score", 0.0) or 0.0)
    if not stage1_ids:
        _run_log("[retrieve] Stage1 produced no chunk_ids")
        return []

    # Stage 2: rerank within stage-1 IDs using cross-encoder
    stage1_rows = []
    for cid in stage1_ids:
        if excluded_chunks and cid in excluded_chunks:
            continue
        row = db.get_chunk_by_id(cid)
        if not row:
            continue
        pid = str(row.get("paper_id") or "").strip()
        if excluded_papers and (pid in excluded_papers):
            continue
        stage1_rows.append(row)

    if not stage1_rows:
        return []

    cross_inputs = [[q, row.get("chunk_text", "")] for row in stage1_rows]
    with stage_timer("retrieval_stage2", extra={"stage2_k": int(stage2_k), "num_candidates": len(stage1_rows)}):
        cross_scores = model_cross.predict(cross_inputs)
    
    results = []
    for row, s2 in zip(stage1_rows, cross_scores):
        cid = row["chunk_id"]
        row_dict = dict(row)
        row_dict["score_stage1"] = float(score_stage1.get(cid, 0.0))
        row_dict["score_stage2"] = float(s2)
        row_dict["score"] = float(s2)
        results.append(row_dict)

        if len(results) >= int(top_n):
            # Wait, cross-encoder scores them all. We don't want to break early because they haven't been sorted!
            # We sort them first, then return top_n.
            pass

    results.sort(key=lambda r: float(r.get("score", 0.0) or 0.0), reverse=True)
    final = results[:top_n]
    log_stage(
        "retrieval",
        num_items=len(final),
        extra={"stage1_k": int(stage1_k), "stage2_k": int(stage2_k), "top_n": int(top_n)},
    )
    _run_log(f"[retrieve] Returning {len(final)} chunks")
    return final


def _save_json_temp(obj: dict, *, prefix: str) -> str:
    import os
    import datetime

    temp_dir = os.path.join(os.path.dirname(__file__), "TEMP")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.json"
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return file_path


def _make_temp_path(*, prefix: str, ext: str) -> str:
    import os
    import datetime

    temp_dir = os.path.join(os.path.dirname(__file__), "TEMP")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(temp_dir, f"{prefix}_{timestamp}.{ext.lstrip('.')}")


def _cosine_similarity(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _build_retrieval_query_from_refined(refined: dict, fallback_question: str = "") -> str:
    refined_q = str(refined.get("refined_question") or "").strip() or str(fallback_question or "").strip()
    hint = str(refined.get("search_hint") or "").strip()
    if hint:
        return f"{refined_q}\n\nSearch hints: {hint}"
    return refined_q


def judge_iterative(
    *,
    refined: dict | None = None,
    refined_json_path: str | None = None,
    user_question: str = "",
    iterations: int = 3,
    top_k_candidates: int = 5,
    similarity_threshold: float = 0.35,
    verbose: bool = True,
    log_path: str | None = None,
    exclude_already_accepted_papers: bool = True,
    stop_when_satisfied: bool = True,
    min_accepted_chunks_to_stop: int = 1,
    faiss_index_dir: str | None = None,
    collection_name: str = "research_papers",
):
    """Iterative LLM-driven judge loop.

    Your requested flow per iteration (K candidates):
    1) Use refined query JSON (no re-refinement unless you pass only user_question).
    2) Retrieve top-K candidate chunks.
    3) Send those candidates + already accepted evidence to the LLM.
    4) LLM decides which candidates to ACCEPT vs REJECT, tracks what is SOLVED vs UNSOLVED,
       and outputs the UNSOLVED portion as the next question.
    5) Next iteration repeats retrieval using the updated question.
    6) Accepted and rejected chunk IDs are excluded from future retrieval.

    Notes:
    - The `similarity_threshold` argument is kept for backward compatibility but is not
      used for acceptance decisions in this LLM-driven mode.

    Returns:
        tuple(final_judge_json: dict, accepted_chunks: list[dict])
    """
    if log_path is None:
        log_path = _make_temp_path(prefix="judge_run_log", ext="txt")

    def _log(msg: str) -> None:
        if not verbose:
            return
        try:
            print(msg)
        except Exception:
            pass
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n")
        except Exception:
            pass

    _log(f"[judge] log_path={log_path}")

    if refined is None and refined_json_path:
        _log(f"[judge] Loading refined JSON from: {refined_json_path}")
        try:
            import os

            if not os.path.exists(refined_json_path):
                _log(f"[judge] WARNING: refined_json_path not found on disk: {refined_json_path}")
            else:
                with open(refined_json_path, "r", encoding="utf-8") as f:
                    refined = json.load(f)
        except Exception:
            refined = None
    if refined is None:
        if not user_question:
            raise ValueError("Provide refined/refined_json_path or user_question")
        _log("[judge] No refined JSON provided; refining user_question now...")
        with stage_timer("llm_refine_query"):
            refined = refine_query(user_question)
    if not refined:
        raise ValueError("Refined query JSON is empty/invalid")

    accepted_chunks: list[dict] = []
    accepted_chunk_ids: set[str] = set()
    accepted_paper_ids: set[str] = set()
    rejected_chunk_ids: set[str] = set()
    rejected_paper_ids: set[str] = set()

    solved_parts_so_far: list[str] = []

    current_refined = dict(refined)
    last_judge_json: dict | None = None
    iteration_summaries: list[dict] = []

    rq0 = str(current_refined.get("refined_question") or "").strip()
    _log("[judge] Starting iterative judge (LLM accept/reject mode)")
    _log(f"[judge] iterations={iterations} top_k_candidates={top_k_candidates}")
    if rq0:
        _log(f"[judge] refined_question: {rq0}")

    def _dedup_str_list(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            s = str(x or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _pack_evidence_for_llm(rows: list[dict], *, max_chars: int = 900) -> list[dict]:
        packed = []
        for r in rows or []:
            text = str(r.get("chunk_text") or "").strip().replace("\n", " ")
            packed.append(
                {
                    "chunk_id": r.get("chunk_id"),
                    "paper_id": r.get("paper_id"),
                    "section": r.get("section"),
                    "year": r.get("year"),
                    "text": _truncate_text(text, int(max_chars)),
                }
            )
        return packed

    def _pack_candidates_for_llm(rows: list[dict], *, max_chars: int = 1400) -> list[dict]:
        packed = []
        for r in rows or []:
            text = str(r.get("chunk_text") or "").strip().replace("\n", " ")
            packed.append(
                {
                    "chunk_id": r.get("chunk_id"),
                    "paper_id": r.get("paper_id"),
                    "section": r.get("section"),
                    "year": r.get("year"),
                    "retr_score": float(r.get("score") or 0.0),
                    "text": _truncate_text(text, int(max_chars)),
                }
            )
        return packed

    try:
        min_accepted_chunks_to_stop = int(min_accepted_chunks_to_stop)
    except Exception:
        min_accepted_chunks_to_stop = 1
    if min_accepted_chunks_to_stop < 1:
        min_accepted_chunks_to_stop = 1

    for it in range(1, max(1, int(iterations)) + 1):
        _log(f"\n[judge] ===== Iteration {it} =====")
        retrieval_query = _build_retrieval_query_from_refined(current_refined, fallback_question=user_question)
        if not retrieval_query:
            # Keep going (up to max iterations). This should be rare because we fall back to user_question.
            _log("[judge] Empty retrieval_query; continuing.")
            retrieval_query = str(user_question or "").strip()

        sh = str(current_refined.get("search_hint") or "").strip()
        _log(f"[judge] retrieval_query (first 200 chars): {retrieval_query[:200].replace(chr(10),' ')}")
        if sh:
            _log(f"[judge] search_hint: {sh}")

        # Exclude rejected papers, and optionally exclude already-accepted papers to force new papers.
        exclude_papers = set(rejected_paper_ids)
        if exclude_already_accepted_papers:
            exclude_papers |= set(accepted_paper_ids)
        exclude_chunks = set(accepted_chunk_ids) | set(rejected_chunk_ids)

        # Retrieve candidates (top_k_candidates) using your 2-stage retriever
        with stage_timer("retrieval_iteration", extra={"iteration": it, "top_k_candidates": int(top_k_candidates)}):
            candidates = retrieve_top_chunks(
                retrieval_query,
                top_n=int(top_k_candidates),
                stage1_k=max(50, int(top_k_candidates) * 10),
                stage2_k=max(20, int(top_k_candidates) * 5),
                exclude_paper_ids=exclude_papers,
                exclude_chunk_ids=exclude_chunks,
                faiss_index_dir=faiss_index_dir,
                collection_name=collection_name,
            )

        # If retrieval yields no candidates, do NOT stop early.
        # Let the LLM reframe the next_question/search_hint to broaden retrieval, and continue until max iterations.
        if not candidates:
            _log("[judge] No candidates retrieved; continuing (LLM will be asked to broaden the query).")
            candidates = []

        _log(f"[judge] Retrieved {len(candidates)} candidate chunks")
        for idx, c in enumerate(candidates or [], 1):
            ct = str(c.get("chunk_text") or "").replace("\n", " ").strip()
            if len(ct) > 160:
                ct = ct[:160] + "..."
            _log(
                f"[judge] cand#{idx}: paper_id={c.get('paper_id')} chunk_id={c.get('chunk_id')} "
                f"retr_score={float(c.get('score') or 0.0):.4f} section={c.get('section')} :: {ct}"
            )
        compact_accepted = _pack_evidence_for_llm(accepted_chunks, max_chars=900)
        compact_candidates = _pack_candidates_for_llm(candidates or [], max_chars=1400)

        prompt = f"""
You are a retrieval judge for a scientific RAG system.

GOAL:
Given:
  (A) the current refined query JSON,
  (B) evidence chunks already accepted so far,
  (C) the TOP-K newly retrieved candidate chunks for this iteration,
decide:
  1) Which candidate chunks should be ACCEPTED as helpful evidence.
  2) Which candidate chunks should be REJECTED as not helpful.
  3) What parts of the question are now SOLVED (supported by accepted evidence).
  4) What parts remain UNSOLVED.
  5) Output the UNSOLVED part as the next question (for the next iteration).

CRITICAL HARD RULES:
- Output ONLY valid JSON. No markdown. No commentary.
- Do NOT answer the user's research question.
- Do NOT propose fixes/solutions.
- Do NOT use ANY prior knowledge.
- You may ONLY use information that appears in ACCEPTED_EVIDENCE and CANDIDATE_CHUNKS.
- Accept/reject decisions MUST reference ONLY the provided CANDIDATE_CHUNKS.
- Do NOT accept any chunk_id that is already in ALREADY_ACCEPTED_CHUNK_IDS or ALREADY_REJECTED_CHUNK_IDS.
- You MAY reject MULTIPLE chunks in a single iteration.
- You MAY reject ENTIRE papers by listing paper_id(s) in rejected_paper_ids.
- Do NOT include any paper_id in rejected_paper_ids if it is already in ALREADY_ACCEPTED_PAPER_IDS or ALREADY_REJECTED_PAPER_IDS.
- If CANDIDATE_CHUNKS is empty, you MUST set satisfied=false, and you MUST propose a broader NEXT_QUESTION and search_hint.
- If CANDIDATE_CHUNKS is empty, accept and reject MUST be empty lists.
- If evidence is insufficient, say so in UNSOLVED_PARTS (e.g., "insufficient evidence for X").
- If satisfied is false, NEXT_QUESTION must be materially different from PREVIOUS_REFINED_QUESTION.
- NEXT_QUESTION MUST be retrieval-friendly: include key technical terms and important keywords.
- search_hint MUST be 10–25 comma-separated keywords/phrases to improve the next retrieval round.

INPUT_REFINED_JSON:
{json.dumps(current_refined, ensure_ascii=False)}

PREVIOUS_REFINED_QUESTION:
{str(current_refined.get("refined_question") or "").strip()}

ALREADY_ACCEPTED_CHUNK_IDS:
{sorted(list(accepted_chunk_ids))}

ALREADY_ACCEPTED_PAPER_IDS:
{sorted(list(accepted_paper_ids))}

ALREADY_REJECTED_CHUNK_IDS:
{sorted(list(rejected_chunk_ids))}

ALREADY_REJECTED_PAPER_IDS:
{sorted(list(rejected_paper_ids))}

SOLVED_PARTS_SO_FAR (high-level notes):
{json.dumps(solved_parts_so_far, ensure_ascii=False)}

ACCEPTED_EVIDENCE:
{json.dumps(compact_accepted, ensure_ascii=False)}

CANDIDATE_CHUNKS (TOP-K):
{json.dumps(compact_candidates, ensure_ascii=False)}

OUTPUT SCHEMA (STRICT):
{{
  "satisfied": <boolean>,
  "solved_parts": ["<string>", "..."],
  "unsolved_parts": ["<string>", "..."],
  "next_question": "<string>",
  "accept": [{{"chunk_id": "<string>", "paper_id": "<string>", "rationale": "<short>"}}],
  "reject": [{{"chunk_id": "<string>", "paper_id": "<string>", "reason": "<short>"}}],
    "rejected_paper_ids": ["<paper_id>", "<paper_id>", "..."],
  "search_hint": "<comma-separated keywords>"
}}
"""

        _log("[judge] Calling LLM to accept/reject top-K candidates + produce next_question...")

        with stage_timer("llm_judge_iteration", extra={"iteration": it, "num_candidates": len(candidates)}):
            llm_out = call_llm(prompt)
        lo = str(llm_out or "")
        _log(f"[judge] LLM raw output (first 600 chars): {lo[:600].replace(chr(10),' ')}")
        judge_json = _parse_json_from_llm(llm_out)
        if judge_json is None or not isinstance(judge_json, dict):
            judge_json = {
                "error": "LLM judge output was not valid JSON",
                "raw": llm_out,
                "satisfied": False,
                "solved_parts": [],
                "unsolved_parts": ["LLM output was not valid JSON"],
                "next_question": str(current_refined.get("refined_question") or "").strip(),
                "accept": [],
                "reject": [],
                "search_hint": str(current_refined.get("search_hint") or "").strip(),
            }

        # Normalize and apply decisions
        accept_list = judge_json.get("accept") or []
        reject_list = judge_json.get("reject") or []
        if not isinstance(accept_list, list):
            accept_list = []
        if not isinstance(reject_list, list):
            reject_list = []

        candidate_by_id = {
            str(c.get("chunk_id") or "").strip(): c for c in (candidates or []) if str(c.get("chunk_id") or "").strip()
        }

        accepted_this_round: list[dict] = []
        rejected_this_round: list[dict] = []

        for a in accept_list:
            if not isinstance(a, dict):
                continue
            cid = str(a.get("chunk_id") or "").strip()
            if not cid:
                continue
            if cid in accepted_chunk_ids or cid in rejected_chunk_ids:
                continue
            row = candidate_by_id.get(cid)
            if not row:
                continue
            out_row = dict(row)
            out_row["llm_judge"] = {
                "iteration": it,
                "decision": "accept",
                "rationale": str(a.get("rationale") or "").strip(),
            }
            accepted_chunks.append(out_row)
            accepted_this_round.append({"chunk_id": out_row.get("chunk_id"), "paper_id": out_row.get("paper_id")})
            accepted_chunk_ids.add(cid)
            pid = str(out_row.get("paper_id") or "").strip()
            if pid:
                accepted_paper_ids.add(pid)

        for r in reject_list:
            if not isinstance(r, dict):
                continue
            cid = str(r.get("chunk_id") or "").strip()
            if not cid:
                continue
            if cid in accepted_chunk_ids or cid in rejected_chunk_ids:
                continue
            # Only allow rejecting provided candidates
            if cid not in candidate_by_id:
                continue
            rejected_chunk_ids.add(cid)
            rejected_this_round.append(
                {
                    "chunk_id": cid,
                    "paper_id": str(r.get("paper_id") or candidate_by_id.get(cid, {}).get("paper_id") or "").strip(),
                    "reason": str(r.get("reason") or "").strip(),
                }
            )

        # Update solved/unsolved tracking
        sp = judge_json.get("solved_parts") or []
        up = judge_json.get("unsolved_parts") or []
        if not isinstance(sp, list):
            sp = []
        if not isinstance(up, list):
            up = []
        solved_parts_so_far = _dedup_str_list(solved_parts_so_far + [str(x) for x in sp])

        # Update rejected papers if the model included them (optional field)
        try:
            new_rej_papers = judge_json.get("rejected_paper_ids") or []
            if isinstance(new_rej_papers, list):
                for pid in new_rej_papers:
                    pid_s = str(pid or "").strip()
                    # Never allow a paper to be rejected if it was already accepted as evidence.
                    if pid_s and (pid_s not in accepted_paper_ids):
                        rejected_paper_ids.add(pid_s)
        except Exception:
            pass

        # Allow (optional) per-chunk paper rejection via reject items.
        # If the model marks a rejected chunk as implying the entire paper is irrelevant,
        # it can set reject_paper=true and we will exclude the paper from future retrieval.
        try:
            for r in reject_list:
                if not isinstance(r, dict):
                    continue
                if bool(r.get("reject_paper")) is not True:
                    continue
                pid_s = str(r.get("paper_id") or "").strip()
                if pid_s and (pid_s not in accepted_paper_ids):
                    rejected_paper_ids.add(pid_s)
        except Exception:
            pass

        # Attach authoritative state
        judge_json["iteration"] = it
        judge_json["accepted_chunks"] = [
            {"chunk_id": c.get("chunk_id"), "paper_id": c.get("paper_id")} for c in accepted_chunks
        ]
        judge_json["accepted_chunk_ids"] = sorted(list(accepted_chunk_ids))
        judge_json["accepted_paper_ids"] = sorted(list(accepted_paper_ids))
        judge_json["rejected_chunk_ids"] = sorted(list(rejected_chunk_ids))
        judge_json["rejected_paper_ids"] = sorted(list(rejected_paper_ids))
        judge_json["accepted_this_round"] = accepted_this_round
        judge_json["rejected_this_round"] = rejected_this_round
        judge_json["solved_parts_so_far"] = solved_parts_so_far

        # Optional: produce a short evidence-grounded "what we learned this iteration" summary.
        # This is NOT a final answer; it only summarizes what the currently accepted evidence supports.
        try:
            mini_prompt = f"""
You are a scientific assistant.

TASK:
Write a brief evidence-grounded summary of what the ACCEPTED_EVIDENCE currently supports about the question.

CRITICAL RULES:
- Use ONLY ACCEPTED_EVIDENCE.
- Do NOT use prior knowledge.
- Do NOT propose fixes/solutions.
- If evidence is insufficient, say so.
- Output ONLY valid JSON.

QUESTION (current):
{str(current_refined.get('refined_question') or retrieval_query).strip()}

ACCEPTED_EVIDENCE:
{json.dumps(compact_accepted, ensure_ascii=False)}

OUTPUT SCHEMA:
{{
  "evidence_summary": ["<string>", "<string>", "..."],
  "evidence_citations": [{{"paper_id": "<string>", "chunk_id": "<string>", "supports": "<short>"}}]
}}
"""
            with stage_timer("llm_iteration_summary", extra={"iteration": it}):
                mini_out = call_llm(mini_prompt)
            mini_obj = _parse_json_from_llm(mini_out)
            if isinstance(mini_obj, dict):
                judge_json["iteration_evidence_summary"] = mini_obj
        except Exception:
            pass
        judge_json["_debug"] = {
            "retrieval_query": retrieval_query,
            "candidates": [
                {
                    "paper_id": c.get("paper_id"),
                    "chunk_id": c.get("chunk_id"),
                    "retr_score": float(c.get("score") or 0.0),
                }
                for c in candidates
            ],
            "excluded_chunk_ids": sorted(list(exclude_chunks)),
            "excluded_paper_ids": sorted(list(exclude_papers)),
            "raw_llm_output_first_2000_chars": _truncate_text(lo, 2000),
        }

        saved_path = _save_json_temp(judge_json, prefix=f"judge_iter_{it}")
        judge_json["_saved_path"] = saved_path
        last_judge_json = judge_json

        iteration_summaries.append(
            {
                "iteration": it,
                "satisfied": bool(judge_json.get("satisfied")),
                "solved_parts": judge_json.get("solved_parts") or [],
                "unsolved_parts": judge_json.get("unsolved_parts") or [],
                "accepted_this_round": judge_json.get("accepted_this_round") or [],
                "rejected_this_round": judge_json.get("rejected_this_round") or [],
                "next_question": str(judge_json.get("next_question") or "").strip(),
                "search_hint": str(judge_json.get("search_hint") or "").strip(),
                "rejected_chunk_ids": judge_json.get("rejected_chunk_ids") or [],
                "saved_path": saved_path,
            }
        )

        _log(f"[judge] Saved judge JSON: {saved_path}")
        _log(f"[judge] satisfied={bool(judge_json.get('satisfied'))}")

        _log(
            f"[judge] Accepted this iter: {len(accepted_this_round)} | Rejected this iter: {len(rejected_this_round)} | "
            f"Total accepted: {len(accepted_chunk_ids)} | Total rejected: {len(rejected_chunk_ids)}"
        )

        # Update refined question for next iteration.
        next_q = str(judge_json.get("next_question") or "").strip()
        prev_q = str(current_refined.get("refined_question") or "").strip()
        if next_q and (not bool(judge_json.get("satisfied"))):
            if next_q.strip() == prev_q.strip():
                # Force a shift if model repeats itself.
                uns = judge_json.get("unsolved_parts") or []
                if isinstance(uns, list) and uns:
                    uns_txt = "; ".join(str(x).strip() for x in uns if str(x).strip())
                    if uns_txt:
                        next_q = f"{next_q.strip()} (focus on unresolved: {uns_txt})"
        if next_q:
            current_refined["refined_question"] = next_q

        sh2 = str(judge_json.get("search_hint") or "").strip()
        if sh2:
            current_refined["search_hint"] = sh2

        # Early stop if satisfied
        if bool(judge_json.get("satisfied")) is True:
            # Only stop early if we actually have accepted evidence.
            if len(accepted_chunk_ids) < int(min_accepted_chunks_to_stop):
                _log(
                    "[judge] LLM marked satisfied=True but accepted evidence is insufficient; continuing. "
                    f"(accepted={len(accepted_chunk_ids)} < min_accepted_chunks_to_stop={int(min_accepted_chunks_to_stop)})"
                )
                judge_json["satisfied"] = False
            else:
                # Only stop when the judge claims nothing is left unsolved.
                up = judge_json.get("unsolved_parts") or []
                if not isinstance(up, list):
                    up = []
                up_clean = [str(x).strip() for x in up if str(x).strip()]
                if up_clean:
                    _log(
                        "[judge] LLM marked satisfied=True but unsolved_parts is not empty; continuing. "
                        f"(unsolved_parts_count={len(up_clean)})"
                    )
                    judge_json["satisfied"] = False
                elif bool(stop_when_satisfied) is True:
                    _log("[judge] satisfied=True, unsolved_parts empty, and accepted evidence sufficient; stopping early.")
                    break
                else:
                    _log("[judge] LLM marked satisfied=True; continuing because stop_when_satisfied=False")

    if last_judge_json is None:
        last_judge_json = {
            "refined_question": str(current_refined.get("refined_question") or "").strip(),
            "search_hint": str(current_refined.get("search_hint") or "").strip(),
            "accepted_chunks": [{"chunk_id": c.get("chunk_id"), "paper_id": c.get("paper_id")} for c in accepted_chunks],
            "accepted_paper_ids": sorted(list(accepted_paper_ids)),
            "rejected_chunk_ids": sorted(list(rejected_chunk_ids)),
            "rejected_paper_ids": sorted(list(rejected_paper_ids)),
            "solved_parts": solved_parts_so_far,
            "satisfied": False,
        }
        last_judge_json["_saved_path"] = _save_json_temp(last_judge_json, prefix="judge_final")

    # Attach a clear, precise summary (what happened across all iterations)
    judge_summary = {
        "iterations_requested": int(iterations),
        "iterations_completed": len(iteration_summaries) if iteration_summaries else (1 if last_judge_json else 0),
        "satisfied": bool((last_judge_json or {}).get("satisfied")),
        "final_refined_question": str(current_refined.get("refined_question") or "").strip(),
        "final_intent": str(current_refined.get("intent") or "").strip(),
        "final_constraints": current_refined.get("constraints") if isinstance(current_refined.get("constraints"), list) else [],
        "final_excluded_topics": current_refined.get("excluded_topics") if isinstance(current_refined.get("excluded_topics"), list) else [],
        "final_search_hint": str(current_refined.get("search_hint") or "").strip(),
        "accepted_chunk_count": len(accepted_chunks),
        "accepted_paper_count": len(accepted_paper_ids),
        "accepted_paper_ids": sorted(list(accepted_paper_ids)),
        "accepted_chunk_ids": sorted(list(accepted_chunk_ids)),
        "rejected_chunk_count": len(rejected_chunk_ids),
        "rejected_chunk_ids": sorted(list(rejected_chunk_ids)),
        "rejected_paper_ids": sorted(list(rejected_paper_ids)),
        "solved_parts": solved_parts_so_far,
        "unsolved_parts_by_iteration": [
            {"iteration": x.get("iteration"), "unsolved_parts": x.get("unsolved_parts") or []} for x in iteration_summaries
        ],
        "log_path": log_path,
    }
    if isinstance(last_judge_json, dict):
        last_judge_json["judge_summary"] = judge_summary
        last_judge_json["_iteration_summaries"] = iteration_summaries

    return last_judge_json, accepted_chunks


def multihop_plan_subproblems(*, refined_question: str, missing_factors: list[str] | None = None, max_subproblems: int = 3) -> list[dict]:
    """Plan subproblems to answer the main question (multi-hop), without judging.

    This is inspired by multihop_rag.py PROMPT_PLANNER, but simplified.
    """
    mf = [str(x).strip() for x in (missing_factors or []) if str(x).strip()]
    mf_block = "\n".join([f"- {x}" for x in mf]) if mf else "(none provided)"

    _run_log(f"[multihop] Planning subproblems (max={int(max_subproblems)})")
    _run_log(f"[multihop] Main question (first 200 chars): {str(refined_question or '')[:200].replace(chr(10),' ')}")
    _run_log(f"[multihop] Missing factors count: {len(mf)}")

    prompt = f"""
You are a multi-hop planner for a scientific RAG system.

TASK:
Break the main research question into a small set of sub-questions that, when answered, fully support a final solution.
Focus on unresolved aspects first.

MAIN QUESTION:
{refined_question}

UNRESOLVED / MISSING FACTORS (if any):
{mf_block}

RULES:
- Output ONLY valid JSON (no markdown).
- Do NOT use any external / pretrained model knowledge to introduce new domain concepts.
- Only reframe the MAIN QUESTION and the provided MISSING FACTORS into retrievable subquestions.
- Produce at most {int(max_subproblems)} subproblems.
- Subproblems must be distinct and non-overlapping.
- Each subproblem must be answerable by retrieving paper chunks.
- Each subproblem MUST be retrieval-friendly: include key technical terms likely to appear in papers.
- Each subproblem MUST include a short comma-separated search_hint (8–20 terms) to improve chunk retrieval.

OUTPUT SCHEMA:
{{
  "subproblems": [
        {{"id": "SP1", "question": "<string>", "rationale": "<string>", "search_hint": "<comma-separated keywords>"}},
        {{"id": "SP2", "question": "<string>", "rationale": "<string>", "search_hint": "<comma-separated keywords>"}}
  ]
}}
"""

    out = call_llm(prompt)
    obj = _parse_json_from_llm(out)
    if not isinstance(obj, dict):
        return []
    sps = obj.get("subproblems") or []
    if not isinstance(sps, list):
        return []
    cleaned: list[dict] = []
    for sp in sps[: max(1, int(max_subproblems))]:
        if not isinstance(sp, dict):
            continue
        sid = str(sp.get("id") or "").strip() or f"SP{len(cleaned)+1}"
        q = str(sp.get("question") or "").strip()
        if not q:
            continue
        sh = str(sp.get("search_hint") or "").strip()
        cleaned.append({"id": sid, "question": q, "rationale": str(sp.get("rationale") or "").strip(), "search_hint": sh})
    return cleaned


def multihop_retrieve_subproblem_chunks(
    subproblems: list[dict],
    *,
    exclude_paper_ids: set[str] | None = None,
    exclude_chunk_ids: set[str] | None = None,
    per_subproblem_top_n: int = 3,
    faiss_index_dir: str | None = None,
    collection_name: str = "research_papers",
):
    """Retrieve chunks per subproblem using the existing two-stage retriever."""
    results: dict[str, list[dict]] = {}
    ex_papers = exclude_paper_ids or set()
    ex_chunks = exclude_chunk_ids or set()
    for sp in subproblems:
        sid = str((sp or {}).get("id") or "").strip() or f"SP{len(results)+1}"
        question = str((sp or {}).get("question") or "").strip()
        if not question:
            continue
        sh = str((sp or {}).get("search_hint") or "").strip()
        q = f"{question}\n\nSearch hints: {sh}" if sh else question
        _run_log(f"[multihop] Retrieve subproblem {sid}: {question[:140]}")
        chunks = retrieve_top_chunks(
            q,
            top_n=int(per_subproblem_top_n),
            stage1_k=max(60, int(per_subproblem_top_n) * 20),
            stage2_k=max(30, int(per_subproblem_top_n) * 10),
            exclude_paper_ids=set(ex_papers),
            exclude_chunk_ids=set(ex_chunks),
            faiss_index_dir=faiss_index_dir,
            collection_name=collection_name,
        )
        results[sid] = chunks
        _run_log(f"[multihop] Subproblem {sid}: retrieved {len(chunks)} chunks")
    return results


def synthesize_final_solution(
    *,
    user_question: str,
    judge_final_json: dict,
    accepted_chunks: list[dict],
    subproblem_chunks: dict[str, list[dict]] | None = None,
    subproblem_answers: dict[str, dict] | None = None,
    max_chunk_chars: int = 2200,
):
    """Produce a final solution using judge + multihop evidence.

    This is inspired by multihop_rag.py PROMPT_SYNTHESIZE style (evidence-grounded synthesis),
    but does not use its chunk-judging pipeline.
    """
    sp_chunks = subproblem_chunks or {}
    sp_answers = subproblem_answers or {}

    _run_log("[final] Synthesizing final solution")
    _run_log(f"[final] Accepted chunks: {len(accepted_chunks or [])} | Subproblems: {len(sp_chunks or {})}")

    def _pack_chunk(c: dict) -> dict:
        return {
            "paper_id": c.get("paper_id"),
            "chunk_id": c.get("chunk_id"),
            "section": c.get("section"),
            "year": c.get("year"),
            "text": _truncate_text(str(c.get("chunk_text") or "").strip(), int(max_chunk_chars)),
        }

    packed_accepted = [_pack_chunk(c) for c in (accepted_chunks or [])]
    packed_sub = {
        str(k): [_pack_chunk(c) for c in (v or [])]
        for k, v in (sp_chunks or {}).items()
    }

    judge_summary = judge_final_json.get("judge_summary") or {}
    required_keywords: list[str] = []
    try:
        hint = str(judge_summary.get("final_search_hint") or "").strip()
        if hint:
            required_keywords.extend([x.strip() for x in hint.split(",") if x.strip()])
    except Exception:
        pass
    try:
        up = judge_summary.get("unsolved_parts_by_iteration") or []
        if isinstance(up, list) and up:
            last = up[-1] or {}
            for x in (last.get("unsolved_parts") or []):
                xs = str(x).strip()
                if xs:
                    required_keywords.append(xs)
    except Exception:
        pass
    try:
        cons = judge_summary.get("final_constraints") or []
        if isinstance(cons, list):
            for x in cons:
                xs = str(x).strip()
                if xs:
                    required_keywords.append(xs)
    except Exception:
        pass

    # De-dup while preserving order
    seen_kw: set[str] = set()
    required_keywords = [k for k in required_keywords if not (k in seen_kw or seen_kw.add(k))]

    prompt = f"""
You are a scientific assistant.

TASK:
Write a detailed final solution/answer to the user's question.
You MUST ground the answer ONLY in the provided paper chunks as evidence.
Do NOT use any external / pretrained model knowledge.
If evidence is insufficient, you MUST state the limitation under unknowns and avoid making the claim.

Your answer MUST explicitly cover the provided keywords/phrases and connect them to the user's problem.

RULES:
- Do NOT invent citations.
- Every major claim must reference chunk identifiers (paper_id + chunk_id) from the evidence.
- Do NOT include general knowledge, common knowledge, or background explanations unless the evidence chunks explicitly support them.
- Output ONLY valid JSON.

STYLE REQUIREMENTS:
- final_answer_paragraphs must contain 4–10 paragraphs and be explanatory (aim for ~500–1200 words total).
- Include a compact "keyword_coverage" list and ensure each required keyword appears at least once.
- Include practical, specific explanation of mechanisms (NOT just a one-line summary).
- IMPORTANT: To keep JSON valid, do NOT include literal newline characters inside any JSON string value.
    If you need paragraphs, use the final_answer_paragraphs array.

USER_QUESTION:
{user_question}

JUDGE_SUMMARY_JSON:
{json.dumps(judge_summary, ensure_ascii=False)}

REQUIRED_KEYWORDS_PHRASES:
{json.dumps(required_keywords, ensure_ascii=False)}

ACCEPTED_EVIDENCE_CHUNKS:
{json.dumps(packed_accepted, ensure_ascii=False)}

MULTIHOP_SUBPROBLEM_CHUNKS:
{json.dumps(packed_sub, ensure_ascii=False)}

SUBPROBLEM_ANSWERS_JSON (evidence-only; may be empty):
{json.dumps(sp_answers, ensure_ascii=False)}

OUTPUT SCHEMA:
{{
    "final_answer_paragraphs": ["<string>", "<string>", "..."],
    "key_points": ["<string>", "<string>", "..."],
    "keyword_coverage": [{{"keyword": "<string>", "where_used": "<short description>"}}],
    "citations": [{{"paper_id": "<string>", "chunk_id": "<string>", "supports": "<short reason>"}}],
    "evidence_backed_sections": [{{"title": "<string>", "content": "<string>", "citations": [{{"paper_id": "<string>", "chunk_id": "<string>"}}]}}],
    "unknowns": ["<string>", "..."],
    "used_chunk_ids": ["<chunk_id>", "..."],
    "used_paper_ids": ["<paper_id>", "..." ],
    "used_papers": [{{"paper_id": "<paper_id>", "used_chunk_ids": ["<chunk_id>", "..."], "notes": "<short>"}}]
}}
"""

    out = call_llm(prompt)
    obj = _parse_json_from_llm(out)
    if not isinstance(obj, dict):
        obj = {
            "final_answer_paragraphs": [str(out or "")],
            "key_points": [],
            "keyword_coverage": [],
            "citations": [],
            "evidence_backed_sections": [],
            "unknowns": ["LLM did not return valid JSON"],
            "used_chunk_ids": [],
            "used_paper_ids": [],
        }

    # Back-compat: if model returns final_answer as a string, convert to paragraphs.
    if "final_answer_paragraphs" not in obj:
        fa = obj.get("final_answer")
        if isinstance(fa, str) and fa.strip():
            obj["final_answer_paragraphs"] = [line.strip() for line in fa.split("\n") if line.strip()]
        elif isinstance(fa, list):
            obj["final_answer_paragraphs"] = [str(x).strip() for x in fa if str(x).strip()]
        else:
            obj["final_answer_paragraphs"] = []
    obj["_raw"] = _truncate_text(str(out or ""), 4000)
    obj["_required_keywords_phrases"] = required_keywords
    _run_log("[final] Final synthesis completed")
    return obj


def answer_subproblems(
    *,
    subproblems: list[dict],
    subproblem_chunks: dict[str, list[dict]],
    max_chunk_chars: int = 1800,
) -> dict[str, dict]:
    """Answer each subproblem using ONLY its retrieved chunks concurrently.

    Returns dict keyed by subproblem id -> answer JSON.
    """
    
    async def _process_subproblem(sp: dict) -> tuple[str, dict]:
        sid = str(sp.get("id") or "").strip()
        question = str(sp.get("question") or "").strip()
        if not sid or not question:
            return sid, {}

        chunks = subproblem_chunks.get(sid) or []
        packed = [_pack_chunk(c) for c in chunks]

        prompt = f"""
You are a scientific assistant.

TASK:
Answer ONE subproblem question using ONLY the provided paper chunks.

CRITICAL RULES:
- Use ONLY the provided chunks as evidence.
- Do NOT use prior knowledge.
- Do NOT invent citations.
- If evidence is insufficient, say so under unknowns.
- Output ONLY valid JSON. No markdown.

SUBPROBLEM_ID:
{sid}

SUBPROBLEM_QUESTION:
{question}

EVIDENCE_CHUNKS:
{json.dumps(packed, ensure_ascii=False)}

OUTPUT SCHEMA:
{{
  "subproblem_id": "<string>",
  "question": "<string>",
  "answer_paragraphs": ["<string>", "<string>", "..."],
  "key_points": ["<string>", "<string>", "..."],
  "citations": [{{"paper_id": "<string>", "chunk_id": "<string>", "supports": "<short reason>"}}],
  "unknowns": ["<string>", "..."],
  "used_chunk_ids": ["<chunk_id>", "..."],
  "used_paper_ids": ["<paper_id>", "..."],
  "sufficient": <boolean>
}}
"""
        _run_log(f"[multihop] Answering subproblem {sid} (evidence chunks={len(packed)})")
        try:
            out = await async_call_llm(prompt)
            obj = _parse_json_from_llm(out)
        except Exception as e:
            _run_log(f"[multihop] Error subproblem {sid}: {e}")
            out = ""
            obj = None

        if not isinstance(obj, dict):
            obj = {
                "subproblem_id": sid,
                "question": question,
                "answer_paragraphs": [str(out or "")],
                "key_points": [],
                "citations": [],
                "unknowns": ["LLM did not return valid JSON"],
                "used_chunk_ids": [],
                "used_paper_ids": [],
                "sufficient": False,
            }
        _run_log(f"[multihop] Subproblem {sid} answered (valid_json={isinstance(obj, dict)})")
        return sid, obj

    def _pack_chunk(c: dict) -> dict:
        return {
            "paper_id": c.get("paper_id"),
            "chunk_id": c.get("chunk_id"),
            "section": c.get("section"),
            "year": c.get("year"),
            "text": _truncate_text(str(c.get("chunk_text") or "").strip(), int(max_chunk_chars)),
        }

    async def _run_all():
        tasks = []
        for sp in subproblems or []:
            if isinstance(sp, dict):
                tasks.append(_process_subproblem(sp))
        results = await asyncio.gather(*tasks)
        return {sid: obj for sid, obj in results if sid}

    return asyncio.run(_run_all())



if __name__ == "__main__":
    # Create a single run log file so you can inspect every step.
    RUN_LOG_PATH = _make_temp_path(prefix="full_run", ext="txt")
    _run_log(f"[run] Full log will be written to: {RUN_LOG_PATH}")

    q = "I’m training a deep neural network and I’m seeing very unstable gradients in the first few epochs. The loss oscillates a lot and sometimes diverges. I want a principled way to stabilize training without redesigning the architecture. What can I do?"
    _run_log("[run] Starting end-to-end pipeline")
    refined = refine_query(q)
    print(refined)

    # Reuse the same refined JSON/dict (do NOT refine twice)
    chunks, _ = retrieve_chunks_for_question(q, refined=refined, top_n=10, stage1_k=50, stage2_k=20)
    for i, ch in enumerate(chunks, 1):
        preview = str(ch.get("chunk_text") or "")
        preview = preview.replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:220] + "..."
        print(
            f"\n{i}. paper_id={ch.get('paper_id')} chunk_id={ch.get('chunk_id')} "
            f"score={ch.get('score'):.4f} year={ch.get('year')} section={ch.get('section')}"
        )
        print(f"   {preview}")

    # Example: run judge loop using the refined dict directly (most robust)
    if refined:
        # Ask user how many iterations (MAX) to run before final answer.
        try:
            raw = input("\nMax judge iterations to run before answering? (default 3): ").strip()
        except Exception:
            raw = ""
        try:
            iterations = int(raw) if raw else 3
        except Exception:
            iterations = 3
        if iterations < 1:
            iterations = 1

        # Early stopping behavior
        try:
            raw_stop = input("Stop early if judge becomes satisfied? (Y/n): ").strip().lower()
        except Exception:
            raw_stop = ""
        stop_when_satisfied = False if raw_stop in {"n", "no", "0", "false"} else True

        run_log = RUN_LOG_PATH
        print(f"\n[run] Full log will be written to: {run_log}")
        final_judge, accepted = judge_iterative(
            refined=refined,
            user_question=q,
            iterations=iterations,
            stop_when_satisfied=stop_when_satisfied,
            verbose=True,
            log_path=run_log,
        )

        try:
            js = final_judge.get("judge_summary") or {}
            print(
                f"\n[run] Judge iterations requested={js.get('iterations_requested')} completed={js.get('iterations_completed')} satisfied={js.get('satisfied')}"
            )
        except Exception:
            pass

        print("\nJUDGE SUMMARY:")
        print(json.dumps(final_judge.get("judge_summary") or {}, ensure_ascii=False, indent=2))

        # Multi-hop final solution phase (do NOT use multihop_rag judging; only planning + retrieval + synthesis)
        missing = []
        try:
            mf = (final_judge.get("judge_summary") or {}).get("missing_factors_by_iteration") or []
            if mf:
                missing = mf[-1].get("missing_factors") or []
        except Exception:
            missing = []

        subproblems = multihop_plan_subproblems(
            refined_question=str((final_judge.get("judge_summary") or {}).get("final_refined_question") or q),
            missing_factors=missing,
            max_subproblems=3,
        )
        _run_log(f"[run] Planned {len(subproblems)} subproblems")
        sub_chunks = multihop_retrieve_subproblem_chunks(
            subproblems,
            exclude_paper_ids=set((final_judge.get("judge_summary") or {}).get("rejected_paper_ids") or []),
            exclude_chunk_ids=set((final_judge.get("judge_summary") or {}).get("rejected_chunk_ids") or []),
        )

        # Answer each subproblem using ONLY its retrieved evidence.
        sub_answers = answer_subproblems(
            subproblems=subproblems,
            subproblem_chunks=sub_chunks,
        )
        solution = synthesize_final_solution(
            user_question=q,
            judge_final_json=final_judge,
            accepted_chunks=accepted,
            subproblem_chunks=sub_chunks,
            subproblem_answers=sub_answers,
        )
        saved_solution = _save_json_temp(
            {
                "judge_summary": final_judge.get("judge_summary") or {},
                "accepted_chunks": accepted,
                "subproblems": subproblems,
                "subproblem_chunks": sub_chunks,
                "subproblem_answers": sub_answers,
                "solution": solution,
            },
            prefix="final_solution",
        )
        _run_log(f"[run] Saved final solution bundle to: {saved_solution}")
        print(f"\n[run] Saved final solution bundle to: {saved_solution}")
        print("\nFINAL ANSWER:")
        paras = solution.get("final_answer_paragraphs")
        import re
        if isinstance(paras, list) and paras:
            paras = [re.sub(r'\[.*?\]|\(.*?(chunk|claim).*?\)', '', str(p)) for p in paras]
            print("\n\n".join(str(p) for p in paras if str(p).strip()))
        else:
            ans = str(solution.get("final_answer") or "")
            ans = re.sub(r'\[.*?\]|\(.*?(chunk|claim).*?\)', '', ans)
            print(ans)