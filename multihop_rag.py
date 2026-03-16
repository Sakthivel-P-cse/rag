"""multihop_rag.py

LLM-driven Multi-hop RAG + Citation Navigator (engineered scaffold)
-----------------------------------------------------------------

This file implements the architecture you described, using your
Modules 1–10 as *prompts* executed by a pluggable LLM client.

Key idea: heuristics are used ONLY for orchestration/prioritization;
scientific reasoning itself is delegated to the LLM. The LLM performs:
- question refinement
- sub-problem planning (tree)
- retrieval query generation
- chunk judging and selection
- evidence extraction
- per-subproblem solution proposal
- validation and confidence estimation
- citation navigation decisions
- sub-problem expansion (depth-controlled)
- final hidden-solution synthesis

Retrieval itself is still a system operation (Qdrant dual-vector search)
using your two embedding models.

Configuration
  - Provide LLM API later via env vars (OpenAI-compatible endpoint):
      LLM_BASE_URL   (default: https://api.openai.com/v1)
      LLM_API_KEY
      LLM_MODEL      (default: gpt-4o-mini)

  - Qdrant/Postgres via existing env vars used across this repo:
      PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
      QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION
"""

from __future__ import annotations

# Load .env file if present (python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # rely on actual env vars if dotenv not installed

import argparse
import json
import os
import re
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

import numpy as np

from Extractor.database import DatabaseManager
from Extractor.citation_graph import CitationGraphManager


_CHAT_SEMAPHORE: Optional[threading.BoundedSemaphore] = None


def _set_chat_semaphore(max_inflight_requests: int) -> None:
	global _CHAT_SEMAPHORE
	try:
		m = int(max_inflight_requests)
	except Exception:
		m = 0
	if m and m > 0:
		_CHAT_SEMAPHORE = threading.BoundedSemaphore(m)
	else:
		_CHAT_SEMAPHORE = None

try:
	from sentence_transformers import SentenceTransformer

	_SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
	SentenceTransformer = None
	_SENTENCE_TRANSFORMERS_AVAILABLE = False


_MECH_DEDUP_MODEL = None
_MECH_DEDUP_LOCK = threading.Lock()


def _semantic_dedup_claims_by_mechanism(
	claims: Sequence["EvidenceClaim"], *, threshold: float = 0.90, model_name: str = "all-MiniLM-L6-v2"
) -> List["EvidenceClaim"]:
	"""Semantic dedup for MECHANISM support: merges near-duplicate mechanism strings.

	This is intentionally narrow (mechanism-field only) to reduce risk and cost.
	If sentence-transformers isn't available, it falls back to no-op.
	"""
	cs = list(claims or [])
	if not _SENTENCE_TRANSFORMERS_AVAILABLE:
		return cs
	# Only embed non-empty mechanism strings; keep non-mechanism claims untouched.
	mech_texts: List[str] = []
	idx_map: List[int] = []
	for idx, c in enumerate(cs):
		m = str(getattr(c, "mechanism", "") or "").strip()
		if m:
			mech_texts.append(m)
			idx_map.append(idx)
	if len(mech_texts) < 2:
		return cs

	with _MECH_DEDUP_LOCK:
		global _MECH_DEDUP_MODEL
		if _MECH_DEDUP_MODEL is None:
			_MECH_DEDUP_MODEL = SentenceTransformer(model_name)
		model = _MECH_DEDUP_MODEL

	# Normalize embeddings so cosine similarity is dot product.
	vecs = model.encode(mech_texts, normalize_embeddings=True)
	if vecs is None:
		return cs

	# Greedy keep: highest-confidence mechanism claims first.
	order = sorted(
		range(len(mech_texts)),
		key=lambda i: float(getattr(cs[idx_map[i]], "confidence", 0.0) or 0.0),
		reverse=True,
	)
	kept_mech_indices: List[int] = []
	for i in order:
		v = vecs[i]
		is_dup = False
		for j in kept_mech_indices:
			try:
				sim = float(np.dot(v, vecs[j]))
			except Exception:
				sim = 0.0
			if sim >= float(threshold):
				is_dup = True
				break
		if not is_dup:
			kept_mech_indices.append(i)

	keep_claim_idxs: set[int] = set(idx_map[i] for i in kept_mech_indices)
	for idx, c in enumerate(cs):
		if not str(getattr(c, "mechanism", "") or "").strip():
			keep_claim_idxs.add(idx)
	return [c for idx, c in enumerate(cs) if idx in keep_claim_idxs]


# ---------------------------
# Simple configuration (edit here)
# ---------------------------
# NOTE: Hard-coding secrets is not recommended for production. This section exists
# only because you requested a single-file, no-env setup.

# If you're using OpenRouter, use:
#   base URL: https://openrouter.ai/api/v1
#   model:    meta-llama/Llama-3.3-70B-Instruct
LLM_BASE_URL_DEFAULT = "https://openrouter.ai/api/v1"
# Do NOT hard-code secrets. Set one of these env vars:
#   - OPENROUTER_API_KEY (recommended)
#   - LLM_API_KEY
LLM_API_KEY_DEFAULT = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY") or ""
LLM_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct"
LLM_MODEL_FAST_DEFAULT = "meta-llama/llama-3.1-8b-instruct"
LLM_TIMEOUT_DEFAULT_S = 60

QDRANT_CONFIG_DEFAULT: Dict[str, Any] = {
	"host": "localhost",
	"port": 6333,
	"collection": "research_papers",
}


# ---------------------------
# Prompts (Modules 1–10)
# ---------------------------

GLOBAL_RULES_SYSTEM = (
	"You are a scientific reasoning agent.\n"
	"You must NOT invent empirical facts, numeric results, datasets, or citations.\n"
	"You MAY abstract and synthesize mechanisms when supported by evidence.\n"
	"For MECHANISM questions, explanatory abstraction is allowed if supported by multiple claims.\n"
	"Separate clearly:\n"
	"- Evidence-backed statements\n"
	"- Inferred mechanisms (hypotheses)\n"
	"If evidence is weak or incomplete, state the uncertainty.\n"
	"Prefer explanatory insight over rigid completeness when discovering mechanisms.\n"
)


# Lenient rules: still JSON-only for tool stability, but far less restrictive.
GLOBAL_RULES_SYSTEM_LENIENT = (
	"You are a scientific assistant.\n"
	"You must output ONLY the JSON object required by the current task and nothing else.\n"
	"You may use general prior knowledge.\n"
	"Do NOT invent citations or claim you read papers you have not been given.\n"
	"When using retrieved evidence claims, citing claim IDs is helpful but not mandatory.\n"
)


# ---------------------------
# Optional: Balanced answering (evidence + model knowledge)
# ---------------------------
# NOTE:
# - This is only used when --use-hypothesis is enabled.
# - Retrieval and claim extraction remain evidence-only.
# - Any model-knowledge content must be explicitly labeled as such.

GLOBAL_RULES_SYSTEM_BALANCED = (
	"You are a scientific assistant.\n"
	"You must output ONLY the JSON object required by the current task and nothing else.\n"
	"You must NOT invent citations or claim you read papers you have not been given.\n"
	"You may use general prior knowledge to provide additional practical guidance, BUT you must clearly label it as MODEL_KNOWLEDGE.\n"
	"When using retrieved evidence claims, cite claim IDs explicitly.\n"
	"Do not mix evidence-backed statements and model-knowledge statements in the same bullet/sentence.\n"
	"If unsure, place content under Unknowns.\n"
)


PROMPT_REFINER = (
	"You are a research question refinement agent.\n"
	"Follow the SYSTEM rules provided. Do NOT answer the question.\n\n"
	"Task:\n"
	"Rewrite the user problem as a precise, unambiguous research question suitable for literature-based reasoning.\n\n"
	"Rules:\n"
	"- Do NOT answer the question\n"
	"- The refined question MUST be a single sentence (≤ 40 words)\n"
	"- Make hidden assumptions explicit (max 6)\n"
	"- Clarify scope and constraints\n"
	"- Remove informal language\n\n"
	"User problem:\n"
	"{{USER_PROBLEM}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"refined_question\": \"...\",\n"
	"  \"explicit_assumptions\": [\"...\"],\n"
	"  \"scope\": \"theoretical | experimental | review\"\n"
	"}\n\n"
	"If refinement is impossible, return:\n"
	"{\n"
	"  \"refined_question\": \"INSUFFICIENT\",\n"
	"  \"explicit_assumptions\": [],\n"
	"  \"scope\": \"review\",\n"
	"  \"inspection\": \"why refinement failed\"\n"
	"}"
)


PROMPT_PLANNER = (
	"You are a scientific planning agent.\n"
	"Follow all GLOBAL RULES.\n\n"
	"Task:\n"
	"Decompose the refined research question into the minimal set of ATOMIC sub-problems.\n\n"
	"Definition:\n"
	"- Atomic = one mechanism AND one outcome\n\n"
	"Constraints:\n"
	"- Produce at most {{MAX_SUBPROBLEMS}} subproblems\n"
	"- Each question ≤ 30 words\n"
	"- Avoid overlap\n"
	"- Explicitly state dependencies\n\n"
	"Refined question:\n"
	"{{REFINED_QUESTION}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"subproblems\": [\n"
	"    {\n"
	"      \"id\": \"SP1\",\n"
	"      \"question\": \"...\",\n"
	"      \"depends_on\": []\n"
	"    }\n"
	"  ]\n"
	"}\n\n"
	"If decomposition is not possible, return:\n"
	"{ \"subproblems\": [] }"
)


PROMPT_RETRIEVAL_QUERY = (
	"You are an information retrieval specialist.\n"
	"Follow the SYSTEM rules provided. Do NOT use prior knowledge.\n\n"
	"Task:\n"
	"Convert the sub-problem into a dense-retrieval search query optimized for scientific papers.\n\n"
	"Rules:\n"
	"- Focus on mechanisms, not conclusions\n"
	"- Include technical synonyms where relevant\n"
	"- Avoid vague or speculative language\n"
	"- Keep query concise and factual\n\n"
	"Sub-problem:\n"
	"{{SUBPROBLEM_QUESTION}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"search_query\": \"...\"\n"
	"}\n\n"
	"If no valid query can be formed, return:\n"
	"{ \"search_query\": \"INSUFFICIENT\" }"
)


PROMPT_CHUNK_JUDGE = (
	"You are a scientific evidence evaluator.\n"
	"Follow the SYSTEM rules provided. Do NOT use prior knowledge.\n\n"
	"Task:\n"
	"Judge whether the chunk provides evidence useful for solving the sub-problem.\n\n"
	"Classification rubric:\n"
	"- DIRECT_EVIDENCE: explicit facts directly answering the sub-problem (score ≥ 0.7)\n"
	"- INDIRECT_EVIDENCE: supports mechanism but not outcome (0.4–0.69)\n"
	"- BACKGROUND_ONLY: contextual information only (0.2–0.39)\n"
	"- IRRELEVANT: no useful information (< 0.2)\n\n"
	"Sub-problem:\n"
	"{{SUBPROBLEM_QUESTION}}\n\n"
	"Chunk:\n"
	"{{CHUNK_TEXT}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"classification\": \"DIRECT_EVIDENCE | INDIRECT_EVIDENCE | BACKGROUND_ONLY | IRRELEVANT\",\n"
	"  \"relevance_score\": 0.0,\n"
	"  \"justification\": \"≤ 30 words\"\n"
	"}"
)


PROMPT_EVIDENCE_EXTRACTOR = (
	"You are an evidence extraction agent.\n"
	"Follow the SYSTEM rules provided. Do NOT use prior knowledge.\n\n"
	"Task:\n"
	"Extract factual claims relevant to the sub-problem.\n\n"
	"Question type: {{QUESTION_TYPE}}\n\n"
	"Rules:\n"
	"- Do NOT invent facts beyond what the text supports\n"
	"- For MECHANISM questions ONLY: you MAY extract implicit mechanistic claims that are clearly implied by the text,\n"
	"  even if not stated as a single sentence. These must be grounded in the chunk.\n"
	"- Separate claim, mechanism, and conditions\n"
	"- Extract up to 5 claims for FACT questions; up to 10 claims for MECHANISM questions\n"
	"- If speculative, mark confidence low\n\n"
	"Sub-problem:\n"
	"{{SUBPROBLEM_QUESTION}}\n\n"
	"Text:\n"
	"{{CHUNK_TEXT}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"claims\": [\n"
	"    {\n"
	"      \"claim\": \"...\",\n"
	"      \"mechanism\": \"...\",\n"
	"      \"conditions\": \"...\",\n"
	"      \"confidence\": 0.0,\n"
	"      \"text_span\": \"verbatim quote ≤ 40 words\"\n"
	"    }\n"
	"  ]\n"
	"}\n\n"
	"If no factual claims exist, return:\n"
	"{ \"claims\": [] }"
)


PROMPT_SOLUTION_PROPOSER = (
	"You are a scientific reasoning agent.\n"
	"Follow all GLOBAL RULES.\n\n"
	"Task:\n"
	"Answer the sub-problem using ONLY the provided evidence claims.\n\n"
	"Rules:\n"
	"- Do NOT introduce new empirical measurements or citations\n"
	"- Evidence-backed sections must cite claims\n"
	"- Mechanistic abstraction is allowed in INFERRED_MECHANISMS\n"
	"- Controlled inference is allowed ONLY in an explicitly labeled section named 'INFERRED_MECHANISMS'.\n"
	"  * Each inferred bullet MUST start with 'INFERRED_MECHANISM:'\n"
	"  * Each inferred bullet SHOULD cite >=1 claim ID locally (best-effort)\n"
	"  * Inferred bullets must be phrased as hypotheses (e.g., 'may', 'could', 'suggests') and must not introduce numeric values, datasets, or named entities not present in claims\n"
	"  * If you cannot support an inference with at least one claim, put it in Unknowns instead\n"
	"- Cite claim IDs explicitly (inline, after the bullet/paragraph that uses them)\n"
	"- Write a DETAILED, structured answer. Use this structure inside proposed_solution:\n"
	"  1) Summary (2-3 sentences)\n"
	"  2) Evidence-backed factors/causes (3-6 bullets)\n"
	"  3) Evidence-backed mitigations/actions (3-6 bullets; ONLY if explicitly supported by claims; otherwise put them in Unknowns)\n"
	"  4) INFERRED_MECHANISMS (optional; bullets as described above)\n"
	"  5) Unknowns / missing evidence (bullets)\n"
	"- Sections (1)-(3) must be supported by used claims, but do NOT force per-sentence citations; cite at paragraph/bullet level.\n\n"
	"If a 'Draft answer (NOT evidence)' block is provided, you MAY also include a separate section at the end:\n"
	"  6) MODEL_KNOWLEDGE (practical guidance)\n"
	"Rules for MODEL_KNOWLEDGE section:\n"
	"- Label the section header exactly 'MODEL_KNOWLEDGE'\n"
	"- Do NOT cite claim IDs in this section\n"
	"- Do NOT present it as proven; keep it actionable and clearly separated\n\n"
	"If evidence is insufficient, respond EXACTLY with:\n"
	"{ \"proposed_solution\": \"INSUFFICIENT\", \"used_claims\": [], \"inspection_reasons\": [\"...\"] }\n\n"
	"Sub-problem:\n"
	"{{SUBPROBLEM_QUESTION}}\n\n"
	"Evidence:\n"
	"{{STRUCTURED_CLAIMS}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"proposed_solution\": \"...\",\n"
	"  \"used_claims\": [\"SP1:chunk_x:claim_1\"]\n"
	"}"
)


PROMPT_SOLUTION_PROPOSER_LENIENT = (
	"You are a scientific reasoning agent (LENIENT).\n\n"
	"Task:\n"
	"Answer the sub-problem using the provided evidence claims as guidance, but do not be blocked by missing citations.\n\n"
	"Rules (lenient):\n"
	"- You MAY synthesize and explain mechanisms; do not require verbatim support for every sentence.\n"
	"- Do NOT invent specific citations, named events, or precise numeric results not present in the evidence.\n"
	"- If the evidence is thin, provide a best-effort explanation and list Unknowns rather than returning INSUFFICIENT.\n"
	"- Include claim IDs when convenient, but do not force per-sentence citations.\n\n"
	"Sub-problem:\n"
	"{{SUBPROBLEM_QUESTION}}\n\n"
	"Evidence:\n"
	"{{STRUCTURED_CLAIMS}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"proposed_solution\": \"...\",\n"
	"  \"used_claims\": [\"SP1:chunk_x:claim_1\"]\n"
	"}"
)


PROMPT_MECHANISM_INFERENCE = (
	"You are a scientific mechanism inference agent.\n"
	"Follow all GLOBAL RULES.\n\n"
	"Task:\n"
	"From the provided evidence claims, infer plausible high-level mechanisms that could explain the sub-problem.\n\n"
	"Critical constraints (controlled inference):\n"
	"- You MUST NOT introduce new empirical facts or citations.\n"
	"- You MAY abstract causal structure from the evidence claims.\n"
	"- Each inferred mechanism SHOULD cite >=1 supporting claim ID (best-effort).\n"
	"- Do NOT introduce specific numbers, datasets, locations, or named entities not present in the claims.\n"
	"- Keep each mechanism statement <= 25 words.\n"
	"- If evidence is insufficient, return an empty list.\n\n"
	"Sub-problem:\n"
	"{{SUBPROBLEM_QUESTION}}\n\n"
	"Evidence claims:\n"
	"{{STRUCTURED_CLAIMS}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"inferred_mechanisms\": [\n"
	"    {\n"
	"      \"label\": \"INFERRED_MECHANISM\",\n"
	"      \"mechanism\": \"...\",\n"
	"      \"supporting_claim_ids\": [\"SP1:chunk_x:claim_1\", \"SP1:chunk_y:claim_2\"],\n"
	"      \"limits\": \"...\",\n"
	"      \"confidence\": 0.0\n"
	"    }\n"
	"  ]\n"
	"}"
)


PROMPT_MECHANISM_AGGREGATION = (
	"You are a scientific mechanism aggregation agent.\n"
	"Follow all GLOBAL RULES.\n\n"
	"Task:\n"
	"Cluster and merge per-claim mechanisms into a small set of higher-level mechanism patterns that can explain the sub-problem.\n\n"
	"Why this exists:\n"
	"- Individual claims may describe different facets (e.g., thresholds, bistability, noise-driven transitions) of one dynamical explanation.\n"
	"- Aggregation should promote repeated patterns into a coherent mechanism explanation.\n\n"
	"Controlled aggregation rules (no free inference):\n"
	"- Use ONLY the provided evidence claims and their mechanism fields.\n"
	"- Each cluster must cite its supporting claim IDs.\n"
	"- Set cluster_strength to one of: emerging | convergent | multi_paper.\n"
	"  * emerging: supported by <2 claims OR support is too weak/ambiguous\n"
	"  * convergent: supported by >=2 claims, but all evidence appears to come from a single paper_id\n"
	"  * multi_paper: supported by >=2 distinct paper_id values\n"
	"- Do NOT introduce numbers, datasets, or named entities not present in the claims.\n"
	"- Prefer 2–5 clusters total; leave unrelated items unclustered.\n\n"
	"Sub-problem:\n"
	"{{SUBPROBLEM_QUESTION}}\n\n"
	"Evidence claims (JSON):\n"
	"{{STRUCTURED_CLAIMS}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"clusters\": [\n"
	"    {\n"
	"      \"cluster_id\": \"M1\",\n"
	"      \"canonical_mechanism\": \"...\",\n"
	"      \"member_claim_ids\": [\"SP1:chunk_x:claim_1\"],\n"
	"      \"supporting_paper_ids\": [\"paper_a\", \"paper_b\"],\n"
	"      \"cluster_strength\": \"multi_paper\",\n"
	"      \"notes\": \"...\"\n"
	"    }\n"
	"  ],\n"
	"  \"unclustered_claim_ids\": [\"...\"]\n"
	"}"
)


PROMPT_GLOBAL_MECHANISM_AGGREGATION = (
	"You are a scientific mechanism aggregation agent.\n"
	"Follow all GLOBAL RULES.\n\n"
	"Task:\n"
	"Cluster and merge mechanisms ACROSS accepted sub-problems into a small set of higher-level mechanism patterns that explain the overall question.\n\n"
	"Controlled aggregation rules (no free inference):\n"
	"- Use ONLY the provided evidence claims and their mechanism fields.\n"
	"- Each cluster must cite its supporting claim IDs.\n"
	"- Set cluster_strength to one of: emerging | convergent | multi_paper.\n"
	"  * emerging: supported by <2 claims OR support is too weak/ambiguous\n"
	"  * convergent: supported by >=2 claims, but all evidence appears to come from a single paper_id\n"
	"  * multi_paper: supported by >=2 distinct paper_id values\n"
	"- Prefer clusters that are supported across distinct subproblem IDs (the prefix before ':' in claim_id).\n"
	"- Do NOT introduce numbers, datasets, or named entities not present in the claims.\n"
	"- Prefer 2–6 clusters total; leave unrelated items unclustered.\n\n"
	"Overall question:\n"
	"{{REFINED_QUESTION}}\n\n"
	"Evidence claims (JSON, from multiple accepted subproblems):\n"
	"{{STRUCTURED_CLAIMS}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"clusters\": [\n"
	"    {\n"
	"      \"cluster_id\": \"GM1\",\n"
	"      \"canonical_mechanism\": \"...\",\n"
	"      \"member_claim_ids\": [\"SP1:chunk_x:claim_1\"],\n"
	"      \"supporting_paper_ids\": [\"paper_a\", \"paper_b\"],\n"
	"      \"cluster_strength\": \"multi_paper\",\n"
	"      \"notes\": \"...\"\n"
	"    }\n"
	"  ],\n"
	"  \"unclustered_claim_ids\": [\"...\"]\n"
	"}"
)


PROMPT_VALIDATOR = (
	"You are a scientific validation agent.\n"
	"Follow all GLOBAL RULES.\n\n"
	"Task:\n"
	"Validate whether the proposed solution is fully supported by the evidence.\n\n"
	"Rules:\n"
	"- ACCEPT if the solution is supported by the used claims (it may synthesize/abstract across multiple claims)\n"
	"- ACCEPT INSUFFICIENT only if no claims answer the sub-problem\n"
	"- FAIL if the solution introduces new EMPIRICAL FACTS that are not present in claims\n"
	"  (empirical facts include: specific numbers, datasets, locations, named events, new measured correlations/causal directions stated as facts).\n"
	"- For MECHANISM questions: abstract causal explanations (e.g., thresholds, feedbacks, bistability, noise-triggered transitions) are NOT empirical facts by themselves.\n"
	"  However, new measurements, datasets, locations, named experimental/natural events, or specific reported results ARE empirical facts and must be present in claims.\n"
	"- Outside an 'INFERRED_MECHANISMS' section: allow paraphrase, summarization, and combining claims, BUT every factual paragraph must be supported by at least one used claim ID (do NOT require per-sentence citations).\n"
	"- FAIL if new details are introduced OUTSIDE an explicitly labeled 'INFERRED_MECHANISMS' section AND those details are not clearly derivable as a paraphrase/synthesis of used claims.\n"
	"- For content inside 'INFERRED_MECHANISMS':\n"
	"  * Allow controlled inference ONLY if each inferred bullet starts with 'INFERRED_MECHANISM:'\n"
	"  * Each inferred bullet should cite >=1 supporting claim ID (best-effort).\n"
	"  * Inferred bullets must be phrased as hypotheses (may/could/suggests), not as proven facts.\n"
	"  * Inferred bullets must NOT add new empirical facts; they may propose causal structure/mechanistic interpretation only.\n"
	"  * If ALL inferred bullets have 0 supporting claim IDs, mark PARTIAL and explain what evidence is missing\n"
	"- Ignore formatting, section headers, and inline claim-ID citations when judging 'new details'\n"
	"- If the solution contains a section labeled 'MODEL_KNOWLEDGE', ignore that section entirely (do not penalize or reward it).\n"
	"- Compute confidence conservatively:\n"
	"  base = min(confidence of used claims)\n"
	"  subtract 0.1 for each indirect or conditional gap\n\n"
	"Target confidence: {{CONFIDENCE_TARGET}}\n\n"
	"Used claim IDs:\n"
	"{{USED_CLAIMS}}\n\n"
	"Proposed solution:\n"
	"{{PROPOSED_SOLUTION}}\n\n"
	"Evidence:\n"
	"{{STRUCTURED_CLAIMS}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"status\": \"ACCEPTED | PARTIAL | FAILED\",\n"
	"  \"confidence\": 0.0,\n"
	"  \"calc_trace\": [\"...\"] ,\n"
	"  \"failure_reasons\": [\"...\"]\n"
	"}"
)


PROMPT_VALIDATOR_LENIENT = (
	"You are a scientific validation agent (LENIENT).\n\n"
	"Task:\n"
	"Decide whether the proposed solution is broadly consistent with the evidence.\n\n"
	"Rules (lenient):\n"
	"- Prefer ACCEPTED unless there is a clear contradiction with evidence.\n"
	"- Do NOT require per-sentence claim-ID citations.\n"
	"- Allow plausible synthesis and mechanistic explanation.\n"
	"- Only FAIL for: (a) direct contradiction, or (b) obviously invented concrete empirical details (numbers/datasets/named events) not present.\n"
	"- If uncertain, return PARTIAL (not FAILED).\n\n"
	"Target confidence: {{CONFIDENCE_TARGET}}\n\n"
	"Used claim IDs:\n"
	"{{USED_CLAIMS}}\n\n"
	"Proposed solution:\n"
	"{{PROPOSED_SOLUTION}}\n\n"
	"Evidence:\n"
	"{{STRUCTURED_CLAIMS}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"status\": \"ACCEPTED | PARTIAL | FAILED\",\n"
	"  \"confidence\": 0.0,\n"
	"  \"calc_trace\": [\"...\"],\n"
	"  \"failure_reasons\": [\"...\"]\n"
	"}"
)


PROMPT_CITATION_NAV = (
	"You are a citation reasoning agent.\n"
	"Follow all GLOBAL RULES.\n\n"
	"Task:\n"
	"Decide whether additional cited papers are needed.\n\n"
	"Rules:\n"
	"- Suggest at most 3 citations\n"
	"- Use numeric citation identifiers only\n\n"
	"Evidence gaps:\n"
	"{{FAILURE_REASONS}}\n\n"
	"Known citations:\n"
	"{{CITATION_LIST}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"need_more_papers\": true,\n"
	"  \"target_citations\": [\"13\", \"21\"]\n"
	"}"
)


PROMPT_EXPAND = (
	"You are a recursive planning agent.\n"
	"Follow all GLOBAL RULES.\n\n"
	"Task:\n"
	"Generate additional ATOMIC sub-problems needed to resolve the failure.\n\n"
	"Original sub-problem:\n"
	"{{SUBPROBLEM_QUESTION}}\n\n"
	"Failure reasons:\n"
	"{{FAILURE_REASONS}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"new_subproblems\": [\n"
	"    {\n"
	"      \"id\": \"SP2.1\",\n"
	"      \"question\": \"...\",\n"
	"      \"depends_on\": []\n"
	"    }\n"
	"  ]\n"
	"}\n\n"
	"If no expansion is possible, return:\n"
	"{ \"new_subproblems\": [] }"
)


PROMPT_SYNTHESIZE = (
	"You are a scientific synthesis agent.\n"
	"Follow all GLOBAL RULES.\n\n"
	"Task:\n"
	"Synthesize an explanatory final answer using accepted sub-problem solutions, and optionally add a clearly separated MODEL_KNOWLEDGE section.\n\n"
	"Rules:\n"
	"- The main answer MUST be grounded in accepted sub-problem solutions\n"
	"- You MAY use prior knowledge only in a final section labeled MODEL_KNOWLEDGE\n"
	"- If accepted solutions are empty, return INSUFFICIENT_EVIDENCE\n"
	"- Propagate confidence conservatively\n"
	"- Reference ONLY provided identifiers\n"
	"- Write a DETAILED, structured final answer in hidden_solution with sections:\n"
	"  Summary; Evidence-backed factors; Evidence-backed mitigations/actions; INFERRED_MECHANISMS (optional); Unknowns\n"
	"- Every factual statement in Summary/Factors/Mitigations MUST cite at least one identifier from accepted solutions\n"
	"  (either a subproblem id like 'SP2' or a claim id like 'SP2:chunk:claim_1').\n\n"
	"Rules for INFERRED_MECHANISMS section (controlled synthesis):\n"
	"- Each bullet MUST start with 'INFERRED_MECHANISM:'\n"
	"- Each bullet MUST cite >=2 identifiers from accepted solutions (prefer claim IDs; otherwise multiple subproblem IDs)\n"
	"- Do NOT introduce new empirical facts (numbers, datasets, named events) not present in accepted solutions\n"
	"- Phrase as hypothesis (may/could/suggests)\n\n"
	"Optional MODEL_KNOWLEDGE section rules:\n"
	"- Put it after Unknowns\n"
	"- Do NOT cite identifiers there\n"
	"- Keep it short and practical\n\n"
	"Main question:\n"
	"{{REFINED_QUESTION}}\n\n"
	"Accepted sub-solutions:\n"
	"{{SUB_SOLUTIONS}}\n\n"
	"Global mechanism clusters (JSON, across accepted subproblems; may be null):\n"
	"{{GLOBAL_MECHANISM_CLUSTERS}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"hidden_solution\": \"... | INSUFFICIENT_EVIDENCE\",\n"
	"  \"why_hidden\": \"...\",\n"
	"  \"evidence_chain\": [\"SP1 used_claims=[...] from chunks=[...]\"] ,\n"
	"  \"final_confidence\": 0.0\n"
	"}"
)


PROMPT_SYNTHESIZE_LENIENT = (
	"You are a scientific synthesis agent (LENIENT).\n\n"
	"Task:\n"
	"Synthesize a coherent final answer using accepted sub-problem solutions as anchors.\n\n"
	"Rules (lenient):\n"
	"- Prefer producing a helpful answer over refusing.\n"
	"- Do NOT invent citations, named events, or precise numeric results not present in the provided sub-solutions.\n"
	"- You may add general background explanation without labeling every sentence, but keep it reasonable and non-specific.\n"
	"- If evidence is thin, include an Unknowns section listing what would need more papers.\n\n"
	"Main question:\n"
	"{{REFINED_QUESTION}}\n\n"
	"Accepted sub-solutions:\n"
	"{{SUB_SOLUTIONS}}\n\n"
	"Global mechanism clusters (JSON; may be null):\n"
	"{{GLOBAL_MECHANISM_CLUSTERS}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"hidden_solution\": \"...\",\n"
	"  \"why_hidden\": \"...\",\n"
	"  \"evidence_chain\": [\"...\"],\n"
	"  \"final_confidence\": 0.0\n"
	"}"
)


# ---------------------------
# Optional: Hypothesis helper (uses model prior knowledge)
# ---------------------------

GLOBAL_RULES_SYSTEM_HYPOTHESIS = (
	"You are a scientific assistant.\n"
	"You may use general prior knowledge to propose tentative hypotheses and search keywords.\n"
	"You must output ONLY the JSON object required by the current task and nothing else.\n"
	"You must clearly mark speculation as hypothesis.\n"
	"Do NOT invent citations or claim you read papers you have not been given.\n"
)


PROMPT_HYPOTHESIS = (
	"You are a hypothesis generator for scientific literature search.\n\n"
	"Task:\n"
	"Given a research sub-problem, propose a likely answer as a HYPOTHESIS (may be wrong) and list search terms.\n\n"
	"Rules:\n"
	"- This is NOT the final answer; it is only for improving retrieval and extraction\n"
	"- Keep hypothesis short and conditional\n"
	"- Provide concrete keywords, synonyms, and related terms for dense retrieval\n\n"
	"Sub-problem:\n"
	"{{SUBPROBLEM_QUESTION}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"hypothesis_answer\": \"...\",\n"
	"  \"keywords\": [\"...\"],\n"
	"  \"synonyms\": [\"...\"],\n"
	"  \"do_not_assume\": [\"...\"]\n"
	"}"
)


PROMPT_QUERY_REWRITE = (
	"You are an information retrieval specialist.\n"
	"Follow all GLOBAL RULES.\n\n"
	"Task:\n"
	"Rewrite the search query to address the evidence gaps and focus on missing information.\n\n"
	"Inputs:\n"
	"- Sub-problem\n"
	"- Previous search query\n"
	"- Extracted evidence snippets (claim texts)\n"
	"- Validation failure reasons\n\n"
	"Rules:\n"
	"- Use ONLY provided context\n"
	"- Keep query concise\n"
	"- Prefer concrete entities/methods/metrics/datasets mentioned\n\n"
	"Sub-problem:\n"
	"{{SUBPROBLEM_QUESTION}}\n\n"
	"Previous search query:\n"
	"{{PREV_QUERY}}\n\n"
	"Evidence snippets:\n"
	"{{EVIDENCE_SNIPPETS}}\n\n"
	"Failure reasons:\n"
	"{{FAILURE_REASONS}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"search_query\": \"...\"\n"
	"}"
)


PROMPT_GLOBAL_PAPER_SELECTION = (
	"You are selecting a GLOBAL shortlist of papers that will be used for ALL subproblems.\n"
	"Goal: maximize downstream answer quality and avoid empty retrieval later.\n\n"
	"Rules:\n"
	"- Output STRICT JSON only.\n"
	"- Choose papers most directly useful for the user question.\n"
	"- Prefer mechanism/model/framework discussion when relevant.\n"
	"- Select K papers unless the candidate list is extremely low-quality.\n\n"
	"User question:\n{{REFINED_QUESTION}}\n\n"
	"K = {{TOP_PAPERS}}\n\n"
	"Candidate papers (each has short snippet(s) from retrieved chunks):\n"
	"{{PAPER_LIST}}\n\n"
	"Return JSON with this schema:\n"
	"{\n"
	"  \"selected_paper_ids\": [\"paper_id_1\", \"paper_id_2\"],\n"
	"  \"notes\": \"brief rationale\"\n"
	"}\n"
)


PROMPT_SLOT_SCHEMA = (
	"You are a scientific planning agent.\n"
	"Follow the SYSTEM rules provided. Do NOT use prior knowledge.\n\n"
	"Task:\n"
	"Given a sub-problem, define the minimal set of evidence slots needed for a complete answer.\n\n"
	"Rules:\n"
	"- 3 to 7 slots\n"
	"- Slots must be short noun phrases\n"
	"- Slots must be answerable from paper evidence (not opinions)\n\n"
	"Sub-problem:\n"
	"{{SUBPROBLEM_QUESTION}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"required_slots\": [\"...\"]\n"
	"}"
)


PROMPT_SLOT_COVERAGE = (
	"You are a scientific evidence evaluator.\n"
	"Follow the SYSTEM rules provided. Do NOT use prior knowledge.\n\n"
	"Task:\n"
	"Given required slots and extracted evidence claims, determine which slots are supported.\n\n"
	"Rules:\n"
	"- Use ONLY the claim text provided\n"
	"- A slot is supported only if a claim directly contains the needed info\n\n"
	"Required slots:\n"
	"{{REQUIRED_SLOTS}}\n\n"
	"Evidence claims (JSON):\n"
	"{{STRUCTURED_CLAIMS}}\n\n"
	"Output (STRICT JSON ONLY):\n"
	"{\n"
	"  \"supported_slots\": [\"...\"],\n"
	"  \"missing_slots\": [\"...\"],\n"
	"  \"notes\": [\"...\"]\n"
	"}"
)


# ---------------------------
# Data models
# ---------------------------


@dataclass(frozen=True)
class RefinedProblem:
	original: str
	refined_question: str
	explicit_assumptions: List[str]
	scope: str


@dataclass(frozen=True)
class PlannedSubProblem:
	id: str
	question: str
	depends_on: List[str]
	depth: int = 0


@dataclass(frozen=True)
class ChunkCandidate:
	chunk_id: str
	score_small: float
	score_big: float
	combined_score: float


@dataclass(frozen=True)
class JudgedChunk:
	chunk_id: str
	classification: str
	relevance_score: float
	justification: str
	combined_score: float


@dataclass(frozen=True)
class EvidenceClaim:
	claim_id: str
	claim: str
	mechanism: str
	conditions: str
	confidence: float
	source_chunk_id: str


def _normalize_claim_text(text: str) -> str:
	val = re.sub(r"\s+", " ", str(text or "").strip().lower())
	val = re.sub(r"[^a-z0-9\s\-_/\.:%]", "", val)
	return val


class GlobalClaimStore:
	"""Global claim de-dup store across subproblems.

	Keeps the first seen claim as canonical; later duplicates are mapped.
	"""

	def __init__(self):
		self.by_norm: Dict[str, EvidenceClaim] = {}
		self.alias_to_canonical: Dict[str, str] = {}

	def ingest(self, claims: Sequence[EvidenceClaim]) -> List[EvidenceClaim]:
		out: List[EvidenceClaim] = []
		for c in claims:
			# Include mechanism + conditions so dedup doesn't collapse distinct contexts.
			combo = f"{c.claim} {c.mechanism} {c.conditions}".strip()
			n = _normalize_claim_text(combo)
			if not n:
				continue
			if n in self.by_norm:
				canon = self.by_norm[n]
				self.alias_to_canonical[c.claim_id] = canon.claim_id
				continue
			self.by_norm[n] = c
			self.alias_to_canonical[c.claim_id] = c.claim_id
			out.append(c)
		return out

	def canonicalize_ids(self, ids: Sequence[str]) -> List[str]:
		out: List[str] = []
		for i in ids or []:
			key = str(i).strip()
			if not key:
				continue
			out.append(self.alias_to_canonical.get(key, key))
		return list(dict.fromkeys(out))


@dataclass(frozen=True)
class SubProblemSolution:
	subproblem_id: str
	subproblem_question: str
	proposed_solution: str
	used_claims: List[str]
	status: str
	confidence: float
	failure_reasons: List[str]
	selected_chunk_ids: List[str]
	citations_followed: List[str]


class ContextGraphStore:
	"""Append-only JSONL store for solved/failed/partial attempts."""

	def __init__(self, out_dir: Path, *, shard_key: Optional[str] = None):
		self.out_dir = out_dir
		self.out_dir.mkdir(parents=True, exist_ok=True)
		# Shard by date (UTC) to avoid unbounded growth of a single file.
		key = str(shard_key or "").strip() or time.strftime("%Y%m%d", time.gmtime())
		self.path = self.out_dir / f"context_graph_{key}.jsonl"
		self.failed_path = self.out_dir / f"failed_container_{key}.jsonl"

	def append(self, record: Dict[str, Any], status: str):
		record = dict(record)
		record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
		line = json.dumps(record, ensure_ascii=False)
		self.path.open("a", encoding="utf-8").write(line + "\n")
		if status == "failed":
			self.failed_path.open("a", encoding="utf-8").write(line + "\n")


# ---------------------------
# LLM client (OpenAI-compatible)
# ---------------------------


class LLMClient:
	def __init__(self, base_url: str, api_key: str, model: str, timeout_s: int = 60):
		self.base_url = (base_url or "").rstrip("/")
		self.api_key = api_key or ""
		self.model = model or ""
		self.timeout_s = int(timeout_s)
		self._client = httpx.Client(
			timeout=httpx.Timeout(timeout=self.timeout_s, connect=10.0, read=float(self.timeout_s), write=10.0, pool=10.0),
			limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
			follow_redirects=True,
		)
		self._cache: Dict[Tuple[str, str, float, int, str], Dict[str, Any]] = {}
		self._cache_lock = threading.Lock()

		if not self.base_url:
			self.base_url = "https://api.openai.com/v1"
		if not self.model:
			self.model = "gpt-4o-mini"
		if not self.api_key:
			raise ValueError("Missing LLM_API_KEY. Set env var LLM_API_KEY or pass --llm-api-key.")

	def chat_json(self, *, user_prompt: str, temperature: float = 0.0, max_tokens: int = 900) -> Dict[str, Any]:
		cache_key = ("GLOBAL", str(user_prompt), float(temperature), int(max_tokens), str(self.model))
		with self._cache_lock:
			cached = self._cache.get(cache_key)
			if cached is not None:
				return dict(cached)

		messages = [
			{"role": "system", "content": GLOBAL_RULES_SYSTEM},
			{"role": "user", "content": user_prompt},
		]
		content = self._chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
		repair_attempts = 0
		while True:
			try:
				out = _parse_json_strictish(content)
				with self._cache_lock:
					self._cache[cache_key] = dict(out)
				return out
			except Exception:
				if repair_attempts > 0:
					raise RuntimeError("JSON repair failed repeatedly")
				repair_attempts += 1
				# Some providers/models occasionally ignore "STRICT JSON" instructions.
				# Do an LLM-driven repair pass to coerce strict JSON output.
				repair_prompt = (
					"Your previous response was NOT valid JSON. "
					"You MUST output ONLY a single valid JSON object and nothing else.\n\n"
					"Re-read and follow the instructions and schema in this prompt:\n"
					f"{user_prompt}\n\n"
					"Here is your previous response (do not repeat it; convert it to the required JSON):\n"
					f"{content}\n"
				)
				repair_messages = [
					{"role": "system", "content": GLOBAL_RULES_SYSTEM},
					{"role": "user", "content": repair_prompt},
				]
				content = self._chat(messages=repair_messages, temperature=0.0, max_tokens=max_tokens)

	def chat_json_custom_system(
		self,
		*,
		system_prompt: str,
		user_prompt: str,
		temperature: float = 0.0,
		max_tokens: int = 500,
	) -> Dict[str, Any]:
		"""Run a one-off JSON task with a custom system prompt.

		Used for optional features like hypothesis generation where rules differ from GLOBAL_RULES_SYSTEM.
		"""
		messages = [
			{"role": "system", "content": str(system_prompt or "").strip() or GLOBAL_RULES_SYSTEM},
			{"role": "user", "content": user_prompt},
		]
		cache_key = (str(system_prompt or "").strip() or "GLOBAL", str(user_prompt), float(temperature), int(max_tokens), str(self.model))
		with self._cache_lock:
			cached = self._cache.get(cache_key)
			if cached is not None:
				return dict(cached)
		content = self._chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
		repair_attempts = 0
		while True:
			try:
				out = _parse_json_strictish(content)
				with self._cache_lock:
					self._cache[cache_key] = dict(out)
				return out
			except Exception:
				if repair_attempts > 0:
					raise RuntimeError("JSON repair failed repeatedly")
				repair_attempts += 1
				repair_prompt = (
					"Your previous response was NOT valid JSON. "
					"You MUST output ONLY a single valid JSON object and nothing else.\n\n"
					"Re-read and follow the instructions and schema in this prompt:\n"
					f"{user_prompt}\n\n"
					"Here is your previous response (do not repeat it; convert it to the required JSON):\n"
					f"{content}\n"
				)
				repair_messages = [
					{"role": "system", "content": str(system_prompt or "").strip() or GLOBAL_RULES_SYSTEM},
					{"role": "user", "content": repair_prompt},
				]
				content = self._chat(messages=repair_messages, temperature=0.0, max_tokens=max_tokens)

	def _chat(self, *, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
		sem = _CHAT_SEMAPHORE
		if sem is not None:
			sem.acquire()
		try:
			url = f"{self.base_url}/chat/completions"
			headers = {
				"Content-Type": "application/json",
				"Authorization": f"Bearer {self.api_key}",
				# Helpful for OpenRouter; safe to include elsewhere.
				"HTTP-Referer": "http://localhost",
				"X-Title": "multihop_rag",
			}

			last_error: Optional[str] = None
			for attempt in range(3):
				payload = {
					"model": self.model,
					"messages": messages,
					"temperature": float(temperature),
					"max_tokens": int(max_tokens),
				}
				try:
					# Use streaming + a hard overall deadline so we can't hang forever even if the server trickles bytes.
					deadline = time.monotonic() + float(self.timeout_s)
					with self._client.stream("POST", url, headers=headers, json=payload) as resp:
						chunks: List[bytes] = []
						for part in resp.iter_bytes():
							if time.monotonic() > deadline:
								raise httpx.TimeoutException(f"Overall request deadline exceeded ({self.timeout_s}s)")
							chunks.append(part)
							# Guardrail: don't allow unbounded bodies.
							if sum(len(c) for c in chunks) > 5_000_000:
								raise RuntimeError("LLM response too large")
						content_bytes = b"".join(chunks)
						status_code = resp.status_code
				except httpx.TimeoutException as e:
					last_error = f"LLM Timeout after {self.timeout_s}s: {e}"
					# small backoff then retry
					time.sleep(0.8 * (attempt + 1))
					continue
				except httpx.RequestError as e:
					last_error = f"LLM RequestError: {e}"
					time.sleep(0.8 * (attempt + 1))
					continue

				# Decode response once (we may need it for errors or JSON)
				text = content_bytes.decode("utf-8", errors="ignore")

				if status_code >= 400:
					err_body = text
					last_error = f"LLM HTTPError {status_code}: {err_body}"
					if status_code == 402:
						m = re.search(r"can only afford\s+(\d+)", err_body)
						if m:
							afford = int(m.group(1))
							if 0 < afford < int(max_tokens):
								max_tokens = afford
								continue
						raise RuntimeError(last_error)
					# Retry some transient statuses.
					if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
						time.sleep(0.8 * (attempt + 1))
						continue
					raise RuntimeError(last_error)

				try:
					data = json.loads(text)
				except Exception:
					last_error = f"LLM returned non-JSON response: {text[:500]}"
					time.sleep(0.8 * (attempt + 1))
					continue

				choices = data.get("choices") or []
				if not choices:
					raise RuntimeError(f"LLM response missing choices: {data}")
				msg = (choices[0].get("message") or {}).get("content")
				if not msg:
					raise RuntimeError(f"LLM response missing message content: {data}")
				return str(msg)

			raise RuntimeError(last_error or "LLM request failed")
		finally:
			if sem is not None:
				sem.release()


def _normalize_cluster_strength(cluster: Dict[str, Any]) -> Dict[str, Any]:
	"""Normalize mechanism cluster fields to a single `cluster_strength` signal."""
	cl = dict(cluster or {})
	strength = str(cl.get("cluster_strength") or "").strip().lower()
	if strength in {"emerging", "convergent", "multi_paper"}:
		cl["cluster_strength"] = strength
		return cl
	spids = [str(x) for x in (cl.get("supporting_paper_ids") or []) if str(x).strip()]
	member_ids = [str(x) for x in (cl.get("member_claim_ids") or []) if str(x).strip()]
	if len(set(spids)) >= 2:
		cl["cluster_strength"] = "multi_paper"
	elif len(member_ids) >= 2:
		cl["cluster_strength"] = "convergent"
	else:
		cl["cluster_strength"] = "emerging"
	return cl


def _cap_confidence_if_all_used_claims_low(
	*,
	base_conf: float,
	used_claim_ids: Sequence[str],
	claims: Sequence["EvidenceClaim"],
	low_threshold: float = 0.4,
	cap: float = 0.6,
) -> Tuple[float, Optional[str]]:
	used = {str(x) for x in (used_claim_ids or []) if str(x).strip()}
	if not used:
		return float(base_conf), None
	by_id: Dict[str, EvidenceClaim] = {c.claim_id: c for c in (claims or [])}
	used_confs: List[float] = []
	for cid in used:
		c = by_id.get(cid)
		if c is None:
			continue
		used_confs.append(float(getattr(c, "confidence", 0.0) or 0.0))
	if not used_confs:
		return float(base_conf), None
	max_used = max(used_confs)
	if max_used < float(low_threshold):
		new_conf = min(float(base_conf), float(cap))
		note = f"All used claims low-confidence (max={max_used:.2f} < {low_threshold:.2f}); capped confidence to {new_conf:.2f}."
		return new_conf, note
	return float(base_conf), None


def _parse_json_strictish(text: str) -> Dict[str, Any]:

    """Parse strict JSON, with a safe fallback to extracting the first JSON object."""
    text = (text or "").strip()
    decoder = json.JSONDecoder()

    # First try: parse full string.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Second try: parse the FIRST JSON object/array and ignore trailing text.
    # This handles cases where the model returns multiple JSON objects or appends commentary.
    for start_char in ("{", "["):
        idx = text.find(start_char)
        if idx == -1:
            continue
        try:
            obj, _end = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Fallback: grab first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"Expected JSON object, got: {text[:500]}")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return obj


def _strip_model_knowledge_section(text: str) -> str:
	"""Remove any MODEL_KNOWLEDGE section from a structured answer to keep evidence payload compact."""
	if not text:
		return ""
	# Split on a header line containing MODEL_KNOWLEDGE (case-insensitive).
	# Keep everything before it.
	m = re.search(r"(?im)^\s*MODEL_KNOWLEDGE\s*:?\s*$", text)
	if not m:
		return text
	return text[: m.start()].rstrip()


def _compact_accepted_for_synthesis(accepted: Dict[str, "SubProblemSolution"], *, max_items: int = 8, max_chars: int = 3000) -> str:
	"""Build a compact JSON string for Module 10 to stay under tight prompt-token limits."""
	items = sorted(accepted.values(), key=lambda s: float(s.confidence), reverse=True)[:max_items]
	compact: List[Dict[str, Any]] = []
	for s in items:
		sol_text = _strip_model_knowledge_section(str(s.proposed_solution or "")).strip()
		sol_text = _truncate_text(sol_text, 900)
		compact.append(
			{
				"id": s.subproblem_id,
				"status": s.status,
				"confidence": float(s.confidence),
				"proposed_solution": sol_text,
				"used_claims": list((s.used_claims or [])[:18]),
				"selected_chunk_ids": list((s.selected_chunk_ids or [])[:10]),
			}
		)
	out = json.dumps({"accepted": compact}, ensure_ascii=False)
	# Hard cap: if still too long, iteratively drop items.
	while len(out) > max_chars and len(compact) > 1:
		compact = compact[:-1]
		out = json.dumps({"accepted": compact}, ensure_ascii=False)
	# Final safeguard: truncate string (last resort) to avoid provider hard failures.
	return out[:max_chars]


def _format_claim_snippets_for_llm(claims: Sequence["EvidenceClaim"], *, max_items: int = 24, max_chars: int = 1800) -> str:
	"""Compact evidence into short, high-signal claim snippets to fit tight prompt limits."""
	lines: List[str] = []
	for c in list(claims)[:max_items]:
		cid = str(getattr(c, "claim_id", "")).strip()
		ctext = _truncate_text(str(getattr(c, "claim", "")).strip(), 160)
		conf = _safe_float(getattr(c, "confidence", 0.0), 0.0)
		if not cid or not ctext:
			continue
		lines.append(f"- {cid} (conf={conf:.2f}): {ctext}")
	out = "\n".join(lines)
	return _truncate_text(out, max_chars)


def _coerce_solution_text(value: Any) -> str:
	"""Ensure proposed_solution is a readable string even if the model returns an object."""
	if value is None:
		return ""
	if isinstance(value, str):
		return value
	try:
		return json.dumps(value, ensure_ascii=False, indent=2)
	except Exception:
		return str(value)


# ---------------------------
# Retrieval (dual model, parallel)
# ---------------------------


class DualRetriever:
	"""Dual-encoder retriever: small (vector_1) + big (vector_2) in parallel with GraphRAG."""

	def __init__(self, db: DatabaseManager, citation_db=None, collection_name: str = "research_papers"):
		self.db = db
		self.citation_db = citation_db
		self.collection_name = collection_name

		if not _SENTENCE_TRANSFORMERS_AVAILABLE:
			raise ImportError(
				"sentence-transformers is required for query-time embeddings. "
				"Install with: pip install sentence-transformers"
			)

		self.model_small = SentenceTransformer("BAAI/bge-small-en-v1.5")
		self.model_big = SentenceTransformer("all-mpnet-base-v2")
		from sentence_transformers import CrossEncoder
		self.model_cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

	def retrieve_channel(
		self,
		query: str,
		*,
		vector_name: str,
		top_k: int,
		prefetch_k: int,
		restrict_chunk_ids: Optional[Sequence[str]] = None,
	) -> List[ChunkCandidate]:
		"""Retrieve using a single embedding channel.

		- vector_1: small encoder (often good for surface similarity)
		- vector_2: big encoder (often better for mechanism/theory semantics)
		"""
		q = (query or "").strip()
		if not q:
			return []
		vn = str(vector_name or "").strip()
		if vn not in {"vector_1", "vector_2"}:
			raise ValueError(f"Unknown vector_name: {vn}")
		if vn == "vector_1":
			vec = self.model_small.encode(q, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
		else:
			vec = self.model_big.encode(q, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
		hits = self._qdrant_query(vec, vn, int(prefetch_k))
		cands: List[ChunkCandidate] = []
		for cid, score in (hits or {}).items():
			s = float(score)
			if vn == "vector_1":
				cands.append(ChunkCandidate(chunk_id=str(cid), score_small=s, score_big=0.0, combined_score=s))
			else:
				cands.append(ChunkCandidate(chunk_id=str(cid), score_small=0.0, score_big=s, combined_score=s))
		if restrict_chunk_ids:
			allowed = restrict_chunk_ids if isinstance(restrict_chunk_ids, set) else set(restrict_chunk_ids)
			cands = [c for c in cands if c.chunk_id in allowed]
		cands.sort(key=lambda c: c.combined_score, reverse=True)
		return cands[: int(top_k)]

	def retrieve(self, query: str, *, top_k: int, prefetch_k: int, restrict_chunk_ids: Optional[Sequence[str]] = None) -> List[ChunkCandidate]:
		query = (query or "").strip()
		if not query:
			return []

		vec_small = self.model_small.encode(query, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
		vec_big = self.model_big.encode(query, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

		with ThreadPoolExecutor(max_workers=2) as ex:
			fut_small = ex.submit(self._qdrant_query, vec_small, "vector_1", prefetch_k)
			fut_big = ex.submit(self._qdrant_query, vec_big, "vector_2", prefetch_k)
			small_hits = fut_small.result()
			big_hits = fut_big.result()

		merged = self._merge_hits(small_hits, big_hits)
		if restrict_chunk_ids:
			allowed = restrict_chunk_ids if isinstance(restrict_chunk_ids, set) else set(restrict_chunk_ids)
			merged = [c for c in merged if c.chunk_id in allowed]
		merged.sort(key=lambda c: c.combined_score, reverse=True)
		return merged[:top_k]

	def retrieve_two_stage(
		self,
		query: str,
		*,
		top_n: int,
		stage1_k: int = 100,
		stage2_k: int = 20,
	) -> List[ChunkCandidate]:
		"""Two-stage retrieval.

		Stage 1 (recall): vector_1 (MiniLM) retrieves a broad candidate pool.
		Stage 2 (semantic refinement): vector_2 (MPNet) reranks ONLY within that pool.

		Returns:
			List[ChunkCandidate] sorted by stage-2 (big) score.
		"""
		q = (query or "").strip()
		if not q:
			return []
		top_n = int(top_n)
		stage1_k = int(stage1_k)
		stage2_k = int(stage2_k)
		if top_n <= 0:
			return []
		if stage1_k <= 0 or stage2_k <= 0:
			raise ValueError("stage1_k and stage2_k must be positive")
		if stage2_k < top_n:
			stage2_k = top_n
		if stage1_k < stage2_k:
			stage1_k = stage2_k

		# Stage 1: broad recall using small encoder.
		stage1 = self.retrieve_channel(
			q,
			vector_name="vector_1",
			top_k=stage1_k,
			prefetch_k=stage1_k,
		)
		if not stage1:
			return []
		small_scores = {c.chunk_id: float(c.score_small or c.combined_score or 0.0) for c in stage1}
		allowed_ids = list(small_scores.keys())

		# Stage 2: rerank within Stage-1 candidates using CrossEncoder.
		stage1_rows = []
		for cid in allowed_ids:
			row = self.db.get_chunk_by_id(cid)
			if row:
				stage1_rows.append(row)
		
		if not stage1_rows:
			return []

		cross_inputs = [[q, row.get("chunk_text", "")] for row in stage1_rows]
		cross_scores = self.model_cross.predict(cross_inputs)
		
		# GraphRAG edge extraction
		cited_papers = set()
		if self.citation_db:
			# Find the top papers from stage 1 to fetch their citations
			top_stage1_papers = set()
			stage1_sorted = sorted(stage1, key=lambda c: float(c.score_small or c.combined_score or 0.0), reverse=True)
			for c in stage1_sorted[:5]:
				row = self.db.get_chunk_by_id(c.chunk_id)
				if row and row.get("paper_id"):
					top_stage1_papers.add(str(row["paper_id"]))
			
			for pid in top_stage1_papers:
				try:
					edges = self.citation_db.get_edges_for_paper(pid)
					for edge in edges:
						cited_papers.add(edge.get("cited_paper_id"))
				except Exception:
					pass

		out: List[ChunkCandidate] = []
		for row, s2 in zip(stage1_rows, cross_scores):
			cid = str(row["chunk_id"])
			b = float(s2)
			
			# GraphRAG Scoring Boost: If the chunk belongs to a cited paper, boost its score.
			pid = str(row.get("paper_id") or "")
			if pid and pid in cited_papers:
				b += 2.0  # Logit boost for being cited by a top paper
			
			s = float(small_scores.get(cid, 0.0))
			out.append(ChunkCandidate(chunk_id=cid, score_small=s, score_big=b, combined_score=b))
		out.sort(key=lambda c: c.score_big, reverse=True)
		return out[:top_n]

	def retrieve_top_chunks(
		self,
		query: str,
		*,
		top_n: int = 10,
		stage1_k: int = 100,
		stage2_k: int = 20,
	) -> List[Dict[str, Any]]:
		"""Return top-N chunk rows (with text + metadata) using two-stage retrieval."""
		cands = self.retrieve_two_stage(query, top_n=top_n, stage1_k=stage1_k, stage2_k=stage2_k)
		rows: List[Dict[str, Any]] = []
		for c in cands:
			row = self.db.get_chunk_by_id(c.chunk_id)
			if not row:
				continue
			row = dict(row)
			row["score_stage1"] = float(c.score_small)
			row["score_stage2"] = float(c.score_big)
			row["score"] = float(c.combined_score)
			rows.append(row)
		return rows

	def _qdrant_query(self, vector: np.ndarray, vector_name: str, limit: int) -> Dict[str, float]:
		res = self.db.qdrant_client.query_points(
			collection_name=self.collection_name,
			query=vector.tolist(),
			using=vector_name,
			limit=limit,
		)
		scores: Dict[str, float] = {}
		for pt in getattr(res, "points", []) or []:
			chunk_id = (pt.payload or {}).get("chunk_id")
			if chunk_id:
				scores[str(chunk_id)] = float(pt.score)
		return scores

	def _merge_hits(self, small: Dict[str, float], big: Dict[str, float]) -> List[ChunkCandidate]:
		ids = set(small.keys()) | set(big.keys())
		out: List[ChunkCandidate] = []
		for cid in ids:
			s = float(small.get(cid, 0.0))
			b = float(big.get(cid, 0.0))
			combined = 0.5 * s + 0.5 * b
			out.append(ChunkCandidate(chunk_id=cid, score_small=s, score_big=b, combined_score=combined))
		return out


def _format_paper_candidates_for_llm(papers: Sequence[Dict[str, Any]]) -> str:
	lines: List[str] = []
	for i, p in enumerate(papers, start=1):
		pid = str(p.get("paper_id") or "").strip()
		if not pid:
			continue
		year = str(p.get("year") or "").strip()
		kw = str(p.get("keywords") or "").strip()
		score = _safe_float(p.get("score"), 0.0)
		snips = p.get("snippets") or []
		if not isinstance(snips, list):
			snips = []
		lines.append(f"[{i}] paper_id={pid} score={score:.3f}" + (f" year={year}" if year else "") + (f" keywords={kw}" if kw else ""))
		for j, s in enumerate(snips[:3], start=1):
			s = _truncate_text(str(s or "").strip(), 700)
			if s:
				lines.append(f"  snippet_{j}: {s}")
		lines.append("")
	return "\n".join(lines).strip()


def _select_global_papers(
	*,
	refined_question: str,
	retriever: DualRetriever,
	db: DatabaseManager,
	llm: LLMClient,
	mechanism_mode: bool,
	top_papers: int,
	pool_papers: int,
	global_candidate_chunks: int,
	prefetch_k: int,
	snippets_per_paper: int,
) -> Tuple[List[str], Optional[Set[str]]]:
	"""Pick a global shortlist of paper_ids, then build a restrict_chunk_ids set spanning all their chunks."""
	q = (refined_question or "").strip()
	if not q:
		return [], None

	# Retrieve a broad pool of candidates (avoid per-subproblem starvation later).
	# For mechanism questions, also try a mechanism-expanded query via vector_2.
	cands: List[ChunkCandidate] = []
	try:
		if mechanism_mode:
			mech_q = _truncate_text(q + " mechanism theory model framework dynamics", 400)
			c1 = retriever.retrieve_channel(
				q,
				vector_name="vector_1",
				top_k=int(global_candidate_chunks),
				prefetch_k=int(max(prefetch_k, global_candidate_chunks)),
			)
			c2 = retriever.retrieve_channel(
				mech_q,
				vector_name="vector_2",
				top_k=int(global_candidate_chunks),
				prefetch_k=int(max(prefetch_k, global_candidate_chunks)),
			)
			by_id: Dict[str, ChunkCandidate] = {}
			for c in (c1 + c2):
				ex = by_id.get(c.chunk_id)
				if ex is None or c.combined_score > ex.combined_score:
					by_id[c.chunk_id] = c
			cands = list(by_id.values())
			cands.sort(key=lambda x: x.combined_score, reverse=True)
			cands = cands[: int(global_candidate_chunks)]
		else:
			cands = retriever.retrieve(
				q,
				top_k=int(global_candidate_chunks),
				prefetch_k=int(max(prefetch_k, global_candidate_chunks)),
			)
	except Exception:
		cands = []

	if not cands:
		return [], None

	# Group by paper_id and capture a few snippets for LLM ranking.
	papers: Dict[str, Dict[str, Any]] = {}
	for c in cands:
		row = db.get_chunk_by_id(c.chunk_id)
		if not row:
			continue
		pid = str(row.get("paper_id") or "").strip()
		if not pid:
			continue
		rec = papers.get(pid)
		if rec is None:
			rec = {
				"paper_id": pid,
				"score": float(c.combined_score),
				"year": row.get("year"),
				"keywords": row.get("keywords"),
				"snippets": [],
			}
			papers[pid] = rec
		else:
			rec["score"] = max(float(rec.get("score") or 0.0), float(c.combined_score))
		if len(rec["snippets"]) < int(snippets_per_paper):
			text = str(row.get("chunk_text") or "").strip()
			if text:
				rec["snippets"].append(_truncate_text(text, 700))

	paper_list = list(papers.values())
	paper_list.sort(key=lambda r: _safe_float(r.get("score"), 0.0), reverse=True)
	pool = paper_list[: max(1, int(pool_papers))]

	selected: List[str] = []
	try:
		prompt = PROMPT_GLOBAL_PAPER_SELECTION
		prompt = prompt.replace("{{REFINED_QUESTION}}", q)
		prompt = prompt.replace("{{TOP_PAPERS}}", str(max(1, int(top_papers))))
		prompt = prompt.replace("{{PAPER_LIST}}", _format_paper_candidates_for_llm(pool))
		obj = llm.chat_json(user_prompt=prompt, temperature=0.0, max_tokens=420)
		ids = obj.get("selected_paper_ids") or []
		if isinstance(ids, list):
			selected = [str(x).strip() for x in ids if str(x).strip()]
	except Exception:
		selected = []

	if not selected:
		selected = [str(p.get("paper_id")) for p in pool[: max(1, int(top_papers))] if str(p.get("paper_id") or "").strip()]

	selected = list(dict.fromkeys(selected))[: max(1, int(top_papers))]

	allowed: Set[str] = set()
	for pid in selected:
		try:
			for cid in (db.get_chunk_ids_by_paper(pid) or []):
				allowed.add(str(cid))
		except Exception:
			continue

	return selected, (allowed if allowed else None)


# ---------------------------
# Orchestrator
# ---------------------------


def _env_int(name: str, default: int) -> int:
	try:
		return int(os.getenv(name, str(default)))
	except Exception:
		return default




def _paper_id_from_chunk_id(chunk_id: str) -> str:
	"""Extract a stable paper id from a chunk id.

	Expected chunk_id formats in this repo commonly look like:
	- '<paper>.tei.tei_chunk-12'
	- '<paper>_chunk-12'
	"""
	val = str(chunk_id or "").strip()
	if not val:
		return ""
	if "_chunk-" in val:
		return val.split("_chunk-", 1)[0]
	return val


_SLOT_STOPWORDS = {
	"the",
	"a",
	"an",
	"and",
	"or",
	"of",
	"to",
	"in",
	"on",
	"for",
	"with",
	"by",
	"from",
	"as",
	"at",
	"into",
	"via",
	"role",
	"effect",
	"effects",
	"impact",
	"impacts",
	"mechanism",
	"mechanisms",
}


def _slot_tokens(slot: str) -> List[str]:
	val = re.sub(r"[^a-z0-9\s\-_/]", " ", str(slot or "").strip().lower())
	val = re.sub(r"\s+", " ", val).strip()
	if not val:
		return []
	parts = [p for p in re.split(r"[\s\-_/]+", val) if p]
	# Keep informative tokens only
	return [p for p in parts if len(p) >= 4 and p not in _SLOT_STOPWORDS]


def _slot_supported_by_text(slot: str, texts: Sequence[str]) -> bool:
	"""Heuristic: consider a slot supported if its key tokens appear in inferred text.

	This is intentionally lightweight/deterministic to avoid extra LLM calls.
	"""
	phrase = str(slot or "").strip().lower()
	if not phrase:
		return False
	toks = _slot_tokens(phrase)
	if not toks:
		return False
	for t in texts or []:
		txt = str(t or "").lower()
		if not txt:
			continue
		# Exact phrase match is strongest.
		if phrase in txt:
			return True
		# Otherwise, any key-token match.
		for tok in toks:
			if tok in txt:
				return True
	return False


def _effective_slot_coverage(
	*,
	required_slots: Sequence[str],
	supported_slots: Sequence[str],
	inferred_texts: Sequence[str],
) -> Tuple[float, List[str], List[str]]:
	"""Return (ratio, effective_supported, effective_missing)."""
	required = [str(s) for s in (required_slots or []) if str(s).strip()]
	if not required:
		return 1.0, [], []
	supported_set = {str(s) for s in (supported_slots or []) if str(s).strip()}
	effective_supported: List[str] = []
	effective_missing: List[str] = []
	for slot in required:
		if slot in supported_set:
			effective_supported.append(slot)
			continue
		if _slot_supported_by_text(slot, inferred_texts):
			effective_supported.append(slot)
		else:
			effective_missing.append(slot)
	ratio = float(len(effective_supported)) / float(max(1, len(required)))
	return ratio, effective_supported, effective_missing


def _classify_question_type(text: str) -> str:
	"""Cheap, deterministic question-type classifier.

	Returns: FACT | MECHANISM | COMPARISON | PROCEDURE | OTHER
	"""
	q = (text or "").strip().lower()
	q0 = re.sub(r"\s+", " ", q)
	if not q0:
		return "OTHER"
	# Strip leading punctuation/quotes
	q0 = re.sub(r"^[^a-z0-9]+", "", q0)
	if q0.startswith(("why ", "why does", "why do", "how ", "how does", "how do", "explain ", "what causes", "what drives")):
		return "MECHANISM"
	if any(k in q0 for k in ["compare", "difference", "vs ", "versus", "better than", "tradeoff"]):
		return "COMPARISON"
	if q0.startswith(("how to", "how can i", "steps", "procedure", "pipeline", "method to")):
		return "PROCEDURE"
	if q0.startswith(("what ", "when ", "who ", "where ", "which ")):
		return "FACT"
	# Fallback: mechanism words
	if any(k in q0 for k in ["mechanism", "regime shift", "tipping", "bifurcation", "nonlinear", "feedback", "dynamics"]):
		return "MECHANISM"
	return "OTHER"


def _section_is_theory_heavy(section: str) -> bool:
	s = (section or "").strip().lower()
	if not s:
		return False
	return any(k in s for k in ["discussion", "conclusion", "conclusions", "interpretation", "theory", "background", "related work"])


def _window_for_chunk(*, base_window: int, section: str, question_type: str) -> int:
	win = int(base_window)
	if question_type == "MECHANISM":
		win = max(win, 1)
		if _section_is_theory_heavy(section):
			win = max(win, 2)
			# Give a bit more context around theory-heavy sections
			win = min(win + 1, 3)
	return max(0, win)


def _claims_look_observational(claims: Sequence["EvidenceClaim"]) -> bool:
	"""Heuristic: mechanisms are mostly empty/low-information."""
	cs = list(claims or [])
	if not cs:
		return True
	mech_nonempty = 0
	for c in cs:
		m = str(getattr(c, "mechanism", "") or "").strip()
		if len(m) >= 12:
			mech_nonempty += 1
	return (float(mech_nonempty) / float(max(1, len(cs)))) < 0.35


def _support_adjust_confidence(
	*,
	base_conf: float,
	used_claim_ids: Sequence[str],
	claims: Sequence["EvidenceClaim"],
	mechanism_clusters: Sequence[Dict[str, Any]],
) -> float:
	"""Boost confidence when support converges across papers/mechanism clusters."""
	conf = max(0.0, min(1.0, float(base_conf)))
	used = {str(x) for x in (used_claim_ids or []) if str(x).strip()}
	if not used:
		return conf
	by_id: Dict[str, EvidenceClaim] = {c.claim_id: c for c in (claims or [])}
	papers: set[str] = set()
	for cid in used:
		c = by_id.get(cid)
		if not c:
			continue
		pid = _paper_id_from_chunk_id(c.source_chunk_id)
		if pid:
			papers.add(pid)

	multi_paper_used = 0
	for cl in mechanism_clusters or []:
		if not isinstance(cl, dict):
			continue
		strength = str(cl.get("cluster_strength") or "").strip().lower()
		if strength != "multi_paper":
			continue
		members = {str(x) for x in (cl.get("member_claim_ids") or []) if str(x).strip()}
		if members and (members & used):
			multi_paper_used += 1

	# Support-based boost: multi-paper agreement + multi_paper cluster usage.
	boost = 0.0
	if len(papers) >= 2:
		boost += 0.06 * min(3, max(0, len(papers) - 1))
	if multi_paper_used:
		boost += 0.05 * min(3, multi_paper_used)
	return max(0.0, min(1.0, conf + boost))


def _format_claims_for_llm(claims: Sequence[EvidenceClaim]) -> str:
	items = []
	for c in claims:
		items.append(
			{
				"id": c.claim_id,
				"claim": c.claim,
				"mechanism": c.mechanism,
				"conditions": c.conditions,
				"confidence": float(c.confidence),
				"source_chunk_id": c.source_chunk_id,
				"paper_id": _paper_id_from_chunk_id(c.source_chunk_id),
			}
		)
	return json.dumps({"claims": items}, ensure_ascii=False)


def _format_dependency_context_for_llm(
	*,
	depends_on: Sequence[str],
	accepted: Dict[str, "SubProblemSolution"],
	failed: Dict[str, "SubProblemSolution"],
) -> str:
	deps_accepted = [asdict(accepted[d]) for d in depends_on if d in accepted]
	deps_failed = [asdict(failed[d]) for d in depends_on if d in failed]
	return json.dumps(
		{
			"depends_on": [str(d) for d in (depends_on or [])],
			"accepted_dependencies": deps_accepted,
			"failed_dependencies": deps_failed,
		},
		ensure_ascii=False,
	)


def _extract_citation_numbers(items: Sequence[str]) -> List[str]:
	out: List[str] = []
	for it in items or []:
		if it is None:
			continue
		s = str(it)
		# accept "13" or "[13]" or "citation 13"
		m = re.search(r"\b(\d+)\b", s)
		if m:
			n = m.group(1)
			if n not in out:
				out.append(n)
	return out

def _normalize_proposed_solution(text: str) -> str:
	"""Normalize proposer output to avoid accidental '| INSUFFICIENT' suffixes."""
	val = (text or "").strip()
	# Common pattern caused by older prompt examples.
	upper = val.upper()
	if upper == "INSUFFICIENT":
		return "INSUFFICIENT"
	# If it ends with a suffix like "| INSUFFICIENT" or "- INSUFFICIENT", strip it.
	for sep in ("|", "-", "—"):
		suffix = f"{sep} INSUFFICIENT"
		if upper.endswith(suffix):
			left = val[: -len(suffix)].strip()
			return left or "INSUFFICIENT"
	# Also handle plain trailing "INSUFFICIENT" after whitespace.
	if upper.endswith(" INSUFFICIENT") and len(val) > len("INSUFFICIENT"):
		left = val[: -len("INSUFFICIENT")].strip(" \t|-—")
		return left or "INSUFFICIENT"
	return val

def _normalize_used_claims(used_claims: Sequence[str], claims: Sequence[EvidenceClaim]) -> List[str]:
	"""Map short claim refs like 'claim_1' to actual claim IDs, and filter to known IDs."""
	claim_ids = [c.claim_id for c in claims]
	claim_id_set = set(claim_ids)
	out: List[str] = []
	for raw in used_claims or []:
		key = str(raw).strip()
		if not key:
			continue
		if key in claim_id_set:
			out.append(key)
			continue
		# Heuristic mapping: 'claim_3' -> third item in the evidence list.
		m = re.match(r"^claim_(\d+)$", key)
		if m:
			idx = int(m.group(1)) - 1
			if 0 <= idx < len(claim_ids):
				out.append(claim_ids[idx])
				continue
	# De-duplicate preserving order
	return list(dict.fromkeys(out))

def _safe_float(x: Any, default: float = 0.0) -> float:
	try:
		v = float(x)
		if np.isnan(v) or np.isinf(v):
			return default
		return v
	except Exception:
		return default


def _truncate_text(text: str, max_chars: int) -> str:
	val = (text or "").strip()
	if len(val) <= max_chars:
		return val
	return val[: max(0, int(max_chars) - 1)].rstrip() + "…"


def _build_running_summary_for_llm(
	*,
	refined_question: str,
	accepted: Dict[str, "SubProblemSolution"],
	failed: Dict[str, "SubProblemSolution"],
	best_effort: Dict[str, "SubProblemSolution"],
	hypotheses: Optional[Dict[str, Dict[str, Any]]] = None,
	max_items: int = 6,
) -> str:
	"""Build a short, strictly-factual run summary for stateless LLM calls.

	This summary is NOT a model-generated summary (no extra LLM calls).
	It only includes already-produced outputs (statuses/solutions/claim ids).
	"""
	items: List[Dict[str, Any]] = []

	# Prefer accepted, then partial/failed from best_effort.
	for sid, sol in accepted.items():
		items.append(
			{
				"id": sid,
				"status": sol.status,
				"confidence": float(sol.confidence),
				"proposed_solution": _truncate_text(sol.proposed_solution, 400),
				"used_claims": list(sol.used_claims or []),
				"selected_chunk_ids": list(sol.selected_chunk_ids or []),
			}
		)

	# Add up to remaining slots from best_effort (excluding those already included)
	if len(items) < int(max_items):
		for sid, sol in best_effort.items():
			if sid in accepted:
				continue
			items.append(
				{
					"id": sid,
					"status": sol.status,
					"confidence": float(sol.confidence),
					"proposed_solution": _truncate_text(sol.proposed_solution, 260),
					"used_claims": list(sol.used_claims or []),
					"selected_chunk_ids": list(sol.selected_chunk_ids or []),
					"failure_reasons": [
						_truncate_text(str(r), 140) for r in (sol.failure_reasons or [])
					][:4],
				}
			)
			if len(items) >= int(max_items):
				break

	# If best_effort is empty for some reason, fall back to failed.
	if not items and failed:
		for sid, sol in failed.items():
			items.append(
				{
					"id": sid,
					"status": sol.status,
					"confidence": float(sol.confidence),
					"proposed_solution": _truncate_text(sol.proposed_solution, 220),
					"failure_reasons": [
						_truncate_text(str(r), 140) for r in (sol.failure_reasons or [])
					][:4],
				}
			)
			if len(items) >= int(max_items):
				break

	return json.dumps(
		{
			"refined_question": _truncate_text(str(refined_question), 500),
			"hypotheses": hypotheses or {},
			"summary_items": items,
			"note": "This is a run summary of prior module outputs only. Use it only as context; do not invent facts.",
		},
		ensure_ascii=False,
	)


def _natural_sort_key(text: str) -> List[Any]:
	"""Sort key that handles IDs like SP2, SP10, SP2.1, SP2.10."""
	parts = re.split(r"(\d+)", str(text))
	key: List[Any] = []
	for p in parts:
		if p.isdigit():
			key.append(int(p))
		else:
			key.append(p)
	return key


def _safe_get_paper_meta(db: DatabaseManager, paper_id: str) -> Dict[str, Any]:
	"""Best-effort paper metadata lookup using first chunk in Qdrant."""
	pid = str(paper_id)
	try:
		chunks = db.get_chunks_by_paper(pid)
		if chunks:
			return {
				"paper_id": pid,
				"title": "",
				"year": chunks[0].get("year"),
				"source": "",
				"total_chunks": len(chunks),
			}
		return {"paper_id": pid}
	except Exception:
		return {"paper_id": pid}


def _get_chunk_window_rows(db: DatabaseManager, chunk_id: str, window: int = 1) -> List[Dict[str, Any]]:
	"""Fetch current chunk and up to N prev/next chunks using Postgres links."""
	cur = db.get_chunk_by_id(str(chunk_id))
	if not cur:
		return []
	rows: List[Dict[str, Any]] = []
	# walk backwards
	prev_id = cur.get("prev_chunk_id")
	back: List[Dict[str, Any]] = []
	for _ in range(max(0, int(window))):
		if not prev_id:
			break
		pr = db.get_chunk_by_id(str(prev_id))
		if not pr:
			break
		back.append(pr)
		prev_id = pr.get("prev_chunk_id")
	# current
	rows.extend(reversed(back))
	rows.append(cur)
	# walk forwards
	next_id = cur.get("next_chunk_id")
	for _ in range(max(0, int(window))):
		if not next_id:
			break
		nr = db.get_chunk_by_id(str(next_id))
		if not nr:
			break
		rows.append(nr)
		next_id = nr.get("next_chunk_id")
	# de-dup by chunk_id preserve order
	out: List[Dict[str, Any]] = []
	seen: set[str] = set()
	for r in rows:
		cid = str(r.get("chunk_id"))
		if cid and cid not in seen:
			seen.add(cid)
			out.append(r)
	return out


def _build_chunk_window_text(rows: Sequence[Dict[str, Any]], max_chars: int) -> str:
	"""Build a windowed text block with lightweight separators."""
	parts: List[str] = []
	for r in rows:
		cid = str(r.get("chunk_id", ""))
		sec = str(r.get("section", "") or "")
		text = str(r.get("chunk_text", "") or "")
		header = f"[chunk_id={cid}]"
		if sec:
			header += f" [section={sec}]"
		parts.append(header + "\n" + text.strip())
	joined = "\n\n---\n\n".join([p for p in parts if p.strip()])
	return _truncate_text(joined, int(max_chars))


def run(
	*,
	problem: str,
	confidence_target: float,
	max_depth: int,
	max_iterations: int,
	top_k: int,
	prefetch_k: int,
	llm: LLMClient,
	max_llm_workers: int,
	max_subproblems: int,
	fast_mode: bool,
	use_hypothesis: bool,
	seed_chunk_id: Optional[str] = None,
	policy: str = "lenient",
	use_global_paper_shortlist: bool = True,
	global_top_papers: int = 12,
	global_paper_pool: int = 24,
	global_candidate_chunks: int = 160,
	global_snippets_per_paper: int = 2,
):
	policy = (policy or "lenient").strip().lower()
	if policy not in {"lenient", "strict", "balanced"}:
		policy = "lenient"
	strict_policies = policy == "strict"
	lenient_policies = policy == "lenient"

	# Fast-mode token budgets (to keep runs quick/cheap)
	if fast_mode:
		TOK_REFINE = 320
		TOK_PLAN = 520
		TOK_QUERY = 160
		TOK_JUDGE = 200
		TOK_EXTRACT = 520
		TOK_PROPOSE = 240
		TOK_VALIDATE = 200
		TOK_NAV = 180
		TOK_EXPAND = 420
		TOK_SYNTH = 520
	else:
		TOK_REFINE = 700
		TOK_PLAN = 900
		TOK_QUERY = 250
		TOK_JUDGE = 300
		TOK_EXTRACT = 900
		TOK_PROPOSE = 700
		TOK_VALIDATE = 450
		TOK_NAV = 350
		TOK_EXPAND = 650
		TOK_SYNTH = 900

	chunk_text_limit = 2500 if fast_mode else 7000
	# Default context window: we will override per subproblem based on question type/section.
	chunk_window = 1

	# Stores
	out_dir = Path("RAG_LOGS")
	graph_store = ContextGraphStore(out_dir)

	# DB connections
	q_host = str(QDRANT_CONFIG_DEFAULT.get("host", "localhost"))
	q_port = int(QDRANT_CONFIG_DEFAULT.get("port", 6333))
	collection = str(QDRANT_CONFIG_DEFAULT.get("collection", "research_papers"))

	db = DatabaseManager(qdrant_host=q_host, qdrant_port=q_port, collection_name=collection)
	db.connect_qdrant()

	citation_db = CitationGraphManager()
	citation_db.connect()

	retriever = DualRetriever(db, citation_db=citation_db, collection_name=collection)
	global_claims = GlobalClaimStore()

	global_claims = GlobalClaimStore()

	# SEMANTIC CACHE: Check for similar queries
	cache_db_path = out_dir / "semantic_cache.sqlite"
	import sqlite3
	try:
		with sqlite3.connect(cache_db_path) as conn:
			conn.execute('''CREATE TABLE IF NOT EXISTS semantic_cache (
								id INTEGER PRIMARY KEY AUTOINCREMENT,
								question TEXT UNIQUE,
								embedding BLOB,
								final_solution TEXT
							)''')
			cursor = conn.cursor()
			cursor.execute("SELECT question, embedding, final_solution FROM semantic_cache")
			cached_rows = cursor.fetchall()

			if cached_rows:
				# Embed the incoming user problem to compare
				prob_vec = retriever.model_small.encode(str(problem), convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

				for cached_q, blob, cached_solution in cached_rows:
					try:
						cached_vec = np.frombuffer(blob, dtype=np.float32)
						# Compute Cosine Similarity
						sim = np.dot(prob_vec, cached_vec) / (np.linalg.norm(prob_vec) * np.linalg.norm(cached_vec))
						if sim > 0.95:
							print(f"\n[SEMANTIC CACHE HIT] Found similar query (>0.95 similarity): '{cached_q}'")
							print("Returning cached final solution immediately.")
							print("\n" + str(cached_solution).strip())
							return
					except Exception:
						continue
	except Exception as e:
		print(f"[CACHE ERROR] Ignored: {e}")

	# MODULE 1 — QUESTION REFINER
	refiner_prompt = PROMPT_REFINER.replace("{{USER_PROBLEM}}", str(problem))
	refined_obj = llm.chat_json(user_prompt=refiner_prompt, temperature=0.0, max_tokens=TOK_REFINE)
	refined = RefinedProblem(
		original=str(problem),
		refined_question=str(refined_obj.get("refined_question", "")).strip(),
		explicit_assumptions=[str(a) for a in (refined_obj.get("explicit_assumptions") or [])],
		scope=str(refined_obj.get("scope", "")).strip() or "review",
	)
	if not refined.refined_question:
		raise ValueError("Refiner returned empty refined_question")

	print("\nRefined question:")
	print(refined.refined_question)
	if refined.explicit_assumptions:
		print("Assumptions:")
		for a in refined.explicit_assumptions:
			print(f"- {a}")
	print(f"Scope: {refined.scope}")

	# MODULE 2 — SUB-PROBLEM PLANNER
	planner_prompt = PROMPT_PLANNER.replace("{{REFINED_QUESTION}}", refined.refined_question)
	planner_prompt = planner_prompt.replace("{{MAX_SUBPROBLEMS}}", str(max(1, int(max_subproblems))))
	planned = llm.chat_json(user_prompt=planner_prompt, temperature=0.0, max_tokens=TOK_PLAN)
	sp_items = planned.get("subproblems") or []
	if not isinstance(sp_items, list) or not sp_items:
		raise ValueError("Planner returned no subproblems")
	# Safety clamp (even if the model ignores MAX_SUBPROBLEMS)
	sp_items = sp_items[: max(1, int(max_subproblems))]

	queue: List[PlannedSubProblem] = []
	for sp in sp_items:
		queue.append(
			PlannedSubProblem(
				id=str(sp.get("id", "")).strip() or f"SP{len(queue)+1}",
				question=str(sp.get("question", "")).strip(),
				depends_on=[str(d) for d in (sp.get("depends_on") or [])],
				depth=0,
			)
		)

	print("\nPlanned sub-problems:")
	for sp in queue:
		deps = ",".join(sp.depends_on) if sp.depends_on else "-"
		print(f"- {sp.id} (deps: {deps}): {sp.question}")
	print(f"\nSteps remaining before final answer: {len(queue)}")

	# Execution state
	accepted: Dict[str, SubProblemSolution] = {}
	failed: Dict[str, SubProblemSolution] = {}
	best_effort: Dict[str, SubProblemSolution] = {}
	best_effort_claims: Dict[str, List[EvidenceClaim]] = {}
	hypotheses: Dict[str, Dict[str, Any]] = {}
	required_slots_by_sp: Dict[str, List[str]] = {}
	citation_priority_ids: Optional[List[str]] = None
	citations_followed: List[str] = []
	global_selected_paper_ids: List[str] = []
	global_allowed_chunk_ids: Optional[Set[str]] = None

	def _run_summary_json() -> str:
		# Keep this small to avoid provider prompt-token limits.
		s = _build_running_summary_for_llm(
			refined_question=refined.refined_question,
			accepted=accepted,
			failed=failed,
			best_effort=best_effort,
			hypotheses=hypotheses,
			max_items=3,
		)
		return _truncate_text(s, 900)

	# Optional seed: treat as initial priority scope (chunks within cited papers)
	if seed_chunk_id:
		try:
			ctx = citation_db.get_citation_context_for_ai(seed_chunk_id)
			# Give the model a chance to choose citations immediately
			nav_prompt = PROMPT_CITATION_NAV
			nav_prompt = nav_prompt.replace("{{FAILURE_REASONS}}", "Initial seed provided; decide if we should follow citations first.")
			nav_prompt = nav_prompt.replace("{{CITATION_LIST}}", ctx)
			dec = llm.chat_json(user_prompt=nav_prompt, temperature=0.0, max_tokens=400)
			if bool(dec.get("need_more_papers")):
				nums = _extract_citation_numbers(dec.get("target_citations") or [])
				if nums:
					res = citation_db.get_vector_ids_for_citations(seed_chunk_id, nums)
					ids = res.get("vector_ids") or []
					if ids:
						citation_priority_ids = list(dict.fromkeys([str(x) for x in ids]))
						for n in nums:
							if n not in citations_followed:
								citations_followed.append(n)
						print(f"Seed: following citations {nums} (priority chunks: {len(citation_priority_ids)})")
		except Exception:
			# Seed is optional; don't block.
			pass

	# Process queue in-order.
	# Use the original user phrasing first so "Why/How" questions remain MECHANISM
	# even if refinement rewrites them as "What triggers...".
	main_qtype = _classify_question_type(problem)
	if main_qtype == "OTHER":
		main_qtype = _classify_question_type(refined.refined_question)
	if fast_mode and main_qtype == "MECHANISM":
		raise RuntimeError(
			"FAST/verification mode cannot be used for mechanism discovery. "
			"It disables citation hops/expansion and will produce false negatives. "
			"Re-run without --fast/--fast-verification."
		)

	# GLOBAL PAPER SHORTLIST — retrieve top papers once and reuse across all subproblems.
	# Best-effort: if it fails, we continue with unrestricted retrieval.
	if use_global_paper_shortlist:
		try:
			global_selected_paper_ids, global_allowed_chunk_ids = _select_global_papers(
				refined_question=refined.refined_question,
				retriever=retriever,
				db=db,
				llm=llm,
				mechanism_mode=(main_qtype == "MECHANISM"),
				top_papers=max(1, int(global_top_papers)),
				pool_papers=max(1, int(global_paper_pool)),
				global_candidate_chunks=max(20, int(global_candidate_chunks)),
				prefetch_k=prefetch_k,
				snippets_per_paper=max(1, int(global_snippets_per_paper)),
			)
			if global_selected_paper_ids:
				print("\nGlobal paper shortlist:")
				for pid in global_selected_paper_ids:
					print(f"- {pid}")
				if global_allowed_chunk_ids is not None:
					print(f"Global allowed chunks: {len(global_allowed_chunk_ids)}")
			if citation_priority_ids and global_allowed_chunk_ids is not None:
				citation_priority_ids = [cid for cid in citation_priority_ids if str(cid) in global_allowed_chunk_ids]
				if not citation_priority_ids:
					citation_priority_ids = None
		except Exception:
			global_selected_paper_ids = []
			global_allowed_chunk_ids = None
	if use_hypothesis:
		print(
			"\nNote: --use-hypothesis can indirectly bias retrieval via query rewriting/keywording. "
			"For strict evaluation, compare runs with and without --use-hypothesis."\
		)

	total_steps = len(queue)
	completed_steps = 0
	idx = 0
	while idx < len(queue):
		sp = queue[idx]
		idx += 1

		sp_qtype = _classify_question_type(sp.question)
		mechanism_mode = sp_qtype == "MECHANISM"

		# Don't process the same id multiple times.
		if sp.id in accepted or sp.id in failed:
			continue

		completed_steps += 1
		steps_left = max(0, total_steps - completed_steps)
		print(f"\nProgress: {steps_left} steps left (processing {completed_steps}/{total_steps}: {sp.id})")

		dep_ctx = _format_dependency_context_for_llm(
			depends_on=sp.depends_on,
			accepted=accepted,
			failed=failed,
		)
		deps_missing = [d for d in sp.depends_on if d not in accepted]
		if deps_missing:
			print(f"\n[{sp.id}] dependencies missing (continuing anyway): {', '.join(deps_missing)}")

		# Solve this subproblem with retries/backtracking
		best_solution: Optional[SubProblemSolution] = None
		best_solution_claims: List[EvidenceClaim] = []
		prev_query: str = ""
		prev_failures: List[str] = []
		prev_claim_snippets: List[str] = []
		for iteration in range(1, max_iterations + 1):
			# OPTIONAL — Determine evidence slots (coverage tracking)
			# IMPORTANT: For non-FACT questions (MECHANISM/OTHER), skip slot generation entirely to avoid anchoring.
			slots: List[str] = []
			if sp_qtype == "FACT":
				slots = required_slots_by_sp.get(sp.id) or []
				if not slots:
					slot_prompt = PROMPT_SLOT_SCHEMA.replace("{{SUBPROBLEM_QUESTION}}", sp.question)
					slot_prompt = slot_prompt + "\n\nRun summary (JSON):\n" + _run_summary_json()
					try:
						so = llm.chat_json(user_prompt=slot_prompt, temperature=0.0, max_tokens=220 if fast_mode else 320)
						slots = [str(x) for x in (so.get("required_slots") or []) if str(x).strip()]
						slots = slots[:7]
					except Exception:
						slots = []
				required_slots_by_sp[sp.id] = list(slots)
			else:
				required_slots_by_sp[sp.id] = []
			# OPTIONAL — Hypothesis (prior-knowledge) to improve retrieval/extraction only.
			hypo_block = ""
			if use_hypothesis and sp.id not in hypotheses:
				h_prompt = PROMPT_HYPOTHESIS.replace("{{SUBPROBLEM_QUESTION}}", sp.question)
				try:
					h_obj = llm.chat_json_custom_system(
						system_prompt=GLOBAL_RULES_SYSTEM_HYPOTHESIS,
						user_prompt=h_prompt,
						temperature=0.2,
						max_tokens=320 if fast_mode else 520,
					)
					hypotheses[sp.id] = {
						"hypothesis_answer": _truncate_text(str(h_obj.get("hypothesis_answer", "")).strip(), 500),
						"keywords": [str(x) for x in (h_obj.get("keywords") or [])][:16],
						"synonyms": [str(x) for x in (h_obj.get("synonyms") or [])][:16],
						"do_not_assume": [str(x) for x in (h_obj.get("do_not_assume") or [])][:10],
					}
				except Exception:
					# Hypothesis is optional; do not block the run.
					hypotheses[sp.id] = {}

			if use_hypothesis and hypotheses.get(sp.id):
				h = hypotheses.get(sp.id) or {}
				hypo_block = (
					"\n\nDraft answer (NOT evidence; may be wrong):\n"
					+ _truncate_text(str(h.get("hypothesis_answer", "")).strip(), 650)
					+ "\n"
					+ "Draft constraints (do_not_assume): "
					+ json.dumps(h.get("do_not_assume") or [], ensure_ascii=False)
				)

			# MODULE 3 — retrieval query generator
			q_prompt = PROMPT_RETRIEVAL_QUERY.replace(
				"{{SUBPROBLEM_QUESTION}}",
				sp.question
				+ "\n\nDependency context (may be incomplete):\n"
				+ dep_ctx
				+ "\n\nRun summary (JSON):\n"
				+ _run_summary_json(),
			)
			try:
				q_obj = llm.chat_json(user_prompt=q_prompt, temperature=0.0, max_tokens=TOK_QUERY)
				search_query = str(q_obj.get("search_query", "")).strip() or sp.question
			except Exception:
				# If the model fails to produce valid JSON, fall back to the raw subproblem text.
				search_query = sp.question

			# Evidence-driven query rewrite (from previous iteration), to improve multi-hop retrieval.
			if iteration > 1 and prev_query and (prev_failures or prev_claim_snippets):
				rewrite_prompt = PROMPT_QUERY_REWRITE
				rewrite_prompt = rewrite_prompt.replace("{{SUBPROBLEM_QUESTION}}", sp.question)
				rewrite_prompt = rewrite_prompt.replace("{{PREV_QUERY}}", prev_query)
				rewrite_prompt = rewrite_prompt.replace(
				"{{EVIDENCE_SNIPPETS}}",
				"\n".join([f"- {s}" for s in prev_claim_snippets[:10]]) or "(none)",
			)
				rewrite_prompt = rewrite_prompt.replace(
				"{{FAILURE_REASONS}}",
				"\n".join([f"- {s}" for s in prev_failures[:10]]) or "(none)",
			)
				rewrite_prompt = rewrite_prompt + "\n\nRun summary (JSON):\n" + _run_summary_json()
				try:
					rw = llm.chat_json(user_prompt=rewrite_prompt, temperature=0.0, max_tokens=max(TOK_QUERY, 220))
					rw_q = str(rw.get("search_query", "")).strip()
					if rw_q and rw_q.upper() != "INSUFFICIENT":
						search_query = rw_q
				except Exception:
					pass

			# NOTE: Hypothesis is intentionally NOT used to alter retrieval.

			# Two-channel retrieval intent:
			# - Observational channel: favors vector_1
			# - Mechanism/theory channel: favors vector_2
			obs_query = str(search_query)
			mech_query = str(search_query)
			if mechanism_mode:
				mech_query = _truncate_text(mech_query + " mechanism theory model framework dynamics", 400)
			else:
				mech_query = ""

			def _interleave_candidates(a: Sequence[ChunkCandidate], b: Sequence[ChunkCandidate], *, limit: int) -> List[ChunkCandidate]:
				out: List[ChunkCandidate] = []
				seen: set[str] = set()
				ia = 0
				ib = 0
				while len(out) < int(limit) and (ia < len(a) or ib < len(b)):
					if ia < len(a):
						cid = a[ia].chunk_id
						if cid not in seen:
							seen.add(cid)
							out.append(a[ia])
						ia += 1
					else:
						ia += 1
					if len(out) >= int(limit):
						break
					if ib < len(b):
						cid = b[ib].chunk_id
						if cid not in seen:
							seen.add(cid)
							out.append(b[ib])
						ib += 1
					else:
						ib += 1
				return out

			# Retrieval
			if mechanism_mode and mech_query.strip():
				obs_cands = retriever.retrieve_channel(
					obs_query,
					vector_name="vector_1",
					top_k=max(top_k * 2, 10),
					prefetch_k=prefetch_k,
					restrict_chunk_ids=global_allowed_chunk_ids,
				)
				mech_cands = retriever.retrieve_channel(
					mech_query,
					vector_name="vector_2",
					top_k=max(top_k * 2, 10),
					prefetch_k=prefetch_k,
					restrict_chunk_ids=global_allowed_chunk_ids,
				)
				candidates = _interleave_candidates(obs_cands, mech_cands, limit=max(top_k * 3, 18))
			else:
				candidates = retriever.retrieve(
					search_query,
					top_k=max(top_k * 3, 18),
					prefetch_k=prefetch_k,
					restrict_chunk_ids=global_allowed_chunk_ids,
				)

			# Citation-followed chunks are a soft boost (not a hard filter).
			if candidates and citation_priority_ids:
				boost_ids = set([str(x) for x in citation_priority_ids])
				for c in candidates:
					if c.chunk_id in boost_ids:
						c.combined_score = float(c.combined_score) + 0.12
				candidates.sort(key=lambda c: c.combined_score, reverse=True)
			if not candidates:
				# Continue best-effort with empty evidence rather than hard-failing.
				prev_failures = list(prev_failures) + ["No retrieval candidates returned"]
				candidate_rows = []
			else:
				# Fetch top candidate chunk texts from Postgres (limit for cost)
				candidate_rows = []
				for cand in candidates[: max(10, top_k * 2)]:
					row = db.get_chunk_by_id(cand.chunk_id)
					if row:
						candidate_rows.append((cand, row))

			# MODULE 4 — Chunk selection
			# In fast mode, give less importance to LLM judging (latency + brittleness).
			# Prefer vector retrieval ranking directly.
			judged: List[JudgedChunk] = []
			if fast_mode:
				selected_chunk_ids = [cand.chunk_id for (cand, _row) in candidate_rows[:5]]
			else:
				# Judge chunks (parallel) and select up to 5 (prefer relevance >= 0.4)
				section_by_chunk: Dict[str, str] = {str(row.get("chunk_id")): str(row.get("section") or "") for (_cand, row) in candidate_rows}
				with ThreadPoolExecutor(max_workers=max_llm_workers) as ex:
					futs = {}
					for cand, row in candidate_rows:
						sec = str(row.get("section") or "")
						win = _window_for_chunk(base_window=chunk_window, section=sec, question_type=sp_qtype)
						if win > 0:
							win_rows = _get_chunk_window_rows(db, cand.chunk_id, window=win)
							chunk_text = _build_chunk_window_text(win_rows, max_chars=6000)
						else:
							chunk_text = str(row.get("chunk_text", ""))
						judge_prompt = PROMPT_CHUNK_JUDGE
						judge_prompt = judge_prompt.replace("{{SUBPROBLEM_QUESTION}}", sp.question)
						judge_prompt = judge_prompt.replace("{{CHUNK_TEXT}}", chunk_text[:6000])
						futs[ex.submit(llm.chat_json, user_prompt=judge_prompt, temperature=0.0, max_tokens=TOK_JUDGE)] = (cand, row)
					for fut in as_completed(futs):
						cand, row = futs[fut]
						try:
							obj = fut.result()
							judged.append(
								JudgedChunk(
									chunk_id=cand.chunk_id,
									classification=str(obj.get("classification", "")).strip(),
									relevance_score=_safe_float(obj.get("relevance_score"), 0.0),
									justification=str(obj.get("justification", "")).strip(),
									combined_score=float(cand.combined_score),
								)
							)
						except Exception:
							continue

				# Sort with a light weight on judge score to keep it from dominating.
				# For mechanism questions, slightly prefer theory-heavy sections (discussion/conclusion) to improve mechanism recall.
				def _sec_boost(chunk_id: str) -> float:
					if not mechanism_mode:
						return 0.0
					sec = section_by_chunk.get(str(chunk_id), "")
					return 0.08 if _section_is_theory_heavy(sec) else 0.0

				judged.sort(
					key=lambda j: (0.2 * j.relevance_score + 0.8 * j.combined_score + _sec_boost(j.chunk_id)),
					reverse=True,
				)
				strict = [j for j in judged if j.relevance_score >= 0.4][:5]
				if strict:
					judged = strict
				else:
					judged = judged[:5]
				selected_chunk_ids = [j.chunk_id for j in judged]

			# Evidence text for selected chunks
			selected_chunks: List[Dict[str, Any]] = []
			for cid in selected_chunk_ids:
				row = db.get_chunk_by_id(cid)
				if row:
					selected_chunks.append(row)

			# MODULE 5 — Evidence extractor (parallel)
			claims: List[EvidenceClaim] = []
			with ThreadPoolExecutor(max_workers=max_llm_workers) as ex:
				futs2 = {}
				for ch in selected_chunks:
					sec = str(ch.get("section") or "")
					win = _window_for_chunk(base_window=chunk_window, section=sec, question_type=sp_qtype)
					if win > 0:
						win_rows = _get_chunk_window_rows(db, str(ch.get("chunk_id")), window=win)
						chunk_text = _build_chunk_window_text(win_rows, max_chars=chunk_text_limit)
					else:
						chunk_text = str(ch.get("chunk_text", ""))
					prompt = PROMPT_EVIDENCE_EXTRACTOR
					prompt = prompt.replace("{{SUBPROBLEM_QUESTION}}", sp.question)
					prompt = prompt.replace("{{QUESTION_TYPE}}", sp_qtype)
					prompt = prompt.replace("{{CHUNK_TEXT}}", chunk_text[:chunk_text_limit])
					futs2[ex.submit(llm.chat_json, user_prompt=prompt, temperature=0.0, max_tokens=TOK_EXTRACT)] = str(ch.get("chunk_id"))
				for fut in as_completed(futs2):
					source_chunk_id = futs2[fut]
					try:
						obj = fut.result()
						for i, cl in enumerate(obj.get("claims") or [], start=1):
							claim_id = f"{sp.id}:{source_chunk_id}:claim_{i}"
							claims.append(
								EvidenceClaim(
									claim_id=claim_id,
									claim=str(cl.get("claim", "")).strip(),
									mechanism=str(cl.get("mechanism", "")).strip(),
									conditions=str(cl.get("conditions", "")).strip(),
									confidence=_safe_float(cl.get("confidence"), 0.0),
									source_chunk_id=source_chunk_id,
								)
							)
					except Exception:
						continue

			# If extraction produced no claims, do a single retry on the best chunk.
			if not claims and selected_chunks:
				best_chunk = selected_chunks[0]
				sec = str(best_chunk.get("section") or "")
				win = _window_for_chunk(base_window=chunk_window, section=sec, question_type=sp_qtype)
				if win > 0:
					win_rows = _get_chunk_window_rows(db, str(best_chunk.get("chunk_id")), window=win)
					chunk_text = _build_chunk_window_text(win_rows, max_chars=chunk_text_limit)
				else:
					chunk_text = str(best_chunk.get("chunk_text", ""))
				prompt = PROMPT_EVIDENCE_EXTRACTOR
				prompt = prompt.replace("{{SUBPROBLEM_QUESTION}}", sp.question)
				prompt = prompt.replace("{{QUESTION_TYPE}}", sp_qtype)
				prompt = prompt.replace("{{CHUNK_TEXT}}", chunk_text[:chunk_text_limit])
				try:
					obj = llm.chat_json(user_prompt=prompt, temperature=0.0, max_tokens=max(TOK_EXTRACT, 900))
					for i, cl in enumerate(obj.get("claims") or [], start=1):
						source_chunk_id = str(best_chunk.get("chunk_id"))
						claim_id = f"{sp.id}:{source_chunk_id}:claim_{i}"
						claims.append(
							EvidenceClaim(
								claim_id=claim_id,
								claim=str(cl.get("claim", "")).strip(),
								mechanism=str(cl.get("mechanism", "")).strip(),
								conditions=str(cl.get("conditions", "")).strip(),
								confidence=_safe_float(cl.get("confidence"), 0.0),
								source_chunk_id=source_chunk_id,
							)
						)
				except Exception:
					pass

			# De-duplicate claims (local + global)
			claims = global_claims.ingest(claims)
			claims_json = _format_claims_for_llm(claims)

			# MODULE 5.5 — Controlled mechanism inference (optional helper for explanatory questions)
			# This does NOT add new evidence; it produces explicitly labeled hypotheses
			# backed by at least one claim (multi-claim support emerges during aggregation).
			inferred_block = ""
			inferred_mechanisms: List[Dict[str, Any]] = []
			if mechanism_mode:
				try:
					inf_prompt = PROMPT_MECHANISM_INFERENCE
					inf_prompt = inf_prompt.replace("{{SUBPROBLEM_QUESTION}}", sp.question)
					inf_prompt = inf_prompt.replace("{{STRUCTURED_CLAIMS}}", claims_json)
					inf = llm.chat_json(user_prompt=inf_prompt, temperature=0.0, max_tokens=280 if fast_mode else 420)
					inferred = inf.get("inferred_mechanisms") or []
					# Keep payload small
					if isinstance(inferred, list) and inferred:
						inferred_mechanisms = [x for x in inferred if isinstance(x, dict)][:6]
						inferred_block = "\n\nInferred mechanisms (JSON; controlled inference; NOT additional evidence):\n" + json.dumps(
							{"inferred_mechanisms": inferred_mechanisms}, ensure_ascii=False
						)
				except Exception:
					inferred_block = ""
					inferred_mechanisms = []

			# MODULE 5.6 — Mechanism aggregation across claims (cluster + promote repeated patterns)
			mechanism_agg_block = ""
			mechanism_clusters: List[Dict[str, Any]] = []
			if mechanism_mode:
				try:
					ag_prompt = PROMPT_MECHANISM_AGGREGATION
					ag_prompt = ag_prompt.replace("{{SUBPROBLEM_QUESTION}}", sp.question)
					ag_prompt = ag_prompt.replace("{{STRUCTURED_CLAIMS}}", claims_json)
					agg = llm.chat_json(user_prompt=ag_prompt, temperature=0.0, max_tokens=320 if fast_mode else 520)
					if isinstance(agg, dict) and (agg.get("clusters") or agg.get("unclustered_claim_ids") is not None):
						# Keep payload small
						clusters = agg.get("clusters") or []
						if isinstance(clusters, list) and len(clusters) > 6:
							clusters = clusters[:6]
						mechanism_clusters = [_normalize_cluster_strength(x) for x in clusters if isinstance(x, dict)]
						mechanism_agg_block = "\n\nMechanism clusters (JSON; derived from evidence claims; cluster_strength indicates support):\n" + json.dumps(
							{
								"clusters": mechanism_clusters,
								"unclustered_claim_ids": agg.get("unclustered_claim_ids") or [],
							},
							ensure_ascii=False,
						)
				except Exception:
					mechanism_agg_block = ""
					mechanism_clusters = []

			# Coverage check from claims (FACT-only)
			coverage_obj: Dict[str, Any] = {"required_slots": list(slots or []), "supported_slots": [], "missing_slots": [], "notes": []}
			coverage_hint = ""
			if slots:
				cov_prompt = PROMPT_SLOT_COVERAGE
				cov_prompt = cov_prompt.replace("{{REQUIRED_SLOTS}}", json.dumps(slots, ensure_ascii=False))
				cov_prompt = cov_prompt.replace("{{STRUCTURED_CLAIMS}}", claims_json)
				cov_prompt = cov_prompt + "\n\nRun summary (JSON):\n" + _run_summary_json()
				try:
					co = llm.chat_json(user_prompt=cov_prompt, temperature=0.0, max_tokens=260 if fast_mode else 380)
					coverage_obj = {
						"required_slots": slots,
						"supported_slots": [str(x) for x in (co.get("supported_slots") or []) if str(x).strip()],
						"missing_slots": [str(x) for x in (co.get("missing_slots") or []) if str(x).strip()],
						"notes": [str(x) for x in (co.get("notes") or []) if str(x).strip()],
					}
				except Exception:
					pass
				coverage_hint = "\n\nCoverage (JSON):\n" + json.dumps(coverage_obj, ensure_ascii=False)

			# Early citation seeding (proactive): if we need mechanisms but extracted claims look observational,
			# follow citations before we burn iterations on isolated evidence.
			if (
				(not fast_mode)
				and mechanism_mode
				and iteration < max_iterations
				and citation_priority_ids is None
				and _claims_look_observational(claims)
			):
				try:
					chunk_for_citations = selected_chunks[0] if selected_chunks else None
					citation_list_str = "No citations available."
					if chunk_for_citations and chunk_for_citations.get("chunk_id"):
						citation_list_str = citation_db.get_citation_context_for_ai(str(chunk_for_citations["chunk_id"]))
					nav_prompt = PROMPT_CITATION_NAV
					nav_prompt = nav_prompt.replace(
						"{{FAILURE_REASONS}}",
						"Need mechanism/theory evidence; current extracted mechanisms look sparse/observational. Follow cited theory papers.",
					)
					nav_prompt = nav_prompt.replace("{{CITATION_LIST}}", citation_list_str)
					dec = llm.chat_json(user_prompt=nav_prompt, temperature=0.0, max_tokens=max(220, TOK_NAV))
					if bool(dec.get("need_more_papers")):
						nums = _extract_citation_numbers(dec.get("target_citations") or [])
						if nums and chunk_for_citations and chunk_for_citations.get("chunk_id"):
							res = citation_db.get_vector_ids_for_citations(str(chunk_for_citations["chunk_id"]), nums)
							ids = res.get("vector_ids") or []
							if ids:
								citation_priority_ids = list(dict.fromkeys([str(x) for x in ids]))
								for n in nums:
									if n not in citations_followed:
										citations_followed.append(n)
								# Try again with citation restriction
								prev_query = str(search_query)
								prev_failures = ["Proactively seeded citations for mechanism/theory"]
								prev_claim_snippets = [
									_truncate_text(str(c.claim), 180) for c in claims[:6] if str(c.claim).strip()
								]
								continue
				except Exception:
					pass

			# Hypothesis is intentionally NOT provided to extraction to avoid biasing claim extraction.
			extractor_hypo = ""

			# MODULE 6 — Local solution proposer
			prop_prompt = PROMPT_SOLUTION_PROPOSER
			prop_prompt = prop_prompt.replace("{{SUBPROBLEM_QUESTION}}", sp.question)
			# Provide the draft answer in parallel (NOT evidence) so the proposer can use it for structure,
			# but it must still cite claim IDs from evidence.
			prop_prompt = prop_prompt.replace(
				"{{STRUCTURED_CLAIMS}}",
				claims_json + coverage_hint + mechanism_agg_block + inferred_block + "\n" + hypo_block,
			)
			try:
				if use_hypothesis:
					proposal = llm.chat_json_custom_system(
						system_prompt=GLOBAL_RULES_SYSTEM_BALANCED,
						user_prompt=prop_prompt,
						temperature=0.0,
						max_tokens=TOK_PROPOSE,
					)
				else:
					# In lenient mode, allow the model to be less constrained (still JSON-only).
					if lenient_policies:
						prop_prompt = PROMPT_SOLUTION_PROPOSER_LENIENT.replace("{{SUBPROBLEM_QUESTION}}", sp.question).replace(
							"{{STRUCTURED_CLAIMS}}",
							claims_json + coverage_hint + mechanism_agg_block + inferred_block + "\n" + hypo_block,
						)
						proposal = llm.chat_json_custom_system(
							system_prompt=GLOBAL_RULES_SYSTEM_LENIENT,
							user_prompt=prop_prompt,
							temperature=0.2,
							max_tokens=TOK_PROPOSE,
						)
					else:
						proposal = llm.chat_json(user_prompt=prop_prompt, temperature=0.0, max_tokens=TOK_PROPOSE)
				proposed_solution = _normalize_proposed_solution(_coerce_solution_text(proposal.get("proposed_solution", "")).strip())
				raw_used = [str(x) for x in (proposal.get("used_claims") or [])]
				used_claims = _normalize_used_claims(raw_used, claims)
				used_claims = global_claims.canonicalize_ids(used_claims)
			except Exception as e:
				sol = SubProblemSolution(
					subproblem_id=sp.id,
					subproblem_question=sp.question,
					proposed_solution="INSUFFICIENT",
					used_claims=[],
					status="FAILED",
					confidence=0.0,
					failure_reasons=[f"LLM error (proposer): {e}"],
					selected_chunk_ids=selected_chunk_ids,
					citations_followed=list(citations_followed),
				)
				best_solution = sol
				break

			# MODULE 7 — Validator
			val_prompt = PROMPT_VALIDATOR_LENIENT if lenient_policies else PROMPT_VALIDATOR
			val_prompt = val_prompt.replace("{{CONFIDENCE_TARGET}}", f"{confidence_target:.2f}")
			val_prompt = val_prompt.replace("{{USED_CLAIMS}}", json.dumps(used_claims, ensure_ascii=False))
			val_prompt = val_prompt.replace("{{PROPOSED_SOLUTION}}", proposed_solution)
			val_prompt = val_prompt.replace(
				"{{STRUCTURED_CLAIMS}}", claims_json + coverage_hint + mechanism_agg_block + inferred_block
			)
			try:
				if lenient_policies:
					val = llm.chat_json_custom_system(
						system_prompt=GLOBAL_RULES_SYSTEM_LENIENT,
						user_prompt=val_prompt,
						temperature=0.0,
						max_tokens=TOK_VALIDATE,
					)
				else:
					val = llm.chat_json(user_prompt=val_prompt, temperature=0.0, max_tokens=TOK_VALIDATE)
			except Exception as e:
				sol = SubProblemSolution(
					subproblem_id=sp.id,
					subproblem_question=sp.question,
					proposed_solution=proposed_solution,
					used_claims=used_claims,
					status="FAILED",
					confidence=0.0,
					failure_reasons=[f"LLM error (validator): {e}"],
					selected_chunk_ids=selected_chunk_ids,
					citations_followed=list(citations_followed),
				)
				best_solution = sol
				break
			status = str(val.get("status", "")).strip().upper() or "FAILED"
			conf = max(0.0, min(1.0, _safe_float(val.get("confidence"), 0.0)))
			failure_reasons = [str(r) for r in (val.get("failure_reasons") or [])]

			# Support-based confidence adjustment (multi-paper + convergent mechanisms)
			try:
				conf = _support_adjust_confidence(
					base_conf=conf,
					used_claim_ids=used_claims,
					claims=claims,
					mechanism_clusters=mechanism_clusters,
				)
			except Exception:
				pass

			# Optional but important: if all used claims are low-confidence, cap the validator confidence.
			try:
				conf2, note = _cap_confidence_if_all_used_claims_low(
					base_conf=conf,
					used_claim_ids=used_claims,
					claims=claims,
					low_threshold=0.4,
					cap=0.6,
				)
				if note and conf2 < conf:
					conf = conf2
					# Preserve ACCEPTED but lower confidence; if gating is in effect this may become best-effort.
					failure_reasons = list(failure_reasons) + [note]
			except Exception:
				pass

			# Slot coverage gating is strict-mode only (lenient mode should not reject on missing slots).
			if strict_policies:
				try:
					req_slots = list(slots or [])
					sup_slots = list((coverage_obj or {}).get("supported_slots") or [])
					inferred_texts: List[str] = []
					inferred_texts.extend(
						[str(m.get("mechanism", "")) for m in (inferred_mechanisms or []) if isinstance(m, dict)]
					)
					inferred_texts.extend(
						[str(c.get("canonical_mechanism", "")) for c in (mechanism_clusters or []) if isinstance(c, dict)]
					)
					if "INFERRED_MECHANISM" in str(proposed_solution or ""):
						inferred_texts.append(str(proposed_solution))
					ratio, eff_supported, eff_missing = _effective_slot_coverage(
						required_slots=req_slots,
						supported_slots=sup_slots,
						inferred_texts=inferred_texts,
					)
					# Slot schema is heuristic; only gate FACT questions.
					if sp_qtype == "FACT" and req_slots and ratio < 0.70 and status == "ACCEPTED":
						status = "PARTIAL"
						conf = min(conf, 0.65)
						failure_reasons = list(failure_reasons) + [
							f"Slot coverage below threshold (FACT gating): effective_coverage={ratio:.2f} (<0.70); supported={len(eff_supported)}/{len(req_slots)}; missing={eff_missing}",
						]
				except Exception:
					pass

			# Carry forward evidence + gaps to the next iteration for evidence-driven retrieval.
			prev_query = str(search_query)
			prev_failures = list(failure_reasons)
			prev_claim_snippets = [
				_truncate_text(str(c.claim), 180)
				for c in sorted(claims, key=lambda x: float(x.confidence), reverse=True)[:12]
				if str(c.claim).strip()
			]

			# Guard: don't treat a pure "INSUFFICIENT" as an accepted contribution when we have extracted claims.
			if proposed_solution.strip().upper() == "INSUFFICIENT":
				used_claims = []
				if claims and status == "ACCEPTED":
					status = "PARTIAL"
					conf = min(conf, 0.45)
					failure_reasons = list(failure_reasons) + [
						"Proposed solution was INSUFFICIENT despite extracted evidence claims."
					]

			sol = SubProblemSolution(
				subproblem_id=sp.id,
				subproblem_question=sp.question,
				proposed_solution=proposed_solution,
				used_claims=used_claims,
				status=status,
				confidence=conf,
				failure_reasons=failure_reasons,
				selected_chunk_ids=selected_chunk_ids,
				citations_followed=list(citations_followed),
			)

			# Track best-effort attempt (even if FAILED) to allow a grounded answer later.
			def _rank_status(st: str) -> int:
				st = (st or "").upper()
				if st == "ACCEPTED":
					return 3
				if st == "PARTIAL":
					return 2
				return 1
			if (
				best_solution is None
				or _rank_status(sol.status) > _rank_status(best_solution.status)
				or (_rank_status(sol.status) == _rank_status(best_solution.status) and sol.confidence > best_solution.confidence)
			):
				best_solution = sol
				best_solution_claims = list(claims)

			graph_store.append(
				{
					"refined_problem": asdict(refined),
					"subproblem": asdict(sp),
					"dependency_context": dep_ctx,
					"iteration": iteration,
					"search_query": search_query,
					"selected_chunks": selected_chunk_ids,
					"claims": [asdict(c) for c in claims],
					"solution": asdict(sol),
					"citation_priority_ids": citation_priority_ids or [],
				},
				status="partial" if status != "FAILED" else "failed",
			)

			print(f"\n[{sp.id}] iter {iteration}/{max_iterations} | status={status} | conf={conf:.3f} | chunks={len(selected_chunk_ids)}")

			# Decide acceptance.
			# Do not gate acceptance on confidence; avoid over-iteration.
			if status == "ACCEPTED":
				break

			# MODULE 8 — Citation navigator (if failed/partial)
			if fast_mode:
				# In fast mode we skip citation-hops to keep latency low.
				citation_priority_ids = None
				best_solution = sol
				continue
			# Provide citation list from the best available chunk (first selected), else from any retrieved chunk.
			chunk_for_citations: Optional[Dict[str, Any]] = None
			if selected_chunks:
				chunk_for_citations = selected_chunks[0]
			elif candidate_rows:
				chunk_for_citations = candidate_rows[0][1]

			citation_list_str = "No citations available."
			if chunk_for_citations and chunk_for_citations.get("chunk_id"):
				try:
					citation_list_str = citation_db.get_citation_context_for_ai(str(chunk_for_citations["chunk_id"]))
				except Exception:
					citation_list_str = "No citations available."

			nav_prompt = PROMPT_CITATION_NAV
			gap_context = {
				"status": status,
				"failure_reasons": failure_reasons,
				"coverage_missing_slots": (coverage_obj.get("missing_slots") if isinstance(coverage_obj, dict) else []) or [],
				"last_search_query": search_query,
				"top_claim_snippets": [
					_truncate_text(str(c.claim), 160)
					for c in sorted(claims, key=lambda x: float(x.confidence), reverse=True)[:6]
					if str(c.claim).strip()
				],
			}
			nav_prompt = nav_prompt.replace("{{FAILURE_REASONS}}", json.dumps(gap_context, ensure_ascii=False))
			nav_prompt = nav_prompt.replace("{{CITATION_LIST}}", citation_list_str)
			nav_prompt = nav_prompt + "\n\nRun summary (JSON):\n" + _run_summary_json()
			try:
				nav = llm.chat_json(user_prompt=nav_prompt, temperature=0.0, max_tokens=TOK_NAV)
			except Exception:
				citation_priority_ids = None
				best_solution = sol
				continue
			need_more = bool(nav.get("need_more_papers"))
			target_nums = _extract_citation_numbers(nav.get("target_citations") or [])

			if need_more and chunk_for_citations and target_nums:
				# Map citations -> chunk ids of cited papers
				res = citation_db.get_vector_ids_for_citations(str(chunk_for_citations["chunk_id"]), target_nums)
				ids = [str(x) for x in (res.get("vector_ids") or [])]
				if ids:
					cids = list(dict.fromkeys(ids))
					if global_allowed_chunk_ids is not None:
						cids = [cid for cid in cids if cid in global_allowed_chunk_ids]
					citation_priority_ids = cids if cids else None
					for n in target_nums:
						if n not in citations_followed:
							citations_followed.append(n)
					print(f"Citation hop: following {target_nums} (priority chunks: {len(citation_priority_ids) if citation_priority_ids else 0})")
					best_solution = sol
					continue

			# MODULE 9 — Sub-problem expansion (depth control)
			# Avoid confidence-driven expansion; expand only when the attempt truly failed.
			if (not fast_mode) and sp.depth < max_depth and (status == "FAILED"):
				expand_prompt = PROMPT_EXPAND
				expand_prompt = expand_prompt.replace("{{SUBPROBLEM_QUESTION}}", sp.question)
				expand_prompt = expand_prompt.replace("{{FAILURE_REASONS}}", "\n".join(failure_reasons) or status)
				expand_prompt = expand_prompt + "\n\nRun summary (JSON):\n" + _run_summary_json()
				expanded = llm.chat_json(user_prompt=expand_prompt, temperature=0.0, max_tokens=TOK_EXPAND)
				new_sps = expanded.get("new_subproblems") or []
				added_any = False
				existing_ids = {x.id for x in queue}
				existing_ids.update(accepted.keys())
				existing_ids.update(failed.keys())
				for nsp in new_sps[:5]:
					qid = str(nsp.get("id", "")).strip() or f"{sp.id}.x{int(time.time())}"
					qq = str(nsp.get("question", "")).strip()
					if not qq:
						continue
					if qid in existing_ids:
						continue
					deps = [str(d) for d in (nsp.get("depends_on") or [])]
					queue.append(PlannedSubProblem(id=qid, question=qq, depends_on=deps, depth=sp.depth + 1))
					added_any = True
				if added_any:
					print(f"Expanded: added {len(new_sps[:5])} sub-problems (depth {sp.depth + 1})")
					best_solution = sol
					# If we expanded, move on; don't burn more iterations here.
					break

			# Otherwise, clear citation scope and retry
			citation_priority_ids = None
			best_solution = sol

		# Record outcome
		if best_solution:
			best_effort[sp.id] = best_solution
			best_effort_claims[sp.id] = best_solution_claims
		if best_solution and best_solution.status in {"ACCEPTED", "PARTIAL"} and list(best_solution.used_claims or []):
			accepted[sp.id] = best_solution
		else:
			final = best_solution or SubProblemSolution(
				subproblem_id=sp.id,
				subproblem_question=sp.question,
				proposed_solution="INSUFFICIENT",
				used_claims=[],
				status="FAILED",
				confidence=0.0,
				failure_reasons=["No successful attempt"],
				selected_chunk_ids=[],
				citations_followed=list(citations_followed),
			)
			failed[sp.id] = final

	# MODULE 10 — Final synthesizer
	if not accepted:
		# Best-effort answer: use whatever evidence was extracted, even if not accepted.
		all_claims: List[EvidenceClaim] = []
		chain: List[str] = []
		min_conf: float = 1.0
		for sid, sol in best_effort.items():
			claims = best_effort_claims.get(sid) or []
			if not claims:
				continue
			all_claims.extend(claims)
			chain.append(f"{sid} from chunks={sol.selected_chunk_ids} claims={[c.claim_id for c in claims]}")
			if sol.confidence > 0:
				min_conf = min(min_conf, float(sol.confidence))
		# If still no claims, return a non-empty response explaining the situation.
		if not all_claims:
				synth = {
					"hidden_solution": "EMERGING_MECHANISM\nNo relevant evidence claims could be extracted from retrieved chunks.",
				"why_hidden": "Retrieval returned candidates, but evidence extraction produced no usable claims.",
				"evidence_chain": [],
				"final_confidence": 0.0,
			}
		else:
			# Prefer globally deduplicated claims for synthesis.
			all_claims = list(global_claims.by_norm.values()) or all_claims
			# Cap evidence volume to avoid provider prompt-token limits.
			all_claims = sorted(all_claims, key=lambda c: float(c.confidence), reverse=True)[:28]
			# Build a strictly-grounded best-effort synthesis using only extracted claims.
			evidence_snips = _format_claim_snippets_for_llm(all_claims, max_items=24, max_chars=1800)
			best_effort_prompt = (
				"You are a scientific synthesis agent.\n"
				"Follow all GLOBAL RULES.\n\n"
				"Write a DETAILED, structured answer to the user's original question.\n"
				"Hard rules:\n"
				"- Evidence-backed statements MUST be supported by the evidence snippets and cite claim IDs.\n"
				"- If you add practical guidance beyond the evidence, put it ONLY under a section labeled MODEL_KNOWLEDGE.\n"
				"- If you cannot answer some part, say what is missing, but still summarize what IS supported.\n"
				"- Use this structure inside the answer field:\n"
				"  1) Summary (2-3 sentences)\n"
				"  2) Evidence-backed factors/causes (3-8 bullets)\n"
				"  3) Evidence-backed mitigations/actions (3-8 bullets; ONLY if explicitly supported by claims; otherwise put them in Unknowns)\n"
				"  4) Diagnostics / what to monitor (bullets; ONLY if explicitly supported by claims; otherwise put in Unknowns)\n"
				"  5) Unknowns / missing evidence (bullets)\n"
				"  6) MODEL_KNOWLEDGE (optional; practical tips; NO claim IDs)\n"
				"- EVERY sentence/bullet in sections (1)-(4) MUST include at least one claim id in parentheses.\n\n"
				"Original question:\n"
				f"{refined.refined_question}\n\n"
				"Evidence snippets:\n"
				+ evidence_snips
				+ "\n\n"
				"Output STRICT JSON:\n"
				"{\n"
				"  \"answer\": \"...\",\n"
				"  \"key_points\": [\"...\"],\n"
				"  \"used_claims\": [\"...\"]\n"
				"}"
			)
			try:
				if use_hypothesis:
					be = llm.chat_json_custom_system(
						system_prompt=GLOBAL_RULES_SYSTEM_BALANCED,
						user_prompt=best_effort_prompt,
						temperature=0.0,
						max_tokens=max(900, TOK_SYNTH),
					)
				else:
					be = llm.chat_json(user_prompt=best_effort_prompt, temperature=0.0, max_tokens=max(900, TOK_SYNTH))
				a = str(be.get("answer", "")).strip()
				uc = [str(x) for x in (be.get("used_claims") or [])]
				uc = _normalize_used_claims(uc, all_claims)
				synth = {
					"hidden_solution": "EMERGING_MECHANISM\n" + (a or "(empty)"),
					"why_hidden": "No sub-problem solutions met the acceptance threshold; returning a best-effort answer grounded in extracted claims.",
					"evidence_chain": [f"used_claims={uc}"],
					"final_confidence": max(0.35, max(0.0, min(1.0, (min_conf if min_conf < 1.0 else 0.45) * 0.75))),
				}
			except Exception:
				synth = {
					"hidden_solution": "EMERGING_MECHANISM\n" + "\n".join([c.claim for c in all_claims[:12]]),
					"why_hidden": "Could not run best-effort synthesizer; returning raw extracted claim summaries.",
					"evidence_chain": chain[:6],
					"final_confidence": max(0.0, min(1.0, (min_conf if min_conf < 1.0 else 0.4) * 0.6)),
				}
	else:
		global_mech_obj: Optional[Dict[str, Any]] = None
		global_mech_json = "null"
		# Final-stage global mechanism aggregation across ACCEPTED subproblems.
		# This helps surface mechanisms that only emerge when SP1+SP2+SP3 are combined.
		global_mechanism_mode = main_qtype == "MECHANISM" or any(
			(_classify_question_type(s.subproblem_question) == "MECHANISM") for s in accepted.values()
		)
		if global_mechanism_mode:
			try:
				acc_claims: List[EvidenceClaim] = []
				for sid in accepted.keys():
					acc_claims.extend(best_effort_claims.get(sid) or [])
				# Deduplicate mechanisms semantically (cheap) so cross-paper paraphrases don't split support.
				try:
					acc_claims = _semantic_dedup_claims_by_mechanism(acc_claims, threshold=0.90)
				except Exception:
					pass
				# Deduplicate (lexical) as a backstop to avoid overcounting support.
				_tmp = GlobalClaimStore()
				_tmp.ingest(acc_claims)
				acc_claims = list(_tmp.by_norm.values())
				acc_claims = sorted(acc_claims, key=lambda c: float(c.confidence), reverse=True)[:48]
				if acc_claims:
					g_prompt = PROMPT_GLOBAL_MECHANISM_AGGREGATION
					g_prompt = g_prompt.replace("{{REFINED_QUESTION}}", refined.refined_question)
					g_prompt = g_prompt.replace("{{STRUCTURED_CLAIMS}}", _format_claims_for_llm(acc_claims))
					gm = llm.chat_json(user_prompt=g_prompt, temperature=0.0, max_tokens=max(260, int(TOK_VALIDATE)))
					if isinstance(gm, dict) and (gm.get("clusters") or gm.get("unclustered_claim_ids") is not None):
						clusters = gm.get("clusters") or []
						if isinstance(clusters, list):
							clusters = [_normalize_cluster_strength(x) for x in clusters if isinstance(x, dict)]
						global_mech_obj = {
							"clusters": clusters,
							"unclustered_claim_ids": gm.get("unclustered_claim_ids") or [],
						}
						global_mech_json = json.dumps(global_mech_obj, ensure_ascii=False)
			except Exception:
				# Optional; don't block synthesis.
				pass

		accepted_json = _compact_accepted_for_synthesis(accepted, max_items=6, max_chars=1800)
		syn_prompt = PROMPT_SYNTHESIZE_LENIENT if lenient_policies else PROMPT_SYNTHESIZE
		syn_prompt = syn_prompt.replace("{{REFINED_QUESTION}}", refined.refined_question)
		syn_prompt = syn_prompt.replace("{{SUB_SOLUTIONS}}", accepted_json)
		syn_prompt = syn_prompt.replace("{{GLOBAL_MECHANISM_CLUSTERS}}", global_mech_json)
		# Do NOT append run summary here; it can blow up prompt tokens.
		if lenient_policies:
			synth = llm.chat_json_custom_system(
				system_prompt=GLOBAL_RULES_SYSTEM_LENIENT,
				user_prompt=syn_prompt,
				temperature=0.2,
				max_tokens=TOK_SYNTH,
			)
		elif use_hypothesis:
			synth = llm.chat_json_custom_system(
				system_prompt=GLOBAL_RULES_SYSTEM_BALANCED,
				user_prompt=syn_prompt,
				temperature=0.0,
				max_tokens=TOK_SYNTH,
			)
		else:
			synth = llm.chat_json(user_prompt=syn_prompt, temperature=0.0, max_tokens=TOK_SYNTH)

		# If the synthesizer refuses (often due to missing coverage), still provide a strictly-grounded partial answer.
		hs = str(synth.get("hidden_solution", "")).strip().upper()
		if hs in {"INSUFFICIENT_EVIDENCE", "INSUFFICIENT"}:
			parts: List[str] = []
			chain: List[str] = []
			min_conf: float = 1.0
			for s in accepted.values():
				parts.append(f"{s.subproblem_id}: {s.proposed_solution}")
				chain.append(
					f"{s.subproblem_id} used_claims={s.used_claims} from chunks={s.selected_chunk_ids}"
				)
				min_conf = min(min_conf, float(s.confidence))

			synth = {
				"hidden_solution": "PARTIAL_ANSWER\n" + "\n\n".join(parts),
				"why_hidden": "Full hidden solution could not be proven from available evidence, but the following sub-results are supported by retrieved chunks.",
				"evidence_chain": chain,
				"final_confidence": max(0.0, min(1.0, min_conf * 0.85)),
			}

	print("\n" + "=" * 90)
	print("FINAL ANSWER")
	print("=" * 90)

	# ------------------------------------------------------------
	# Final report (requested format)
	# ------------------------------------------------------------
	# Collect results for ALL processed problems, including expanded sub-sub problems.
	all_ids = set(best_effort.keys()) | set(accepted.keys()) | set(failed.keys())
	all_sorted = sorted(all_ids, key=_natural_sort_key)
	results: List[Dict[str, Any]] = []
	all_chunk_ids: List[str] = []
	all_citations_followed: List[str] = []

	for sid in all_sorted:
		sol = best_effort.get(sid) or accepted.get(sid) or failed.get(sid)
		if not sol:
			continue
		results.append(
			{
				"id": sol.subproblem_id,
				"question": sol.subproblem_question,
				"status": sol.status,
				"confidence_index": float(sol.confidence),
				"result": sol.proposed_solution,
				"used_claims": list(sol.used_claims or []),
				"selected_chunk_ids": list(sol.selected_chunk_ids or []),
				"citations_followed": list(sol.citations_followed or []),
				"failure_reasons": list(sol.failure_reasons or []),
			}
		)
		for cid in sol.selected_chunk_ids or []:
			if cid and cid not in all_chunk_ids:
				all_chunk_ids.append(str(cid))
		for c in sol.citations_followed or []:
			if c and c not in all_citations_followed:
				all_citations_followed.append(str(c))

	# Resolve papers used from chunk metadata.
	papers_used_map: Dict[str, Dict[str, Any]] = {}
	for cid in all_chunk_ids:
		row = db.get_chunk_by_id(cid)
		if not row:
			continue
		pid = row.get("paper_id")
		if not pid:
			continue
		pid = str(pid)
		if pid not in papers_used_map:
			papers_used_map[pid] = _safe_get_paper_meta(db, pid)
		# Keep a few extra fields tied to this run.
		papers_used_map[pid].setdefault("chunk_ids_used", [])
		if cid not in papers_used_map[pid]["chunk_ids_used"]:
			papers_used_map[pid]["chunk_ids_used"].append(cid)
		if row.get("year") is not None and "year" not in papers_used_map[pid]:
			papers_used_map[pid]["year"] = row.get("year")

	papers_used = list(papers_used_map.values())
	papers_used.sort(key=lambda x: _natural_sort_key(str(x.get("paper_id", ""))))

	# Build report in the requested order:
	# 1) per-subproblem results
	# 2) outcome (single final answer)
	# 3) citations used
	# 4) papers used
	# 5) rest of previous format
	import re
	outcome = str(synth.get("hidden_solution", ""))
	outcome = re.sub(r'\[.*?\]|\(.*?(chunk|claim).*?\)|\[SP\d+.*?\]|\(SP\d+.*?\)', '', outcome)
	outcome = re.sub(r'\s+', ' ', outcome).strip()

	final_report: Dict[str, Any] = {
		"subproblem_results": results,
		"outcome": outcome,
		"citations_used": all_citations_followed,
		"papers_used": papers_used,
		"why_hidden": synth.get("why_hidden"),
		"evidence_chain": synth.get("evidence_chain"),
		"final_confidence": synth.get("final_confidence"),
	}

	print(outcome)

	# Persist final
	final_conf = float(final_report.get("final_confidence") or 0.0)
	final_status = "solved"
	graph_store.append(
		{
			"refined_problem": asdict(refined),
			"accepted": [asdict(s) for s in accepted.values()],
			"failed": [asdict(s) for s in failed.values()],
			"final": final_report,
		},
		status=final_status,
	)

	# Update Semantic Cache with final solution
	try:
		prob_vec = retriever.model_small.encode(str(problem), convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
		with sqlite3.connect(cache_db_path) as conn:
			conn.execute(
				"INSERT OR REPLACE INTO semantic_cache (question, embedding, final_solution) VALUES (?, ?, ?)",
				(str(problem), prob_vec.tobytes(), str(outcome))
			)
			conn.commit()
	except Exception as e:
		print(f"[CACHE UPDATE ERROR] Ignored: {e}")


def main():
	parser = argparse.ArgumentParser(description="LLM-driven multi-hop RAG + citations")
	parser.add_argument("--problem", required=True, help="User problem / question")
	parser.add_argument("--confidence", type=float, default=0.75, help="Target confidence in [0,1]")
	parser.add_argument(
		"--policy",
		choices=["lenient", "strict", "balanced"],
		default="lenient",
		help="Policy strictness. lenient=reduce rejections (default). strict=maximum evidence policing. balanced=keep strict evidence but allow MODEL_KNOWLEDGE in some steps.",
	)
	parser.add_argument("--max-depth", type=int, default=2, help="Max expansion depth")
	parser.add_argument("--max-iterations", type=int, default=3, help="Max retries per subproblem")
	parser.add_argument("--max-subproblems", type=int, default=3, help="How many sub-problems to plan (fast runs: 3)")
	parser.add_argument("--top-k", type=int, default=8, help="How many candidates to judge")
	parser.add_argument("--prefetch-k", type=int, default=24, help="Qdrant prefetch per vector")
	parser.add_argument(
		"--no-global-paper-shortlist",
		action="store_true",
		help="Disable the global top-paper shortlist and allow per-subproblem retrieval over the entire corpus.",
	)
	parser.add_argument("--global-top-papers", type=int, default=12, help="Global number of papers to keep for all subproblems")
	parser.add_argument("--global-paper-pool", type=int, default=24, help="How many candidate papers to consider for the global shortlist")
	parser.add_argument("--global-candidate-chunks", type=int, default=160, help="How many retrieved chunks to form the global paper candidate pool")
	parser.add_argument("--global-snippets-per-paper", type=int, default=2, help="Snippets per paper shown to the global selector")
	parser.add_argument("--seed-chunk-id", default=None, help="Optional seed chunk id to bias citations")
	parser.add_argument(
		"--fast",
		action="store_true",
		help="DEPRECATED alias for --fast-verification.",
	)
	parser.add_argument(
		"--fast-verification",
		action="store_true",
		help="Verification mode: fewer tokens, no citation hops/expansion. Not valid for mechanism discovery (will error).",
	)
	parser.add_argument(
		"--use-hypothesis",
		action="store_true",
		help="Optional: generate per-subproblem hypotheses (uses model prior knowledge) to improve retrieval/extraction only. Note: may indirectly bias retrieval via query rewriting/keywords.",
	)
	parser.add_argument("--llm-base-url", default=LLM_BASE_URL_DEFAULT)
	parser.add_argument("--llm-api-key", default=LLM_API_KEY_DEFAULT)
	parser.add_argument("--llm-model", default=LLM_MODEL_DEFAULT)
	parser.add_argument("--llm-timeout", type=int, default=int(LLM_TIMEOUT_DEFAULT_S))
	parser.add_argument("--max-llm-workers", type=int, default=4)
	parser.add_argument(
		"--max-inflight-requests",
		type=int,
		default=8,
		help="Global cap on concurrent LLM requests across all thread pools/retries (prevents burst overload).",
	)
	args = parser.parse_args()
	# Backwards compatible alias.
	args.fast = bool(args.fast or args.fast_verification)

	if args.fast:
		# Override for speed
		args.max_depth = 0
		args.max_iterations = 1
		args.top_k = 6
		args.prefetch_k = 18
		args.global_top_papers = min(int(getattr(args, "global_top_papers", 12)), 10)
		args.global_paper_pool = min(int(getattr(args, "global_paper_pool", 24)), 18)
		args.global_candidate_chunks = min(int(getattr(args, "global_candidate_chunks", 160)), 90)
		args.max_llm_workers = min(int(args.max_llm_workers), 2)
		args.llm_timeout = min(int(args.llm_timeout), 25)
		# Use a smaller model by default for speed, unless user explicitly set another one.
		if str(args.llm_model).strip() == str(LLM_MODEL_DEFAULT).strip():
			args.llm_model = LLM_MODEL_FAST_DEFAULT

	# Global concurrency cap across nested ThreadPoolExecutors + retry loops.
	_set_chat_semaphore(max(0, int(args.max_inflight_requests)))

	conf = max(0.0, min(1.0, float(args.confidence)))
	llm = LLMClient(
		base_url=str(args.llm_base_url),
		api_key=str(args.llm_api_key),
		model=str(args.llm_model),
		timeout_s=int(args.llm_timeout),
	)

	run(
		problem=str(args.problem),
		confidence_target=conf,
		max_depth=int(args.max_depth),
		max_iterations=int(args.max_iterations),
		top_k=int(args.top_k),
		prefetch_k=int(args.prefetch_k),
		llm=llm,
		max_llm_workers=max(1, int(args.max_llm_workers)),
		max_subproblems=max(1, int(args.max_subproblems)),
		fast_mode=bool(args.fast),
		use_hypothesis=bool(args.use_hypothesis),
		seed_chunk_id=str(args.seed_chunk_id) if args.seed_chunk_id else None,
		policy=str(args.policy),
		use_global_paper_shortlist=not bool(args.no_global_paper_shortlist),
		global_top_papers=int(args.global_top_papers),
		global_paper_pool=int(args.global_paper_pool),
		global_candidate_chunks=int(args.global_candidate_chunks),
		global_snippets_per_paper=int(args.global_snippets_per_paper),
	)


if __name__ == "__main__":
	main()


