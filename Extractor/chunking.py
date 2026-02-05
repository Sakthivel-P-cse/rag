"""Chunking utilities and CLI.

Produces chunk JSON files used for Postgres loading. Output is intentionally minimal:
- chunk_id, paper_id, section, paragraph_index, text, year
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    import tiktoken

    def make_tiktoken_tokenizer(model_name: str = "gpt-4o-mini"):
        enc = tiktoken.encoding_for_model(model_name)
        return {
            "encode": lambda s: enc.encode(s),
            "decode": lambda tokens: enc.decode(tokens),
            "count": lambda s: len(enc.encode(s)),
        }


except Exception:
    tiktoken = None


def default_tokenizer():
    if tiktoken:
        return make_tiktoken_tokenizer()
    return {
        "encode": lambda s: s.split(),
        "decode": lambda tokens: " ".join(tokens),
        "count": lambda s: max(1, len(str(s or "").split())),
    }


def chunk_paragraphs(
    paragraphs: List[Dict[str, Any]],
    *,
    tokenizer: Optional[Dict] = None,
    min_tokens: int = 400,
    max_tokens: int = 1200,
    overlap_percent: float = 0.30,
) -> List[Dict[str, Any]]:
    """Combine consecutive paragraphs into bounded-size chunks.

    paragraphs items should include:
      - text (str)
      - section_heading (str) optional
      - paper_id (str) optional
      - year (int) optional

    Returns chunk dicts containing:
      - chunk_id, paper_id, section, paragraph_index, text, year, token_count
    """
    if tokenizer is None:
        tokenizer = default_tokenizer()

    assert 0 < min_tokens <= max_tokens
    assert 0.0 < overlap_percent <= 1.0

    all_chunks = []
    i = 0
    while i < len(paragraphs):
        current_tokens = 0
        chunk_paras = []
        start_para_idx = i
        current_section = paragraphs[i].get("section_heading")

        while i < len(paragraphs) and current_tokens < min_tokens:
            para = paragraphs[i]
            para_text = str(para.get("text", "")).strip()
            if not para_text:
                i += 1
                continue

            para_tokens = int(tokenizer["count"](para_text))
            if current_tokens > 0 and current_tokens + para_tokens > max_tokens:
                break

            if current_section and para.get("section_heading") != current_section:
                if current_tokens >= min_tokens:
                    break
                current_section = para.get("section_heading")

            chunk_paras.append(para)
            current_tokens += para_tokens
            i += 1

        if current_tokens < min_tokens and i < len(paragraphs):
            continue

        if chunk_paras:
            chunk_text = "\n\n".join([str(p.get("text", "")).strip() for p in chunk_paras if str(p.get("text", "")).strip()])
            chunk_id = f"{chunk_paras[0].get('paper_id', 'paper')}_chunk-{len(all_chunks) + 1}"
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": chunk_paras[0].get("paper_id"),
                    "section": chunk_paras[0].get("section_heading"),
                    "paragraph_index": start_para_idx,
                    "text": chunk_text,
                    "year": chunk_paras[0].get("year"),
                    "token_count": current_tokens,
                }
            )

    return all_chunks


def extract_year_from_text(text: str) -> Optional[int]:
    m = re.search(r"\b(19|20)\d{2}\b", str(text or "")[:500])
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def _parse_markdownish_sections(content: str, *, paper_id: str, year: Optional[int]) -> List[Dict[str, Any]]:
    paragraphs_raw = [p.strip() for p in str(content or "").split("\n\n") if p.strip()]
    current_section_heading: Optional[str] = None

    paragraphs: List[Dict[str, Any]] = []
    for idx, para_text in enumerate(paragraphs_raw):
        if para_text.startswith("##"):
            heading_match = re.match(r"^##\s*(\d+(?:\.\d+)*)\s+(.+)$", para_text)
            if heading_match:
                current_section_heading = heading_match.group(2).strip()
            else:
                current_section_heading = para_text.replace("##", "").strip() or None
            continue
        if para_text.startswith("#"):
            continue

        paragraphs.append(
            {
                "text": para_text,
                "section_heading": current_section_heading,
                "paper_id": paper_id,
                "paragraph_id": f"para_{idx}",
                "year": year,
            }
        )

    return paragraphs


def run(
    *,
    workspace_root: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    min_tokens: int = 400,
    max_tokens: int = 1200,
    overlap_percent: float = 0.30,
) -> None:
    """Chunk all .txt files from input_dir and write *_chunks.json to output_dir."""
    import json

    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[1]
    if input_dir is None:
        input_dir = workspace_root / "OUTPUT" / "text" / "main"
    if output_dir is None:
        output_dir = workspace_root / "OUTPUT" / "Chunked_text"

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = default_tokenizer()

    for txt_file in input_dir.rglob("*.txt"):
        try:
            content = txt_file.read_text(encoding="utf-8")
            doc_year = extract_year_from_text(content)
            paragraphs = _parse_markdownish_sections(content, paper_id=txt_file.stem, year=doc_year)

            chunks = chunk_paragraphs(
                paragraphs,
                tokenizer=tokenizer,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                overlap_percent=overlap_percent,
            )

            # Strip internal-only fields (e.g., token_count) for storage
            out_chunks = []
            for c in chunks:
                out_chunks.append(
                    {
                        "chunk_id": c.get("chunk_id"),
                        "paper_id": c.get("paper_id"),
                        "section": c.get("section"),
                        "paragraph_index": c.get("paragraph_index"),
                        "text": c.get("text"),
                        "year": c.get("year"),
                    }
                )

            rel = txt_file.relative_to(input_dir)
            out_parent = output_dir / rel.parent
            out_parent.mkdir(parents=True, exist_ok=True)
            output_file = out_parent / f"{txt_file.stem}_chunks.json"
            output_file.write_text(json.dumps(out_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Written: {output_file} ({len(out_chunks)} chunks)")
        except Exception as e:
            print(f"Failed to process {txt_file}: {e}")


if __name__ == "__main__":
    run()
