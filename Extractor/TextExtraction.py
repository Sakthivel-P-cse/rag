import re
from pathlib import Path
from typing import Optional

import requests


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_text_from_tei(tei: str) -> str:
    tei = re.sub(r'xmlns(:\w+)?="[^"]*"', '', tei)
    paras = re.findall(r"<p[^>]*>(.*?)</p>", tei, flags=re.DOTALL)
    out = []
    for p in paras:
        p = re.sub(r"<[^>]+>", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return "\n\n".join(out)


def process_pdf(pdf: Path, *, grobid_server: str, timeout_s: int = 120) -> bytes:
    url = grobid_server.rstrip("/") + "/api/processFulltextDocument"
    files = {"input": (pdf.name, pdf.open("rb"), "application/pdf")}
    r = requests.post(url, files=files, timeout=timeout_s)
    r.raise_for_status()
    return r.content


def run(
    *,
    workspace_root: Optional[Path] = None,
    grobid_server: str = "http://localhost:8070",
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Extract TEI and raw text from PDFs using a running GROBID service."""
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[1]

    if input_dir is None:
        input_dir = workspace_root / "Research Papers"
    if output_dir is None:
        output_dir = workspace_root / "OUTPUT"

    tei_dir = output_dir / "tei"
    text_dir = output_dir / "text"
    ensure_dir(tei_dir)
    ensure_dir(text_dir)

    pdfs = sorted(input_dir.glob("**/*.pdf"))
    if not pdfs:
        print(f"No PDFs in {input_dir}")
        return

    for pdf in pdfs:
        rel = pdf.relative_to(input_dir)
        tei_out = tei_dir / rel.with_suffix(".tei.xml")
        text_out = text_dir / rel.with_suffix(".txt")
        ensure_dir(tei_out.parent)
        ensure_dir(text_out.parent)

        try:
            tei_bytes = process_pdf(pdf, grobid_server=grobid_server)
            tei_out.write_bytes(tei_bytes)
            text = extract_text_from_tei(tei_bytes.decode("utf-8", errors="ignore"))
            text_out.write_text(text, encoding="utf-8")
            print(f"OK: {rel}")
        except Exception as e:
            print(f"FAIL: {rel} -> {e}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
