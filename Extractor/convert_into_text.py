import re
import textwrap
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

from rag_utils.metrics import stage_timer


NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def clean_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def node_text(el) -> str:
    if el is None:
        return ""
    return clean_whitespace("".join(el.itertext()))


def wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(text, width)) if text else ""


def parse_header(root):
    header = root.find("tei:teiHeader", NS)
    if header is None:
        return {"title": "", "pub_date": "", "abstract_paras": []}

    title_el = header.find(".//tei:titleStmt/tei:title[@type='main']", NS)
    title = node_text(title_el)

    date_el = header.find(".//tei:publicationStmt/tei:date", NS)
    pub_date = node_text(date_el)

    abstract_el = header.find(".//tei:profileDesc/tei:abstract", NS)
    abstract_paras = []
    if abstract_el is not None:
        for p in abstract_el.findall(".//tei:p", NS):
            para = node_text(p)
            if para:
                abstract_paras.append(para)

    return {"title": title, "pub_date": pub_date, "abstract_paras": abstract_paras}


def parse_body(root):
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return []

    sections = []
    for div in body.findall("tei:div", NS):
        head = div.find("tei:head", NS)
        heading = node_text(head)
        sec_num = head.get("n") if head is not None else None

        paragraphs = []
        for p in div.findall("tei:p", NS):
            para = node_text(p)
            if para:
                paragraphs.append(para)

        figures = []
        for fig in div.findall("tei:figure", NS):
            label = node_text(fig.find("tei:label", NS))
            fig_head = node_text(fig.find("tei:head", NS))
            desc = node_text(fig.find("tei:figDesc", NS))
            figures.append({"label": label, "head": fig_head, "desc": desc})

        sections.append({"number": sec_num, "heading": heading, "paragraphs": paragraphs, "figures": figures})

    return sections


def parse_notes(root):
    notes = []
    for note in root.findall(".//tei:note[@place='foot']", NS):
        n = note.get("n")
        note_id = note.get("{http://www.w3.org/XML/1998/namespace}id")
        text = node_text(note)
        notes.append({"id": note_id, "n": n, "text": text})
    return notes


def tei_to_markdown(tei_path: Path) -> str:
    tree = ET.parse(tei_path)
    root = tree.getroot()

    header = parse_header(root)
    sections = parse_body(root)
    notes = parse_notes(root)

    lines = []
    if header["title"]:
        lines.append(f"# {header['title']}")
        lines.append("")

    if header["pub_date"]:
        lines.append(f"**Published:** {header['pub_date']}")
        lines.append("")

    if header["abstract_paras"]:
        lines.append("## Abstract")
        lines.append("")
        for para in header["abstract_paras"]:
            lines.append(wrap(para))
            lines.append("")

    for sec in sections:
        if sec["heading"]:
            if sec["number"]:
                lines.append(f"## {sec['number']} {sec['heading']}")
            else:
                lines.append(f"## {sec['heading']}")
            lines.append("")

        for para in sec["paragraphs"]:
            lines.append(wrap(para))
            lines.append("")

        for fig in sec["figures"]:
            cap_parts = []
            if fig["label"]:
                cap_parts.append(fig["label"])
            if fig["head"]:
                cap_parts.append(fig["head"])
            if fig["desc"]:
                cap_parts.append(fig["desc"])
            caption = " — ".join(cap_parts)
            if caption:
                lines.append(f"_Figure_: {wrap(caption)}")
                lines.append("")

    if notes:
        lines.append("---")
        lines.append("## Footnotes")
        lines.append("")
        for note in notes:
            prefix = note["n"] if note["n"] else note["id"]
            lines.append(f"[{prefix}] {wrap(note['text'])}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def convert_folder(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for p in input_dir.glob("**/*") if p.is_file()]
    with stage_timer("doc_loading", extra={"step": "tei_to_markdown", "num_docs": len(paths)}):
        for path in paths:
            if path.suffix.lower() not in {".xml", ".tei"} and not path.name.endswith(".tei.xml"):
                continue

            try:
                text = tei_to_markdown(path)
            except Exception as e:
                print(f"Failed to parse {path}: {e}")
                continue

            rel = path.relative_to(input_dir)
            out_file = (output_dir / rel).with_suffix(".txt")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(text, encoding="utf-8")
            print(f"Written: {out_file}")


def run(*, workspace_root: Optional[Path] = None, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> None:
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[1]
    if input_dir is None:
        input_dir = workspace_root / "OUTPUT" / "tei_main"
    if output_dir is None:
        output_dir = workspace_root / "OUTPUT" / "text" / "main"

    convert_folder(Path(input_dir), Path(output_dir))


if __name__ == "__main__":
    run()
