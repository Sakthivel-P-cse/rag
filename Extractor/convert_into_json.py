import json
import re
import sys
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET


NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def clean_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def node_text(el) -> str:
    if el is None:
        return ""
    return clean_whitespace("".join(el.itertext()))


def parse_paper_metadata(root):
    header = root.find("tei:teiHeader", NS)
    if header is None:
        return {"paper_id": "UNKNOWN_ID", "title": "", "arxiv_id": None, "authors": []}

    title_el = header.find(".//tei:titleStmt/tei:title[@type='main']", NS)
    title = node_text(title_el)

    bibl = header.find(".//tei:sourceDesc/tei:biblStruct", NS)
    paper_id = None
    arxiv_id = None
    if bibl is not None:
        for idno in bibl.findall("tei:idno", NS):
            t = idno.get("type")
            val = node_text(idno)
            if t == "MD5":
                paper_id = val
            elif t == "arXiv":
                arxiv_id = val
    if not paper_id:
        paper_id = arxiv_id or title or "UNKNOWN_ID"

    authors = []
    for author in header.findall(".//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author", NS):
        pn = author.find("tei:persName", NS)
        if pn is None:
            continue
        given_parts = [node_text(fn) for fn in pn.findall("tei:forename", NS)]
        given_name = " ".join([g for g in given_parts if g]).strip() or None
        family = node_text(pn.find("tei:surname", NS)) or None
        if given_name or family:
            authors.append({"given": given_name, "family": family})

    return {"paper_id": paper_id, "title": title, "arxiv_id": arxiv_id, "authors": authors}


def parse_references(root):
    refs = []
    for bibl in root.findall(".//tei:text/tei:back/tei:listBibl/tei:biblStruct", NS):
        bibl_id = bibl.get("{http://www.w3.org/XML/1998/namespace}id")

        title_el = (
            bibl.find("tei:analytic/tei:title[@type='main']", NS)
            or bibl.find("tei:monogr/tei:title[@type='main']", NS)
            or bibl.find("tei:monogr/tei:title", NS)
        )
        title = node_text(title_el)

        container_el = (
            bibl.find("tei:monogr/tei:title[@level='j']", NS)
            or bibl.find("tei:monogr/tei:title[@level='m']", NS)
            or bibl.find("tei:monogr/tei:title", NS)
        )
        container_title = node_text(container_el)

        year = None
        date_el = bibl.find("tei:monogr/tei:imprint/tei:date", NS)
        if date_el is not None:
            year_text = node_text(date_el)
            m = re.search(r"\d{4}", year_text)
            year = m.group(0) if m else year_text

        idnos = []
        for idno in bibl.findall("tei:monogr/tei:idno", NS):
            id_type = idno.get("type") or "other"
            idnos.append({"type": id_type, "value": node_text(idno)})

        authors = []
        for author in bibl.findall(".//tei:author", NS):
            pn = author.find("tei:persName", NS)
            if pn is None:
                continue
            given_parts = [node_text(fn) for fn in pn.findall("tei:forename", NS)]
            given = " ".join([g for g in given_parts if g]) or None
            family = node_text(pn.find("tei:surname", NS)) or None
            if given or family:
                authors.append({"given": given, "family": family})

        refs.append(
            {
                "bibl_id": bibl_id,
                "title": title,
                "container_title": container_title,
                "year": year,
                "idnos": idnos,
                "authors": authors,
            }
        )

    return refs


def parse_body_chunks(root):
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return []

    chunks = []
    chunk_idx = 0

    for div in body.findall("tei:div", NS):
        head = div.find("tei:head", NS)
        section_heading = node_text(head) if head is not None else None
        section_number = head.get("n") if head is not None else None

        for p in div.findall("tei:p", NS):
            text = node_text(p)
            if not text:
                continue

            cited_ids = set()
            for ref_el in p.findall(".//tei:ref[@type='bibr']", NS):
                target = ref_el.get("target")
                if target and target.startswith("#"):
                    cited_ids.add(target[1:])

            chunk_idx += 1
            chunks.append(
                {
                    "chunk_id": f"para-{chunk_idx}",
                    "section_heading": section_heading,
                    "section_number": section_number,
                    "text": text,
                    "citations": sorted(cited_ids),
                }
            )

    return chunks


def tei_to_paper_db(tei_path: Path) -> dict:
    tree = ET.parse(tei_path)
    root = tree.getroot()

    meta = parse_paper_metadata(root)
    refs = parse_references(root)
    chunks = parse_body_chunks(root)

    return {
        "paper_id": meta["paper_id"],
        "title": meta["title"],
        "arxiv_id": meta["arxiv_id"],
        "authors": meta["authors"],
        "references": refs,
        "chunks": chunks,
    }


def run(*, workspace_root: Optional[Path] = None, input_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> None:
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[1]
    if input_dir is None:
        input_dir = workspace_root / "OUTPUT" / "tei_references"
    if output_dir is None:
        output_dir = workspace_root / "OUTPUT" / "text" / "reference"

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tei_file in input_dir.glob("**/*"):
        if not tei_file.is_file():
            continue
        if tei_file.suffix.lower() not in {".xml", ".tei"} and not tei_file.name.endswith(".tei.xml"):
            continue

        try:
            paper_db = tei_to_paper_db(tei_file)
            rel = tei_file.relative_to(input_dir)
            out_path = (output_dir / rel).with_suffix(".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(paper_db, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Written: {out_path}")
        except Exception as e:
            print(f"Failed to process {tei_file}: {e}")


if __name__ == "__main__":
    run()
