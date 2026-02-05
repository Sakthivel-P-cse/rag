import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple


NS = {
    "tei": "http://www.tei-c.org/ns/1.0",
}


def clone_element(elem: ET.Element) -> ET.Element:
    new = ET.Element(elem.tag, elem.attrib)
    new.text = elem.text
    new.tail = elem.tail
    for child in list(elem):
        new.append(clone_element(child))
    return new


def get_parent(root: ET.Element, target: ET.Element) -> Optional[ET.Element]:
    for parent in root.iter():
        for child in list(parent):
            if child is target:
                return parent
    return None


def find_first(root: ET.Element, xpath: str) -> Optional[ET.Element]:
    matches = root.findall(xpath, NS)
    return matches[0] if matches else None


def write_xml(root: ET.Element, path: str) -> None:
    ET.register_namespace("", NS["tei"])
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def find_listbibl(root: ET.Element) -> Optional[ET.Element]:
    candidates = []
    candidates.extend(root.findall(".//tei:text/tei:back/tei:listBibl", NS))
    candidates.extend(root.findall(".//tei:listBibl", NS))
    return candidates[0] if candidates else None


def extract_all_text(node: ET.Element) -> str:
    texts = []
    if node.text:
        texts.append(node.text)
    for child in list(node):
        texts.append(extract_all_text(child))
        if child.tail:
            texts.append(child.tail)
    return "".join(texts)


def normalize_whitespace(s: str) -> str:
    return " ".join(s.split())


def extract_main_text(root: ET.Element, listbibl: Optional[ET.Element]) -> str:
    parts = []
    for text_node in root.findall(".//tei:text", NS):
        for section_xpath in ("tei:front", "tei:body"):
            for sec in text_node.findall(f"./{section_xpath}", NS):
                parts.append(extract_all_text(sec))
        back = text_node.find("./tei:back", NS)
        if back is not None:
            for child in list(back):
                if child.tag == f"{{{NS['tei']}}}listBibl":
                    continue
                parts.append(extract_all_text(child))
    if not parts:
        for child in list(root):
            if listbibl is not None and child is listbibl:
                continue
            parts.append(extract_all_text(child))
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def extract_references_text(listbibl: Optional[ET.Element]) -> str:
    if listbibl is None:
        return ""
    lines = []
    for bibl in listbibl.findall(".//tei:bibl", NS):
        line = normalize_whitespace(extract_all_text(bibl))
        if line:
            lines.append(line)
    if not lines:
        raw = normalize_whitespace(extract_all_text(listbibl))
        if raw:
            lines.append(raw)
    return "\n".join(lines)


def process_file(xml_path: str) -> Tuple[str, str]:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        return ("", f"ERROR parsing XML: {e}")
    listbibl = find_listbibl(root)
    main_text = extract_main_text(root, listbibl)
    refs_text = extract_references_text(listbibl)
    return (main_text, refs_text)


def save_outputs(*, xml_path: Path, rel_path: Path, out_main_xml_dir: Path, out_refs_xml_dir: Path) -> None:
    base = os.path.splitext(xml_path.name)[0]
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except ET.ParseError:
        return

    main_root = clone_element(root)

    main_header = find_first(main_root, ".//tei:teiHeader")
    if main_header is not None:
        file_desc = main_header.find("./tei:fileDesc", NS)
        if file_desc is not None:
            source_desc = file_desc.find("./tei:sourceDesc", NS)
            if source_desc is not None:
                file_desc.remove(source_desc)

    main_text_node = find_first(main_root, ".//tei:text")
    if main_text_node is not None:
        for node in list(main_text_node.findall("./tei:front", NS)):
            main_text_node.remove(node)
        for node in list(main_text_node.findall("./tei:back", NS)):
            main_text_node.remove(node)
        body = main_text_node.find("./tei:body", NS)
        if body is None:
            body = ET.Element(f"{{{NS['tei']}}}body")
            main_text_node.append(body)
        for ref_div in main_text_node.findall('.//tei:div[@type="references"]', NS):
            parent = get_parent(main_text_node, ref_div)
            if parent is not None:
                parent.remove(ref_div)
        for tag in ("biblStruct", "bibl", "listBibl"):
            for node in list(main_text_node.findall(f".//tei:{tag}", NS)):
                parent = get_parent(main_text_node, node)
                if parent is not None:
                    parent.remove(node)

    refs_root = clone_element(root)
    refs_text_node = find_first(refs_root, ".//tei:text")
    if refs_text_node is not None:
        back = refs_text_node.find("./tei:back", NS)
        if back is None:
            back = ET.Element(f"{{{NS['tei']}}}back")
            refs_text_node.append(back)
        for ref_div in list(refs_text_node.findall('.//tei:div[@type="references"]', NS)):
            parent = get_parent(refs_text_node, ref_div)
            if parent is not None:
                parent.remove(ref_div)
            back.append(ref_div)
        for lb in list(refs_text_node.findall(".//tei:listBibl", NS)):
            parent = get_parent(refs_text_node, lb)
            if parent is not None and parent is not back:
                parent.remove(lb)
                back.append(lb)
        for node in list(refs_text_node.findall("./tei:front", NS)):
            refs_text_node.remove(node)
        for node in list(refs_text_node.findall("./tei:body", NS)):
            refs_text_node.remove(node)
    else:
        tei_ns = NS["tei"]
        tei_text = ET.Element(f"{{{tei_ns}}}text")
        back = ET.Element(f"{{{tei_ns}}}back")
        div_refs = ET.Element(f"{{{tei_ns}}}div", {"type": "references"})
        div_refs.append(ET.Element(f"{{{tei_ns}}}listBibl"))
        back.append(div_refs)
        refs_root.append(tei_text)
        tei_text.append(back)

    out_main_path = (out_main_xml_dir / rel_path).with_name(f"{base}.tei.xml")
    out_refs_path = (out_refs_xml_dir / rel_path).with_name(f"{base}.tei.xml")
    out_main_path.parent.mkdir(parents=True, exist_ok=True)
    out_refs_path.parent.mkdir(parents=True, exist_ok=True)
    write_xml(main_root, str(out_main_path))
    write_xml(refs_root, str(out_refs_path))


def run(
    *,
    workspace_root: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    out_main_xml_dir: Optional[Path] = None,
    out_refs_xml_dir: Optional[Path] = None,
) -> None:
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[1]

    if input_dir is None:
        input_dir = workspace_root / "OUTPUT" / "tei"
    if out_main_xml_dir is None:
        out_main_xml_dir = workspace_root / "OUTPUT" / "tei_main"
    if out_refs_xml_dir is None:
        out_refs_xml_dir = workspace_root / "OUTPUT" / "tei_references"

    out_main_xml_dir.mkdir(parents=True, exist_ok=True)
    out_refs_xml_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted([p for p in Path(input_dir).rglob("*.xml") if p.is_file()])
    if not xml_files:
        print(f"No XML files found in {input_dir} (searched recursively)")
        return

    total = 0
    input_dir = Path(input_dir)
    for xml_path in xml_files:
        rel_path = xml_path.relative_to(input_dir)
        _main_text, _refs_text = process_file(str(xml_path))
        save_outputs(xml_path=xml_path, rel_path=rel_path, out_main_xml_dir=out_main_xml_dir, out_refs_xml_dir=out_refs_xml_dir)
        total += 1

    print(f"Processed {total} file(s).")
    print(f"Main TEI in: {out_main_xml_dir}")
    print(f"References TEI in: {out_refs_xml_dir}")


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
