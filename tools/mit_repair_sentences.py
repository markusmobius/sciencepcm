import argparse
import difflib
import hashlib
import json
import unicodedata
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pysbd


def normalize(text):
    return "".join(character.lower() for character in unicodedata.normalize("NFD", text) if character.isalnum())


def normalized_positions(text):
    positions = []
    for index, character in enumerate(text):
        positions.extend(index for part in unicodedata.normalize("NFD", character) if part.isalnum())
    if len(positions) != len(normalize(text)):
        raise ValueError("Normalization changed a character's length unexpectedly.")
    return positions


def split_sentences(segmenter, text):
    spans = segmenter.segment(text)
    if not spans or "".join(text[span.start:span.end] for span in spans) != text:
        raise ValueError("Sentence offsets do not cover the original abstract exactly.")
    sentences = []
    start = 0
    for span in spans:
        end = span.end
        while end < len(text) and text[end] in "\"'\u2019\u201d)]}":
            end += 1
        sentence = text[start:end].strip()
        if not sentence:
            raise ValueError("Empty sentence after adjusting closing punctuation.")
        sentences.append(sentence)
        start = end
    if "".join("".join(sentences).split()) != "".join(text.split()):
        raise ValueError("Sentence boundaries changed abstract text.")
    return sentences


def match_body_abstract(paper):
    reference = normalize(paper["reference_abstract"] or "")
    if not reference:
        return {"status": "no_published_abstract", "edits": []}
    source = paper["source_sentences"]
    normalized = [normalize(sentence["text"]) for sentence in source]
    combined = "".join(normalized)
    start = combined.find(reference)
    offsets = [0]
    for sentence in normalized:
        offsets.append(offsets[-1] + len(sentence))
    if start >= 0 and combined.find(reference, start + 1) < 0:
        end = start + len(reference)
        edits = []
        for index, sentence in enumerate(source):
            if offsets[index + 1] <= start or offsets[index] >= end:
                continue
            local_start = max(0, start - offsets[index])
            local_end = min(len(normalized[index]), end - offsets[index])
            positions = normalized_positions(sentence["text"])
            raw_start = positions[local_start] if local_start else 0
            raw_end = positions[local_end] if local_end < len(positions) else len(sentence["text"])
            replacement = (sentence["text"][:raw_start] + sentence["text"][raw_end:]).strip()
            edits.append({"path": sentence["path"], "original_text": sentence["text"], "replacement_text": replacement or None})
        status = "partial_sentence_match" if any(edit["replacement_text"] for edit in edits) else "complete_normalized_match"
        return {"status": status, "edits": edits}
    if start >= 0:
        return {"status": "multiple_complete_matches", "edits": []}
    candidates = []
    for index, sentence in enumerate(normalized[:80]):
        if len(sentence) < 40:
            continue
        prefix = reference[:len(sentence)]
        similarity = difflib.SequenceMatcher(None, sentence, prefix, autojunk=False).ratio()
        candidates.append((similarity, index))
    best_similarity, first = max(candidates, default=(0, 0))
    windows = []
    for last in range(first + 1, min(len(source), first + 40) + 1):
        text = "".join(normalized[first:last])
        similarity = difflib.SequenceMatcher(None, reference, text, autojunk=False).ratio()
        windows.append((similarity, last))
        if len(text) > 2 * len(reference):
            break
    similarity, last = max(windows, default=(0, first))
    return {
        "status": "needs_boundary_review",
        "edits": [],
        "suggested_start_similarity": best_similarity,
        "suggested_similarity": similarity,
        "suggested_source": source[first:last],
    }


def reviewed_body_abstract(paper, review):
    doi = paper["reference_doi"]
    source = paper["source_sentences"]
    if doi in review["preserve_body_missing_abstract"]:
        if source[0]["heading"] != "1 Introduction":
            raise ValueError(f"Reviewed introduction boundary changed: {doi}")
        return {"status": "abstract_absent_from_body", "edits": [], "review_note": "Extracted body begins at the introduction; preserved unchanged."}
    decision = review["body_reviews"].get(doi)
    if not decision:
        return match_body_abstract(paper)
    first, last = decision["source_start"], decision["source_end"]
    if not 0 <= first < last <= len(source):
        raise ValueError(f"Invalid reviewed source window: {doi}")
    window = source[first:last]
    preserved = decision.get("preserve_texts", [])
    for text in preserved:
        if sum(sentence["text"] == text for sentence in window) != 1:
            raise ValueError(f"Reviewed contribution note changed: {doi}")
    removed = [sentence for sentence in window if sentence["text"] not in preserved]
    similarity = difflib.SequenceMatcher(None, normalize(paper["reference_abstract"]),
                                         normalize(" ".join(sentence["text"] for sentence in removed)), autojunk=False).ratio()
    if similarity < 0.85:
        raise ValueError(f"Reviewed source window no longer matches: {doi}: {similarity}")
    return {
        "status": "reviewed_source_match",
        "review_note": decision["reason"],
        "preserved_notes": preserved,
        "edits": [{"path": sentence["path"], "original_text": sentence["text"], "replacement_text": None} for sentence in removed],
    }


def write_examples(directory):
    report = json.loads((directory / "repair-report.json").read_text(encoding="utf-8-sig"))
    candidate_path = directory / "repair-candidate.json"
    if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != report["candidate_sha256"]:
        raise ValueError("The example candidate differs from the verified PDS candidate.")
    with Path(report["output_path"]).open("rb") as artifact:
        if hashlib.file_digest(artifact, "sha256").hexdigest() != report["output_sha256"]:
            raise ValueError("The verified PDS has changed.")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    plan_path = directory / "repair-plan.json"
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != candidate["reference_plan_sha256"]:
        raise ValueError("The example source plan has changed.")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    originals = {paper["reference_doi"]: paper for paper in plan["papers"]}
    selected = ("10.1162/neco_a_01243", "10.1162/neco_a_01245", "10.1162/neco_a_01341",
                "10.1162/neco_a_01335", "10.1162/neco_c_01397")
    papers = {paper["reference_doi"]: paper for paper in candidate["papers"]}
    examples = []
    upload_status = (f"Verified cloud copy: {report['cloud_path']}." if report.get("uploaded")
                     else "Nothing has been uploaded.")
    config_status = ("The MIT input configuration points to this file." if report.get("config_changed")
                     else "The configuration is unchanged.")
    markdown = ["# Corrected MIT PDS Examples", "",
                f"Local artifact: [input_2026_09_23_fixed.pds](input_2026_09_23_fixed.pds)", "",
                f"Verified {report['records']} records, {report['published_abstracts']} published abstracts, "
                f"{report['abstract_sentences']} sentences, and {report['authors_changed']} changed author lists.", "",
                f"The original PDS is unchanged. {upload_status} {config_status}", "",
                "The abstract field is a flat array of objects with sentence_index (1-based) and text. "
                "The JSON companion contains the exact corrected values verified by the PDS read-back check.", ""]
    for doi in selected:
        paper = papers[doi]
        source = originals[doi]
        body = paper["body_abstract"]
        first_edit = body["edits"][0] if body["edits"] else None
        original_location = next((sentence for sentence in source["source_sentences"]
                                  if first_edit and sentence["path"] == first_edit["path"]), None)
        examples.append({
            "doi": doi,
            "key_path": paper["key_path"],
            "before": {"metadata": {"title": paper["title"], "authors": paper["original_authors"]}, "abstract": []},
            "original_abstract_location": original_location,
            "after": {"metadata": {"title": paper["title"], "authors": paper["authors"]}, "abstract": paper["abstract"]},
            "abstract_status": paper["abstract_status"],
            "body_action": body["status"],
            "formula_repairs": paper["formula_repairs"],
        })
        markdown.extend([f"## {paper['title']}", "", f"DOI: https://doi.org/{doi}", "",
                         "Before authors: " + ("; ".join(paper["original_authors"]) or "(empty)"), "",
                         "Before abstract: []", "",
                         "After authors: " + "; ".join(paper["authors"]), "",
                         f"Body handling: {body['status']}", ""])
        if original_location:
            markdown.extend(["Original abstract location: section heading " +
                             json.dumps(original_location["heading"], ensure_ascii=False) + ".", ""])
        if paper["abstract"]:
            markdown.extend(["After abstract:", ""])
            markdown.extend(f"{sentence['sentence_index']}. {sentence['text']}" for sentence in paper["abstract"])
        else:
            markdown.append("After abstract: []. This erratum has no published abstract; its three existing authors "
                            "are confirmed by the corrected article and the erratum's deposited title. No summary was invented.")
        markdown.append("")
    (directory / "repair-examples.json").write_text(json.dumps({"verified_output_sha256": report["output_sha256"],
                                                               "examples": examples}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "repair-examples.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Created {len(examples)} complete before/after examples from the verified candidate.")
    print(directory / "repair-examples.md")
    print(directory / "repair-examples.json")


def main():
    parser = argparse.ArgumentParser(description="Prepare sentence-level MIT PDS corrections from verified DOI metadata.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()
    if args.examples:
        write_examples(args.directory)
        return
    if version("pysbd") != "0.3.4":
        raise ValueError("Use pysbd==0.3.4 for reproducible sentence boundaries.")

    plan = json.loads((args.directory / "repair-plan.json").read_text(encoding="utf-8-sig"))
    journal = json.loads((args.directory / "crossref-journal.json").read_text(encoding="utf-8-sig"))
    review = json.loads((args.directory / "reviewed-corrections.json").read_text(encoding="utf-8-sig"))
    if review["source_sha256"] != plan["source_sha256"]:
        raise ValueError("The manual review belongs to a different source file.")
    works = {work["DOI"].lower(): work for work in journal["message"]["items"]}
    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    corrections = []
    short_sentences = []
    for paper in plan["papers"]:
        doi = paper["reference_doi"]
        authors = paper["reference_authors"]
        abstract = paper["reference_abstract"]
        formula_fixes = review["formula_replacements"].get(doi)
        if formula_fixes:
            parts = abstract.split("[Formula: see text]")
            if len(parts) != len(formula_fixes["values"]) + 1:
                raise ValueError(f"Formula placeholder count changed: {doi}")
            abstract = parts[0] + "".join(value + part for value, part in zip(formula_fixes["values"], parts[1:]))
            paper = {**paper, "reference_abstract": abstract}
        text_repairs = review["abstract_text_repairs"].get(doi)
        if text_repairs:
            for replacement in text_repairs["replacements"]:
                if abstract.count(replacement["old"]) != 1:
                    raise ValueError(f"Reviewed abstract expression changed: {doi}")
                abstract = abstract.replace(replacement["old"], replacement["new"], 1)
            paper = {**paper, "reference_abstract": abstract}
        if abstract and "[Formula: see text]" in abstract:
            raise ValueError(f"Unresolved mathematical expression: {doi}")
        status = "published_abstract"
        if paper["issues"]:
            if doi != "10.1162/neco_c_01397" or abstract:
                raise ValueError(f"Unresolved reference: {doi}: {paper['issues']}")
            original = works["10.1162/neco_a_01300"]
            authors = [" ".join(filter(None, (author.get("given"), author.get("family")))) for author in original["author"]]
            if authors != paper["original_authors"]:
                raise ValueError("The erratum authors do not match the corrected article.")
            if normalize(original["title"][0]) not in normalize(paper["title"]):
                raise ValueError("The erratum title does not identify the corrected article.")
            if not all(normalize(author) in normalize(paper["reference_title"]) for author in authors):
                raise ValueError("The erratum deposit does not confirm its authors.")
            status = "no_published_abstract_erratum"
            sentences = []
        else:
            if not authors or not abstract:
                raise ValueError(f"Missing required reference fields: {doi}")
            sentences = split_sentences(segmenter, abstract)
            if not sentences or any(not sentence for sentence in sentences):
                raise ValueError(f"Empty sentence: {doi}")
            for sentence in sentences:
                if len(sentence) < 25:
                    short_sentences.append({"doi": doi, "text": sentence})

        corrections.append({
            "key_path": paper["key_path"],
            "title": paper["title"],
            "reference_doi": doi,
            "reference_title": paper["reference_title"],
            "original_authors": paper["original_authors"],
            "authors": authors,
            "abstract_status": status,
            "formula_repairs": formula_fixes,
            "abstract_text_repairs": text_repairs,
            "abstract": [{"sentence_index": index + 1, "text": sentence} for index, sentence in enumerate(sentences)],
            "body_abstract": reviewed_body_abstract(paper, review),
        })

    unresolved = [paper["reference_doi"] for paper in corrections if paper["body_abstract"]["status"] in ("needs_boundary_review", "multiple_complete_matches")]
    if unresolved:
        raise ValueError(f"Unresolved body boundaries: {unresolved}")
    output = {
        "source_path": plan["source_path"],
        "source_sha256": plan["source_sha256"],
        "review_sha256": hashlib.sha256((args.directory / "reviewed-corrections.json").read_bytes()).hexdigest(),
        "reference_plan_sha256": hashlib.sha256((args.directory / "repair-plan.json").read_bytes()).hexdigest(),
        "reference_provider": "Crossref publisher-deposited DOI metadata",
        "sentence_segmenter": "pysbd==0.3.4, English, original character spans with closing punctuation attached",
        "papers": corrections,
        "short_sentences_for_review": short_sentences,
    }
    output_path = args.directory / "repair-candidate.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(corrections)} records, {sum(len(paper['abstract']) for paper in corrections)} abstract sentences.")
    print(f"Explicit no-abstract exceptions: {sum(paper['abstract_status'] != 'published_abstract' for paper in corrections)}")
    print(f"Short sentences to inspect: {len(short_sentences)}")
    print(f"Body matches: {dict(Counter(paper['body_abstract']['status'] for paper in corrections))}")
    print(output_path)


if __name__ == "__main__":
    main()