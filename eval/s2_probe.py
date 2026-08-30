"""Measure how much of OpenAlex's abstract gap Semantic Scholar could actually fill.

Both OpenAlex and S2 take most abstracts from the same Crossref deposits, so they
inherit the same publisher gap. S2's extra value is abstracts parsed out of PDFs,
which only exist where a PDF was openly available. That makes the overlap an
empirical question, and a wrong guess costs a multi-hundred-GB download and a merge
pipeline. This asks S2 about a sample of DOIs and reports the fill rate.

The number that decides the merge is ``s2_has_abstract`` on the missing-abstract
cohort. The control cohort exists to prove the harness works: if OpenAlex has an
abstract and S2 claims not to, something is wrong with the join, not with S2.

``s2_has_oa_pdf`` is a rough upper bound on how many of these papers S2ORC could
supply full text for, which on our own measurements is worth far more than an
abstract (0.932 vs 0.247 nDCG@10 on methods questions).

Sampling from Parquet needs a v3 digest: v2 dropped abstract-less works entirely, so
there is nothing to sample. Against a v2 digest, or before the ingest finishes, pass
a DOI list with --dois instead.

Usage:
    python eval/s2_probe.py --parquet '~/mcp/data/openalex/abstracts/abstracts-part-000*.parquet' \\
        --sample 500 --control --out runs/s2-probe.jsonl

    python eval/s2_probe.py --dois missing.txt --out runs/s2-probe.jsonl

An API key raises the rate limit a long way. Get one at
https://www.semanticscholar.org/product/api and pass --api-key, or set SEMANTIC_S3
(or S2_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
OPENALEX_URL = "https://api.openalex.org/works"
FIELDS = "externalIds,title,year,abstract,isOpenAccess,openAccessPdf"

# The batch endpoint caps a request at 500 ids.
BATCH_SIZE = 500

DOI_PREFIX = re.compile(r"^(https?://)?(dx\.)?doi\.org/", re.IGNORECASE)


def normalise_doi(raw: str | None) -> str | None:
    """OpenAlex stores a DOI as a URL; S2 wants the bare suffix."""
    if not raw:
        return None
    doi = DOI_PREFIX.sub("", raw.strip()).lower()
    return doi if doi.startswith("10.") else None


def sample_from_parquet(glob: str, sample: int, control: bool, where: str | None) -> list[dict]:
    import duckdb

    connection = duckdb.connect()
    cohorts = [("missing-abstract", "(abstract IS NULL OR trim(abstract) = '')")]
    if control:
        cohorts.append(("has-abstract", "(abstract IS NOT NULL AND trim(abstract) <> '')"))

    records: list[dict] = []
    for cohort, predicate in cohorts:
        if where:
            predicate = f"{predicate} AND ({where})"

        # The filter has to sit in a subquery: DuckDB applies USING SAMPLE before WHERE,
        # so the obvious phrasing samples the whole corpus and then filters the sample
        # down to a handful of rows.
        rows = connection.execute(
            f"""
            SELECT * FROM (
                SELECT openalex_id, title, doi, publication_year, type, cited_by_count
                FROM read_parquet(?)
                WHERE {predicate}
            )
            USING SAMPLE {int(sample)} ROWS
            """,
            [os.path.expanduser(glob)],
        ).fetchall()

        for openalex_id, title, doi, year, work_type, cited_by in rows:
            records.append({
                "cohort": cohort,
                "openalex_id": openalex_id,
                "title": title,
                "doi": normalise_doi(doi),
                "year": year,
                "type": work_type,
                "cited_by_count": cited_by,
            })

        print(f"sampled {len(rows):,} from cohort '{cohort}'", file=sys.stderr)

    return records


def sample_from_openalex_api(filter_expr: str, sample: int, mailto: str | None) -> list[dict]:
    """Pull a cohort straight from the OpenAlex API, so the missing-abstract question can
    be answered without waiting for a local digest that contains such works."""
    records: list[dict] = []
    cursor = "*"

    while len(records) < sample and cursor:
        params = {
            "filter": filter_expr,
            "per-page": min(200, sample - len(records)),
            "cursor": cursor,
            "select": "id,title,doi,publication_year,type,cited_by_count",
        }
        if mailto:
            params["mailto"] = mailto

        payload = requests.get(OPENALEX_URL, params=params, timeout=60)
        payload.raise_for_status()
        body = payload.json()

        for work in body.get("results", []):
            records.append({
                "cohort": "openalex-api",
                "openalex_id": work.get("id"),
                "title": work.get("title"),
                "doi": normalise_doi(work.get("doi")),
                "year": work.get("publication_year"),
                "type": work.get("type"),
                "cited_by_count": work.get("cited_by_count"),
            })

        cursor = body.get("meta", {}).get("next_cursor")
        if not body.get("results"):
            break

    print(f"sampled {len(records):,} from OpenAlex filter '{filter_expr}'", file=sys.stderr)
    return records


def load_doi_file(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept a plain DOI list or the JSONL this script writes.
        raw = json.loads(line).get("doi") if line.startswith("{") else line
        records.append({"cohort": "from-file", "openalex_id": None, "title": None,
                        "doi": normalise_doi(raw), "year": None, "type": None,
                        "cited_by_count": None})
    return records


def titles_agree(left: str | None, right: str | None) -> bool | None:
    """Cheap guard against a silently misaligned join: a DOI lookup that returns a
    different paper's title means the answers came back out of order."""
    if not left or not right:
        return None
    reduce = lambda text: "".join(c for c in text.lower() if c.isalnum())[:40]
    return reduce(left) == reduce(right)


def fetch_batch(session: requests.Session, dois: list[str], sleep: float) -> list[dict | None]:
    body = {"ids": [f"DOI:{doi}" for doi in dois]}

    for attempt in range(5):
        response = session.post(BATCH_URL, params={"fields": FIELDS}, json=body, timeout=120)
        if response.status_code == 200:
            time.sleep(sleep)
            return response.json()

        if response.status_code in (429, 500, 502, 503, 504):
            backoff = sleep * (2 ** attempt) + 1
            print(f"  HTTP {response.status_code}, retrying in {backoff:.0f}s", file=sys.stderr)
            time.sleep(backoff)
            continue

        # S2 rejects the whole chunk when not one id resolves, which is a legitimate
        # answer - none of them are in S2 - rather than a reason to abandon the run.
        if response.status_code == 400 and "No valid paper ids" in response.text:
            return [None] * len(dois)

        raise RuntimeError(
            f"batch failed: HTTP {response.status_code} {response.text[:300]}")

    raise RuntimeError(f"batch failed after 5 attempts (last status {response.status_code})")


def probe(records: list[dict], api_key: str | None, sleep: float) -> list[dict]:
    session = requests.Session()
    if api_key:
        session.headers["x-api-key"] = api_key

    resolvable = [record for record in records if record["doi"]]
    print(f"{len(resolvable):,} of {len(records):,} records carry a usable DOI", file=sys.stderr)

    for start in range(0, len(resolvable), BATCH_SIZE):
        chunk = resolvable[start:start + BATCH_SIZE]
        answers = fetch_batch(session, [record["doi"] for record in chunk], sleep)

        # The endpoint answers positionally, with null where the id did not resolve. If
        # that ever stops holding, every field below is attached to the wrong paper, so
        # fail loudly rather than produce a plausible-looking wrong answer.
        if len(answers) != len(chunk):
            raise RuntimeError(
                f"batch returned {len(answers)} answers for {len(chunk)} ids; "
                "positional alignment is unsafe")

        for record, answer in zip(chunk, answers):
            record["s2_found"] = answer is not None
            record["s2_abstract"] = (answer or {}).get("abstract")
            record["s2_title"] = (answer or {}).get("title")
            record["s2_corpus_id"] = (answer or {}).get("externalIds", {}).get("CorpusId")
            record["s2_is_oa"] = (answer or {}).get("isOpenAccess")
            record["s2_oa_pdf"] = ((answer or {}).get("openAccessPdf") or {}).get("url")
            record["title_match"] = titles_agree(record.get("title"), record.get("s2_title"))

        print(f"  {min(start + BATCH_SIZE, len(resolvable)):,}/{len(resolvable):,}", file=sys.stderr)

    return records


def summarise(records: list[dict]) -> dict:
    summary = {}
    for cohort in sorted({record["cohort"] for record in records}):
        rows = [record for record in records if record["cohort"] == cohort]
        with_doi = [record for record in rows if record["doi"]]
        found = [record for record in with_doi if record.get("s2_found")]
        with_abstract = [record for record in found if (record.get("s2_abstract") or "").strip()]
        with_pdf = [record for record in found if record.get("s2_oa_pdf")]
        comparable = [record for record in found if record.get("title_match") is not None]
        agreeing = [record for record in comparable if record["title_match"]]

        summary[cohort] = {
            "sampled": len(rows),
            "with_doi": len(with_doi),
            "s2_found": len(found),
            "s2_has_abstract": len(with_abstract),
            "s2_has_oa_pdf": len(with_pdf),
            # Two denominators, because they answer different questions. Rates over the
            # whole cohort say what a merge would recover; rates over the resolvable
            # subset say whether S2 and the DOI join work at all. A corpus with few DOIs
            # drags the first down while the second stays high.
            "doi_rate": len(with_doi) / len(rows) if rows else 0.0,
            "abstract_fill_rate": len(with_abstract) / len(rows) if rows else 0.0,
            "oa_pdf_rate": len(with_pdf) / len(rows) if rows else 0.0,
            "s2_found_of_resolvable": len(found) / len(with_doi) if with_doi else 0.0,
            "abstract_of_found": len(with_abstract) / len(found) if found else 0.0,
            # Well below 1.0 means the join is wrong and every other number here is junk.
            "title_agreement": len(agreeing) / len(comparable) if comparable else None,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parquet", help="Glob over the OpenAlex v3 digest to sample from.")
    source.add_argument("--dois", type=Path, help="File of DOIs, one per line.")
    source.add_argument("--openalex-filter",
                        help="OpenAlex API filter defining the cohort, e.g. "
                             "'has_abstract:false,type:article,cited_by_count:>50'. Needs no local "
                             "digest, so the missing-abstract cohort can be measured immediately.")
    parser.add_argument("--mailto", help="Contact address for OpenAlex's polite pool.")
    parser.add_argument("--sample", type=int, default=500, help="Records per cohort. Default 500.")
    parser.add_argument("--control", action="store_true",
                        help="Also sample works that DO have an OpenAlex abstract, to validate the join.")
    parser.add_argument("--where", help="Extra SQL predicate on the sample, e.g. "
                                        "\"type = 'article' AND cited_by_count >= 50\". A random draw "
                                        "from all of OpenAlex is mostly long tail that S2 never indexed, "
                                        "which is not the population a news-matching service serves.")
    parser.add_argument("--api-key", default=os.environ.get("SEMANTIC_S3") or os.environ.get("S2_API_KEY"))
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between batches. Default 1.")
    parser.add_argument("--out", type=Path, default=Path("runs/s2-probe.jsonl"))
    args = parser.parse_args()

    if not args.api_key:
        print("no S2 API key: the shared rate limit is low, keep --sample small", file=sys.stderr)

    if args.dois:
        records = load_doi_file(args.dois)
    elif args.openalex_filter:
        records = sample_from_openalex_api(args.openalex_filter, args.sample, args.mailto)
    else:
        records = sample_from_parquet(args.parquet, args.sample, args.control, args.where)
    if not records:
        print("nothing to probe", file=sys.stderr)
        return 1

    records = probe(records, args.api_key, args.sleep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarise(records)
    print()
    for cohort, counts in summary.items():
        total = counts["sampled"]
        resolvable = counts["with_doi"]
        print(f"cohort: {cohort}  sampled {total:,}")
        print(f"  {'with DOI':<22}{resolvable:>7,} ({resolvable / total:6.1%} of cohort)")
        for label, key in (("found in S2", "s2_found"),
                           ("S2 has abstract", "s2_has_abstract"),
                           ("S2 has OA PDF", "s2_has_oa_pdf")):
            value = counts[key]
            share = value / resolvable if resolvable else 0.0
            print(f"  {label:<22}{value:>7,} ({value / total:6.1%} of cohort, {share:6.1%} of DOIs)")
        print()

    print(f"wrote {args.out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
