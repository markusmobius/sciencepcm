"""How much of OpenAlex has no abstract, by population.

This decides whether abstract-less works belong in the index at all. Dropping them
is tempting - they retrieve on title alone - but the gap is not random: it widens
with citation count, so dropping them loses exactly the papers most likely to be
asked for. The fielded schema keeps them instead (see retrieval.md).

Two ``per-page=1`` requests per row, reading only ``meta.count``, so it downloads
almost nothing and needs no key. Re-run it when the digest is refreshed.

Usage:
    python eval/abstract_coverage.py
"""

import json
import urllib.parse
import urllib.request


def count(filter_expr: str) -> int:
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"per-page": "1", "filter": filter_expr})
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["meta"]["count"]


print(f"{'population':<44}{'total':>14}{'no abstract':>14}{'share':>8}")
print("-" * 80)

populations = []
for year in (2019, 2021, 2023, 2024, 2025):
    populations.append((f"{year} type=article", f"publication_year:{year},type:article"))
populations.append(("2025 article, cited>0", "publication_year:2025,type:article,cited_by_count:>0"))
populations.append(("2025 article, cited>=10", "publication_year:2025,type:article,cited_by_count:>9"))
populations.append(("2025 article, has DOI", "publication_year:2025,type:article,has_doi:true"))
populations.append(("all works", "type:article"))

for label, base in populations:
    total = count(base)
    missing = count(base + ",has_abstract:false")
    share = missing / total if total else 0
    print(f"{label:<44}{total:>14,}{missing:>14,}{share:>7.1%}")
