# Retrieval

Both services share one engine: `src/SciencePcm.Index/LexicalIndex.cs` for the index
and query, `src/SciencePcm.Server/RetrievalService.cs` for the pipeline. A query is BM25F
over a fielded Lucene index and a `bge-reranker-v2-m3` cross-encoder over the top 100,
**fused by reciprocal rank** rather than one replacing the other.

## Fielded BM25F

Title, abstract, authors, institutions, venue, identifiers and topics are **separate
indexed fields**. A query is one `DisjunctionMaxQuery` per analysed term across those
fields, tie-breaker 0.1 — the Lucene 4 way of writing BM25F (Robertson, Zaragoza and
Taylor, CIKM 2004).

| field | boost | notes |
| --- | --- | --- |
| `identifiers` | 4.0 | DOI, PMID, PMCID, ISSN — a match here is a certainty, not a hint |
| `title` | 3.0 | prose names papers by title |
| `authors` | 2.0 | analysed, for names appearing in prose |
| `venue` | 1.5 | journal and publisher |
| `body` | 1.0 | the abstract, or the passage |
| `institutions` | 1.0 | "MIT economists ..." |
| `topics_search` | 1.0 | topics and keywords |

Per-term rather than per-field dismax, deliberately: a query naming an author *and* a
topic has to score both, and whole-query dismax would take the best single field and
discard the rest.

**Per-field length normalisation is the whole point.** BM25 divides each term's
contribution by document length relative to the corpus average, so under a single
concatenated field one long field distorts every other. A 765-author trial paper was 27x
the average length and every term was scaled to 8.5% of its value. Now an author list can
only dilute `authors`.

It also means abstract-less works need no special handling — they have no `body` field
and compete on the fields they do have. That matters more than it sounds: 21.6% of 2025
articles have no abstract, rising to **29.1% among those with 10 or more citations**.

The `author` and `journal` filters are phrase queries over the analysed `authors` and
`venue` fields. They used to be exact terms over `author_exact` / `venue_exact`, which
fails on real data: `"Jennifer Doudna"`, `"Doudna, Jennifer A."` and bare `"Doudna"` all
occur upstream and each exact form matched a *disjoint* set of papers, while
`journal="Lancet"` returned nothing because the stored name is `"The Lancet"`. The
`*_exact` fields are still written so existing indexes stay schema-valid.

## Static priors and deduplication

`--citation-prior 2.0` multiplies the BM25 score by up to `1 + weight`, scaled by
`log10(1 + cited_by_count)/5.5`. Capped and multiplicative rather than additive: an
additive bonus has to be tuned against BM25's score scale, and at any weight large enough
to matter the most-cited papers in the corpus win every query outright. At weight 20 a
semaglutide query returned "Using thematic analysis in psychology" — 174,403 citations.

This is the documented approach for known-item retrieval (Kraaij, Westerveld and
Hiemstra, SIGIR 2002: document priors matter more than term weighting). It measurably
helped — MRR on the landmark set went 0.125 to 0.333 — and it is disabled when a
non-relevance sort is requested. Raising the default from 1.0 to 2.0 moved the landmark
sets from MRR 0.653 to 0.750 (long) and 0.625 to 0.667 (terse), and took the ChAdOx1
*Lancet* paper — 4,980 citations — from outside the top 10 to rank 6.

Results are deduplicated by title to the **best-cited** record, not the first seen.
OpenAlex holds the same paper more than once — a preprint, a stub typed `other`, a
merged-but-not-removed record — and the copies carry few or no citations, so keeping
whichever scored highest discarded the canonical one.

## Fusing the two stages

The cross-encoder score does not replace the BM25 ordering; the two rankings are combined
with reciprocal rank fusion, `1/(60 + rank)` summed over both. Sorting on the
cross-encoder logit alone throws away BM25 *and* the citation prior folded into it, and
the two scores are on unrelated scales so they cannot simply be added.

Measured on the live endpoint, 30 topical queries, 418 pooled LLM judgements:

| system | nDCG@10 | mean grade | % >=2 | hit@1 | MRR |
| --- | --- | --- | --- | --- | --- |
| BM25F + citation prior | 0.6277 | 1.618 | 49.5% | 73.3% | 0.808 |
| **fused with the cross-encoder** | **0.8686** | **2.063** | **66.7%** | **90.0%** | **0.928** |

The trade-off is real but lopsided. On the two known-item landmark sets BM25 alone is
*better* — MRR 0.875 and 0.786, against 0.667 and 0.653 fused — because a known-item query
is an identity lookup that BM25F plus the citation prior already answers, and the
cross-encoder only adds noise. Eight unjudged known-item queries do not outweigh 30 judged
topical ones, and topical questions are what the services are for.

Not to be confused with the rejected *BM25 + dense* fusion below. Fusing two rankings of
very different quality dilutes the stronger one; fusing two of comparable quality that
fail differently is the case RRF is for.

## Rejected

Kept here because re-proposing them costs a day each. The scripts that produced these
numbers have been deleted — they depend on vectors and qrels that no longer exist — so
these are the surviving record.

| approach | result |
| --- | --- |
| dense retrieval (MedCPT) | 0.117 nDCG@10 HNSW, 0.125 exact, against BM25's 0.2255. Not a bug: stored vectors re-encoded with PyTorch matched at cosine 1.0000 on 198/198, and HNSW cost only ~6% against brute force. |
| RRF fusion of BM25 + dense | recall@100 0.4639 → 0.4736, and nDCG@10 *dropped*: fusion dilutes the stronger system. With rerank, 0.7647 against 0.7713 for BM25 + rerank. Noise. |
| MedCPT encoder variants | Reference usage (pair input, raw dot product) beat ours (concatenated, cosine) 0.7586 to 0.7134 on a small pool. Real, but ~6% on the leg that lost anyway. |
| `--max-doc-freq-ratio` | Changed results, improved nothing. Removed. |
| `--bm25-b` | Length normalisation is not the lever once fields are separate. Removed. |
| `--rerank-candidates` 500 or 1000 | 500 changed nothing; 1000 made the landmark set slightly worse. Stays at 100. |

**BioASQ was the bigger trap.** It ranked BM25+rerank *below* BM25 alone, which reversed
once an LLM judge looked at the same runs. Only ~10% of its gold judgements exist in this
corpus — about 4 judged documents per query among 5.3M — so unjudged-but-relevant papers
scored as errors, which specifically penalises semantic methods while BM25's exact-term
matches are likelier to be the ones annotators found. Do not let a sparse external
benchmark pick the architecture. The qrels builder and its harness are gone.

The other lesson from that period: the cross-encoder is not the weak stage. On the
landmark set it lifted papers from BM25 rank 14 to rank 1, and rescued papers BM25 ranked
outside its own top 50. When a paper is missing, look at the first stage.

## Reranker

`BAAI/bge-reranker-v2-m3`, exported to ONNX, shared by both services.

| reranker | nDCG@10 | hit@1 | notes |
| --- | --- | --- | --- |
| bge-reranker-v2-m3 (568M) | **0.8368** | 40.0% | XLM-R, multilingual |
| MedCPT cross-encoder (110M) | 0.7898 | 23.3% | |
| bge-reranker-base (278M) | 0.7585 | — | newer and bigger than MedCPT, and worse |
| BM25 alone | 0.6859 | — | |

Two things that cost time. **"Modern" is not the variable** — `bge-reranker-base` is 2.5x
MedCPT's size and newer, and lost. And **`%>=2` and mean grade are blind to ordering**,
so on known-item work they showed BGE and MiniLM as identical while hit@1 differed by
17 points. For known-item use hit@1 and MRR.

Being XLM-R, BGE needs SentencePiece, which FastBertTokenizer cannot do. The tokenizer is
therefore its own ONNX graph via `onnxruntime-extensions`; `CrossEncoderFactory` picks the
implementation by whether `tokenizer.onnx` is present. Parity against HuggingFace is
verified at export time — a tokenizer that disagrees degrades retrieval silently.

## Diagnosing a missing paper

```bash
# is it in the index, and which stage loses it?
$LAB eval/known_item.py --endpoint <url> --questions eval/questions-landmark.jsonl \
  --stages --show 5

# Lucene's own account of a score
dotnet run --project src/OpenAlex.Index -c Release -- explain \
  --index /datadisk/index/openalex-abstracts \
  --id 'https://openalex.org/W3111590711' --query '...'
```

`known_item.py` scores against ground-truth DOIs, so it needs no LLM and gives the same
answer every time. `--stages` prints the BM25 rank beside the reranked rank, which is what
separates "never retrieved" from "retrieved and demoted" — they need opposite fixes.
`explain` prints matched terms with their IDF and the length norm.

Two failure modes these were built to tell apart, after both were guessed at wrongly:

- **`IN INDEX but not retrieved`** is ambiguous on its own — it covers both "BM25 never
  scored it" and "we retrieved it and then dropped it in dedup".
- **A truncated, unordered query cannot prove absence.** A `LIKE '%...%' LIMIT 10` over
  millions of matches returned only commentaries, and a paper was declared missing that
  had been present all along.

## Schema changes

`LexicalIndex.SchemaVersion` is written into `index-stamp.json` beside every index, along
with a fingerprint of the source shards. Bump it when the field layout changes and every
index invalidates itself — `prepare` rebuilds rather than serving an index whose fields no
longer exist. There is no manual cleanup step and no `--force` flag.
