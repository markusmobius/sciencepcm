# ScienceMCP

Neuroscience literature, answering what a study actually did and found. Two tiers over
the same corpus, both BM25 followed by a `bge-reranker-v2-m3` cross-encoder.

| tier | contents | tool |
| --- | --- | --- |
| abstracts | 5,298,493 papers | `search_literature` |
| full text | 14,973,319 passages of ~300 words, 2019-2025 | `search_full_text` |

Public at `https://www.sciencemcp.econlabs.org/mcp`, port 8080 locally.

## Which tier answers what

Full text is not a refinement of the abstract tier, it is a different service. Measured
on a methods benchmark with an LLM judge:

| tier | nDCG@10 | useful results |
| --- | --- | --- |
| full text | 0.9317 | 89% |
| abstracts | 0.2472 | 20.7% |

Abstracts state what a paper is about; they do not state sample sizes, drug
concentrations or procedures. So `search_full_text` is the default for anything
methodological, and `search_literature` is for breadth — which papers exist on a topic.

**Full-text coverage is only ~23% of 2019-2025 paper-like works, and nothing before
2019.** Absence from the full-text tier says nothing about the literature. `corpus_stats`
reports the measured coverage, and the tool descriptions tell the model to say so.

## Tools

- `search_literature` — abstracts. Takes `query`, `limit`, `author`, `journal`, `sort`,
  `yearMin`, `yearMax`, `fast`.
- `search_full_text` — passages, with `section` to restrict to Methods, Results and so
  on, and `maxPerArticle` so one thorough paper cannot fill the result set.
- `get_passage_context` — the passages either side of a hit, for when one starts
  mid-argument.
- `get_paper` — the complete abstract and all bibliographic metadata for one article key.
- `corpus_stats` — what the corpus covers, to be called before concluding anything from
  an absence of results.

`author` matches exactly, in `Surname, Given` form, against one indexed term per name —
so it lists a researcher's papers rather than matching two common words. With `author` or
`journal` set, `query` may be omitted and `sort=citations` or `sort=year` is usually what
you want; reranking is skipped for a non-relevance sort, because a browse has no
relevance signal to rerank on.

## Ranking

Shared with the OpenAlex service — see [retrieval.md](retrieval.md) for the fielded BM25F
schema, the citation prior and the measurements behind them.

Retracted papers are included and flagged `is_retracted`, rather than hidden. A retracted
paper is often exactly what a question is about, and silently dropping it would make the
corpus lie about itself.

## Building it

```bash
source ~/mcp/env.sh
bash tools/sciencemcp-a100.sh prepare      # corpus, reranker, both indexes
bash tools/sciencemcp-a100.sh serve
bash tools/sciencemcp-a100.sh check        # paths, sizes, index schema version
```

`prepare` is idempotent: the blob store transfers only what differs, the reranker is
exported only if absent, and each index rebuilds only when its `index-stamp.json` no
longer matches the source shards and schema version.

The corpus comes from `\\nerds21\sciencepcm\dataset` via the blob store — 402,708 JATS
XML files ingested to 6,028,782 chunks with zero failures, sectioned as Methods 25.0%,
Results 18.6%, Discussion 14.0%, Unknown 11.8%, FigureCaption 10.6%, Introduction 9.3%,
TableCaption 3.8%.

## Is there money left on the table

Probably, but not in retrieval. Adding a dense leg to a reranked pipeline changed nothing
measurable (0.7713 with rerank alone against 0.7647 fused), and BM25 beat MedCPT dense
retrieval outright (0.2255 vs 0.117 nDCG@10 on BioASQ).

The gap is corpus coverage. Full text scores 0.9317 against abstracts' 0.2472, and only
~23% of papers have it. Expanding full-text coverage beats any reranker work by an order
of magnitude — see the S2ORC note in [openalex.md](openalex.md#why-there-is-no-semantic-scholar-merge).

The cheap second thing: the tool description currently tells the model to re-query with
different terminology when results look thin. That is a recall gap papered over with
instructions, and an LLM query-expansion pass server-side would close it without a
reindex.
