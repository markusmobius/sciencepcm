# Evaluation

Three questions get asked of these services, and each has its own tool. Everything in
`eval/` runs against the current code; experiments that depended on the retired dense
index and on BioASQ qrels have been deleted, and what they taught is recorded in
[retrieval.md](retrieval.md#rejected).

| question | tool | needs an LLM |
| --- | --- | --- |
| is *this* paper returned, and at what rank | `known_item.py` | no |
| are these results any good | `mcp_run.py` → `judge.py` | yes |
| what do two configurations return, side by side | `compare_runs.py` | no |
| does the corpus contain what we think | `s2_probe.py` | no |
| does an LLM answer better with the tools | `llm_ab.py` | yes |

## Known-item: is the right paper returned

Scored against ground-truth DOIs written down in advance, so it is instant, free and
deterministic. This is the first thing to run after any ranking or schema change.

```bash
$LAB eval/known_item.py --endpoint https://www.openalexmcp.econlabs.org/mcp \
  --questions eval/questions-landmark.jsonl --stages --show 5
```

`--stages` prints the BM25 rank beside the reranked rank, which separates "never
retrieved" from "retrieved then demoted" — opposite fixes. `--show N` prints what
outranked the target, with `type` and `cited_by_count`. On a miss it looks the paper up
by ID and reports `IN INDEX but not retrieved` or `NOT IN INDEX`, so a corpus problem
never gets mistaken for a ranking problem.

Baselines: `questions-landmark.jsonl` (long news-style) and
`questions-landmark-short.jsonl` (terse) cover four papers with known DOIs.

## Judged: are the results any good

`mcp_run.py` drives the live endpoint and writes a run file; `judge.py` pools results
across systems so each document is graded once, then scores nDCG@10, mean grade,
fraction relevant, hit@1, hit@k and MRR.

```bash
$LAB eval/mcp_run.py --endpoint $OA --token "$OPENALEX_TOKEN" \
  --questions eval/questions-newsmatch.jsonl --tool search_openalex --out runs/v3.jsonl

$LAB eval/judge.py --run runs/v3.jsonl --prompt-file eval/prompts/newsmatch.txt \
  --unit paper --workers 32
```

Two things to hold onto. **Graded nDCG normalises against the pool**, so absolute values
shift when the set of systems changes — only compare within one judge run. And **mean
grade and "fraction ≥2" are blind to ordering**: on known-item work they showed two
rerankers as identical while hit@1 differed by 17 points. For known-item use hit@1 and
MRR; for topical work use nDCG.

There is no local grade cache; the LLM server caches on prompt and model.

## The rest

`compare_runs.py` prints run files side by side with titles, for reading rather than
scoring — the fastest way to see *how* two configurations differ. `s2_probe.py` measures
what fraction of a defined cohort another source could supply, and produced the decision
in [openalex.md](openalex.md#why-there-is-no-semantic-scholar-merge).

---

# Evaluating an LLM with and without ScienceMCP

The rest of this document is the paired, end-to-end protocol: the same LLM answering the
same neuroscience questions, with and without the tools.

- **Control:** the LLM has no ScienceMCP tools and no other retrieval tools.
- **Treatment:** the LLM can use the ScienceMCP tools.

Keep the model, model version, system prompt, decoding settings, question order and
scoring procedure identical. Tool availability should be the only experimental
difference. This measures whether giving an agent access to ScienceMCP improves its
answers; it is different from evaluating the retriever alone.

## 1. Choose the experiment

Two related experiments answer different questions. Record which one is being run.

### A. Tool-availability experiment (recommended)

Give both conditions the same prompt. In the treatment condition, expose ScienceMCP;
in the control, expose no tools. Let the model decide whether and how to search.

This tests the product as users experience it, including tool selection, query writing,
result interpretation and final answer synthesis. A treatment answer counts even when
the model elects not to call a tool. Report the percentage of treatment runs that made
at least one ScienceMCP call.

### B. Forced-retrieval experiment

Require the treatment agent to search ScienceMCP before answering. The control still
answers without retrieval. This isolates the value of retrieved evidence more clearly,
but no longer measures whether the agent chooses tools correctly.

Run A as the primary experiment and B as a diagnostic if the model rarely calls tools.
Do not mix the two designs in one aggregate score.

## 2. Configure LlmClient and `sciencemcp`

The evaluation uses `newsprinceton-llmclient`. From the repository root in PowerShell,
create the evaluation environment with `uv`:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements/eval.txt
```

On the corporate network pypi.org is blocked, so point uv at the proxy first. uv does
not read `pip.ini`, so this must be set in the environment:

```powershell
$env:UV_DEFAULT_INDEX = 'https://packagefeedproxy.microsoft.io/pypi/simple/'
$env:UV_INDEX_URL = 'https://packagefeedproxy.microsoft.io/pypi/simple/'
```

The commands below invoke `.venv` directly, so activation is unnecessary and the
PowerShell execution policy does not need to be changed.

Set the LlmClient environment variables required by your LLM server:

```powershell
$env:LLM_SERVER_URL = '<server URL>'
$env:LLM_USER_CODE = '<user code>'
$env:LLM_CACHE = "$PWD\.llm-cache"
```

These assignments apply to the current PowerShell session. Set them again in a new
terminal. Do not put credentials in the repository.

The named LlmClient tool is exactly `sciencemcp`. Enable it in the treatment chat with:

```python
Chat(responseSchema=None, model=MODEL, tools=["sciencemcp"])
```

Create the control chat without the `tools` argument:

```python
Chat(responseSchema=None, model=MODEL)
```

The LLM server's tool registry connects `sciencemcp` to the MCP service. The evaluation
script does not implement MCP transport and does not pass the ScienceMCP bearer token.
Before the experiment, confirm with the LlmClient service owner that the registered
`sciencemcp` tool points to this production endpoint:

```text
https://www.sciencemcp.econlabs.org/mcp
```

Check that the service is reachable:

```powershell
Invoke-RestMethod https://www.sciencemcp.econlabs.org/health | ConvertTo-Json
```

The health response should report nonzero `documents` and `passages`. Authentication
between the LLM server and ScienceMCP belongs in the server-side tool registration;
never put that bearer token in evaluation prompts or result files.

After connecting, confirm that the treatment agent sees these tools:

| Tool | Intended use |
| --- | --- |
| `search_literature` | Topic breadth and finding relevant papers from abstracts |
| `search_full_text` | Methods, parameters, procedures and specific findings |
| `get_passage_context` | Text immediately around a full-text passage |
| `get_paper` | Complete abstract and bibliographic metadata for one paper |
| `corpus_stats` | Corpus coverage and retrieval caveats |

The control agent must have none of these tools. Disable web search, browsing, code
execution, file search and any provider-side retrieval in both conditions unless those
capabilities are explicitly part of a separate experiment.

## 3. Build a held-out question set

Use at least 50 questions for an exploratory comparison and 100 or more for a result
that will be shared broadly. Define the questions before inspecting treatment answers.
Use a stable `query_id` and assign each question to a stratum:

- **Literature breadth:** which papers or findings exist on a topic.
- **Methods:** concentrations, sample sizes, apparatus, parameters and procedures.
- **Specific results:** what a study found, with enough detail to verify the claim.
- **Synthesis:** conclusions requiring evidence from multiple papers.
- **Coverage checks:** questions for which the corpus may have insufficient evidence.

Balance broad and methods questions. Most papers in the ScienceMCP corpus have **no**
full text, and full text covers 2019-2025 only, so an evaluation made entirely of
methods questions tests both answer quality and this coverage ceiling. Call
`corpus_stats` for the measured figure rather than assuming one; the server reports the
number of papers with full text against the number of indexed papers. Include some
questions whose best answer is qualified uncertainty rather than a confident number.

The existing [`eval/questions-methods.jsonl`](eval/questions-methods.jsonl) can seed the
methods stratum. It was created for retrieval evaluation and is not by itself a final
answer-quality benchmark: several questions admit different valid protocols in
different papers. A domain expert should add reference evidence and acceptable answer
ranges before using it for answer scoring.

Recommended question format:

```json
{"query_id":"q001","stratum":"methods","query_text":"What concentrations of TTX are commonly used to block sodium channels in acute brain-slice recordings?"}
```

Keep reference answers and grading notes in a separate file hidden from the answering
model. For each question, record the expected key points, acceptable variability and
source passages or papers. Do not use the final evaluation set while tuning prompts.

## 4. Freeze the answering protocol

Pin the exact model identifier and provider version. Use deterministic decoding when
available (`temperature=0`); otherwise run at least three replicates and treat question,
not individual generation, as the unit of analysis.

Start a fresh conversation for every question and condition. Never ask the control and
treatment versions in the same conversation: retrieved text or tool state can leak
between them. Randomize whether control or treatment runs first for each question.

Use the same system prompt in both conditions for experiment A:

```text
You are answering a neuroscience research question. Use any tools available to you
when they would improve the answer. Base factual claims on identifiable evidence.
Distinguish findings from individual studies from general consensus. Cite sources by
DOI, PMID or PMCID when available. Do not invent citations. If the available evidence
is insufficient or protocols vary across studies, say so explicitly. Give a concise
answer followed by a Sources section.
```

For forced-retrieval experiment B, add this sentence only to the treatment prompt:

```text
Before answering, use ScienceMCP to search for evidence relevant to this question.
```

This prompt difference is intentional in experiment B and must be reported.

Set a tool-call budget in advance, for example 8 calls per question, and the same answer
token limit in both conditions. A budget prevents one difficult query from consuming
unbounded time and makes latency and cost comparable.

### Minimal LlmClient A/B check

This adapts the known LlmClient named-tool pattern. Both chats receive the same system
and user messages; only `tools=["sciencemcp"]` differs. Pin `MODEL` to the exact model
version being evaluated rather than relying on the client default.

The complete runnable harness is [`eval/llm_ab.py`](eval/llm_ab.py). Run the primary
tool-availability experiment with:

```powershell
.\.venv\Scripts\python.exe eval\llm_ab.py --questions eval\questions-methods.jsonl --model '<exact LlmClient model version>' --out eval\results\answers.jsonl
```

An interrupted run can continue without duplicating completed pairs:

```powershell
.\.venv\Scripts\python.exe eval\llm_ab.py --questions eval\questions-methods.jsonl --model '<exact LlmClient model version>' --out eval\results\answers.jsonl --resume
```

For diagnostic experiment B, use a separate output path and add `--force-tool`. For a
nondeterministic model, add `--repeats 3`. Do not combine primary and forced-retrieval
records in the same output file.

```python
import asyncio
from time import perf_counter

from LlmClient.LlmLib import LlmFactory
from LlmClient.Models import Chat


MODEL = "replace-with-an-exact-model-version"
SYSTEM_PROMPT = """You are answering a neuroscience research question. Use any tools
available to you when they would improve the answer. Base factual claims on
identifiable evidence. Cite sources by DOI, PMID or PMCID when available. Do not
invent citations. If evidence is insufficient or protocols vary, say so explicitly.
Give a concise answer followed by a Sources section."""


async def ask(client, question: str, condition: str) -> dict:
  tools = ["sciencemcp"] if condition == "sciencemcp" else None
  chat = Chat(responseSchema=None, model=MODEL, tools=tools)
  chat.AddSystemMessage(SYSTEM_PROMPT)
  chat.AddUserMessage(question)

  started = perf_counter()
  output = await client.Ask(chat, tags=["sciencepcm-eval", condition])
  latency_ms = round((perf_counter() - started) * 1000)

  return {
    "condition": condition,
    "model": MODEL,
    "answer": output.answer.ChatAnswer if output.answer else None,
    "latency_ms": latency_ms,
    "error": str(output.error) if output.error is not None else None,
  }


async def main():
  factory = LlmFactory()
  client = await factory.create_client()
  try:
    question = "What concentration of TTX is used to block sodium channels in acute brain-slice recordings?"
    control = await ask(client, question, "no_tool")
    treatment = await ask(client, question, "sciencemcp")
    print(control)
    print(treatment)
  finally:
    await client.Close()


if __name__ == "__main__":
  asyncio.run(main())
```

This is a connectivity check, not the full evaluation harness. The full run must read a
held-out JSONL question set, randomize condition order per question, start each condition
from a new `Chat`, and write one result record immediately after each answer. Reusing one
LlmClient connection is fine; reusing one `Chat` is not.

Do not copy the OpenAlex example's treatment-only system instruction into primary
experiment A. Adding "Use the sciencemcp tool" only to treatment changes both the tool
availability and prompt. That wording is appropriate only for forced-retrieval
experiment B.

## 5. Capture complete run records

Save one JSONL record per question and condition. Preserve raw answers and tool traces;
do not retain only an aggregate score.

```json
{
  "run_id": "2026-08-26-model-x-r1",
  "query_id": "q001",
  "condition": "sciencemcp",
  "model": "provider/model-version",
  "answer": "...",
  "tool_calls": [
    {"name": "search_full_text", "arguments": {"query": "...", "limit": 10}}
  ],
  "citations": ["PMID:12345678", "10.1000/example"],
  "latency_ms": 8420,
  "input_tokens": 6400,
  "output_tokens": 510,
  "error": null
}
```

Capture at least:

- Exact model and model-version identifier.
- Exact system and user prompts, or their versioned hashes.
- Condition and randomized execution order.
- Final answer and all tool names/arguments/results.
- Input/output tokens, wall-clock latency, errors and provider cost when available.

Tool results can be large. They may be stored in a separate trace file referenced by
hash, but they must remain available for citation verification and debugging.

## 6. Grade answers blind

Hide condition labels, tool traces and execution order from graders. Randomly label
answers `A` and `B`; do not consistently show control first. Prefer two neuroscience
graders for a shared result and adjudicate substantial disagreements.

Use the following preregistered rubric for each answer:

| Dimension | Score | Definition |
| --- | ---: | --- |
| Factual correctness | 0-4 | From mostly wrong/hallucinated to fully correct and appropriately qualified |
| Completeness | 0-2 | Covers none, some, or all essential reference points |
| Evidence support | 0-2 | Claims are unsupported, partly supported, or supported by cited evidence |
| Calibration | 0-2 | Overconfident, partly qualified, or appropriately states limits/variation |

Also record objective citation measures:

- **Citation validity:** cited DOI/PMID/PMCID resolves to a real paper.
- **Citation correctness:** that paper supports the claim attached to it.
- **Citation coverage:** fraction of externally verifiable claims supported by a citation.
- **Fabricated citation rate:** fraction of cited identifiers that do not resolve.

A citation is not correct merely because it exists or is topically related. Verify the
claim against the retrieved passage, abstract or paper. For methods claims, prefer
full-text evidence; an abstract that does not state the parameter is not support.

An LLM judge may be used for inexpensive first-pass grading, but give it the question,
reference answer and reference evidence, and instruct it to judge only those materials.
Do not let the same model answer and judge when avoidable. Manually audit a random
sample plus every large control-treatment disagreement. The repository's
[`eval/judge.py`](eval/judge.py) grades retrieval hits, not synthesized answers, so it
cannot be used unchanged for this experiment.

## 7. Analyze as paired data

Pair answers by `query_id`. For each rubric dimension, compute the per-question change:

```text
delta = treatment score - control score
```

Report:

- Mean and median paired delta with a 95% bootstrap confidence interval.
- Treatment wins, ties and losses.
- Scores by question stratum, especially breadth versus methods.
- Citation validity, correctness, coverage and fabrication rates.
- Tool-use rate and tools called in the treatment condition.
- Median and p95 latency, token usage and cost by condition.
- Failure and timeout rates.

Bootstrap whole questions, preserving the control-treatment pair, with at least 10,000
resamples. If multiple generations were made for each condition, average replicates
within each question before computing the primary paired result. Do not treat repeated
generations as independent questions.

Predeclare one primary outcome, such as factual correctness, before grading. Treat the
other dimensions and per-stratum results as secondary diagnostics. Publish confidence
intervals and effect sizes rather than relying only on a significance threshold.

## 8. Interpret failures

Use tool traces to classify treatment failures after blind scoring:

1. **No tool call:** the agent failed to recognize a retrieval need.
2. **Wrong tool:** for example, abstract search used for a methods parameter.
3. **Poor query:** the search omitted terminology used in the literature.
4. **Retrieval miss:** relevant evidence was not returned.
5. **Coverage miss:** relevant full text is not in the corpus.
6. **Synthesis error:** useful evidence was returned but misread or ignored.
7. **Citation error:** the answer cited a paper that did not support its claim.

This decomposition matters. A small end-to-end gain can reflect weak retrieval, weak
tool use by the chosen LLM, or both. The existing scripts under `eval/` evaluate
retrieval separately and can help distinguish these causes.

## Minimum report checklist

- [ ] Exact model/version and date recorded.
- [ ] Same model settings and prompt in both primary conditions.
- [ ] All non-ScienceMCP retrieval disabled in both conditions.
- [ ] Fresh conversation for every question and condition.
- [ ] Question order randomized and grading labels blinded.
- [ ] Held-out questions cover both abstracts and full-text use cases.
- [ ] Tool traces, latency, token usage and errors retained.
- [ ] Citations checked for both existence and claim support.
- [ ] Paired deltas and bootstrap confidence intervals reported.
- [ ] Corpus coverage limitations included in conclusions.