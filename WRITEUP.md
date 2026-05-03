# Write-Up

> Max 2 pages. Cover the sections below with specifics — not just what you did, but why.

## Data Exploration

The knowledge base is domain-specific (Steadfast wording, resolutions, fixed category/priority mix) and historically skewed—priorities lean medium/high, categories aren’t uniform. The eval set is artificially balanced on priority (similar counts for low/medium/high, few critical) and slightly shifted on topics vs the KB, with some longer ticket bodies. So overall the data is not i.i.d. between KB and eval: the eval stresses edge urgency and boundaries more than raw ticket traffic would.

Why BM25 + down-weighting snippet labels helps:
BM25 matches lexical overlap (product terms, errors, integration names) without assuming generic embeddings, which fits short, jargon-heavy tickets and brings in relevant resolution text for the reply. Retrieved rows still carry their own category/priority headers, which don’t always match the live ticket when mixes or lengths differ—so treating those labels as soft priors (ground the facts and steps, not the urgency) avoids the model copying the wrong bucket from look-alike history while keeping concrete grounding from the KB.

## Pipeline Design Decisions

Work was split across **feature branches** (`feature/loader`, `feature/preprocess`, `feature/agent`, `feature/validate`, `feature/eval`, `feature/postprocess`, `feature/analyze`, `feature/v1` / `v2`, …) and merged on **`main`** so each pipeline stage could be tested in isolation.

I didn't have access to the assignment models, so I had to use my own
gemini-flash model. I used litellm and put emphasis on token limit to make sure the pipeline will work with most models.

I used pydentic to ensure fields validity, tenacity for retry logic.
in stage 4 I'v implemnted validations for category/prioriy/confidance values. And in stage 5 I played with few huristics, and based on the trails decided to keep 3 rules:  high_impact, low_confidence_flag, 
multi_issue_flag.

## Iteration Log

What did you try, what worked, what didn't? Include metrics across iterations.

| Iteration | Change | Category Acc | Priority Acc | Response Quality |
|-----------|--------|-------------|--------------|-------------------|
| v1 | Baseline: BM25 k≈3, prompts + validate + postprocess, Gemini Flash (`gemini-flash-latest`); full `pipeline.py --eval` (46 tickets) | **87.0%** | **58.7%** | KB proxy overlap mean **~0.116**; actionability hint **8.7%**; avg reply **~332** chars |
| v2 | Impact-first priority rubric (blast radius vs KB hints); tighter triage JSON / reply caps; same model + eval set | **84.8%** | **71.7%** | KB proxy mean **~0.126**; actionability hint **10.9%**; avg reply **~230** chars |
| v3 | Dropped six postprocess unused/bad rules; prompt-only guidance for connector misbehaviour → **medium+**, org-critical primary workload + hard failure → **critical**, dependency/planning **feature_request** gaps → **medium**; full `--eval` → `output/v3/` | **87.0%** | **78.3%** | KB proxy mean **~0.126**; priority cost mean **~0.947**; actionability hint **6.5%**; avg reply **~238** chars |

## Response Quality Metric

There is **no human judgment** in the loop, so “quality” is approximated with **cheap automatic proxies** wired in `src/evaluate/metrics/`:

1. **`kb_alignment_proxy` (primary grounding signal)** — For each ticket we re-run the same **BM25 retriever** used at inference, join the top chunks, and compute **stopword-filtered token overlap** between the **model response** and those chunks: \(|\text{resp\_tokens} \cap \text{chunk\_tokens}| / |\text{chunk\_tokens}|\). **Why:** It matches the assignment emphasis on **Steadfast-specific** phrasing from the KB (workarounds, paths, known issues) instead of generic platitudes. **Limits:** High overlap can still be **wrong advice**; shared tokens can be **generic**; overlap **punishes** concise paraphrases that are correct; it is **tied to whatever BM25 retrieved**, not to an oracle “right” passage.

2. **`actionability_hint_rate`** — **Regex** on the response for patterns like “please try”, “next steps”, “Settings”, `KB-…`, `docs.steadfast`, etc. **Why:** Quick check that the reply **pushes the customer toward a concrete action** rather than only empathy. **Limits:** Easy to **game** with boilerplate; misses good answers that don’t match the patterns; regex misses phrasing that *is* actionable.

3. **`avg_response_char_count_joined`** — Mean **length** of trimmed replies. **Why:** Sanity that we are not emitting **empty stubs** after validation/clipping (correlates weakly with substance). **Limits:** Long **≠** helpful; short can be optimal.

## What I’d Do Differently

- **Model sweep** — With more time I’d **benchmark** several models through the same pipeline.
- **LLM-judge quality** — Add thin **human labels** or a **stronger model-as-judge** on a fixed slice (correctness, safety, tone) so we are not optimizing only **kb_alignment_proxy** and regex hints.
- **Retrieval** — Try **hybrid BM25 + embeddings**.
- **Calibration** — set review thresholds from data, not a single magic **0.6**.
- **Multi-issue routing** — Today we mostly **flag** `ambiguous_category`; with more time we could refine or add a decision stage for ambiguous cases.

