# Write-Up

> Max 2 pages. Cover the sections below with specifics — not just what you did, but why.

## Data Exploration

How did you approach the knowledge base and eval set? What did you learn and how did it inform your pipeline design?

## Pipeline Design Decisions

How did you approach each stage? What model did you choose and why? How did you select context for the LLM? What validation and heuristic strategies did you implement?

## Iteration Log

What did you try, what worked, what didn't? Include metrics across iterations.

| Iteration | Change | Category Acc | Priority Acc | Response Quality |
|-----------|--------|-------------|--------------|-------------------|
| v1 | Baseline: BM25 k≈3, prompts + validate + postprocess, Gemini Flash (`gemini-flash-latest`); full `pipeline.py --eval` (46 tickets) | **87.0%** | **58.7%** | KB proxy overlap mean **~0.116**; actionability hint **8.7%**; avg reply **~332** chars |
| v2 | Impact-first priority rubric (blast radius vs KB hints); tighter triage JSON / reply caps; default `LLM_MAX_TOKENS` 2048 + `LLM_MAX_RESPONSE_CHARS`; same model + eval set | **84.8%** | **71.7%** | KB proxy mean **~0.126**; actionability hint **10.9%**; avg reply **~230** chars |
| v3 | Dropped six postprocess rules (category reroutes, data-loss bump, enterprise low→medium); prompt-only guidance for connector misbehaviour → **medium+**, org-critical primary workload + hard failure → **critical**, dependency/planning **feature_request** gaps → **medium**; `output/v3/` full `--eval` | **87.0%** | **78.3%** | KB proxy mean **~0.126**; priority cost mean **~0.947**; actionability hint **6.5%**; avg reply **~238** chars |

## Response Quality Metric

What metric did you use to evaluate response quality? Why did you choose it? What are its limitations?

## What I'd Do Differently

With more time or in a production setting, what would you change?
