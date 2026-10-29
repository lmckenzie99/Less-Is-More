#!/usr/bin/env python3
# judge.py
# LLM-as-judge evaluation of RAG pipeline outputs using Claude Haiku.
# Scores each run on 5 dimensions: faithfulness, context_leakage,
# citation_accuracy, answer_relevance, repetition.
# Checkpoints to judge_results.csv so restarts pick up where they left off.

import anthropic
import csv
import json
import os
import time

BASE_DIR = "/home/liam/Documents/fellow-result"
OUTPUT_CSV = os.path.join(BASE_DIR, "judge_results.csv")

MODELS = ["mistral", "saul", "llama"]
PRECISIONS = ["fp16", "int8", "int4", "gptq", "awq"]
VARIANTS = ["constrained", "unconstrained"]

FIELDNAMES = [
    "model", "precision", "query_num", "variant", "run",
    "faithfulness", "context_leakage", "citation_accuracy",
    "answer_relevance", "repetition", "error",
]

SYSTEM_PROMPT = """You are an expert legal AI evaluator. You will be given:
1. A legal query
2. Retrieved context (source documents used for generation)
3. A generated legal memo

Score the memo on these five dimensions. Respond ONLY with a JSON object, no preamble or explanation.

Scoring criteria:

faithfulness (0-3):
  3 = All claims directly supported by retrieved context
  2 = Most claims supported, minor unsupported details
  1 = Some claims supported but significant unsupported content
  0 = Claims contradict or ignore retrieved context

context_leakage (0-2):
  0 = No verbatim reproduction of retrieved text
  1 = Some phrases copied from context
  2 = Large passages reproduced verbatim from context

citation_accuracy (0-3):
  3 = All citations present in retrieved context
  2 = Most citations from context, minor deviations
  1 = Mix of context citations and external citations
  0 = Citations not from retrieved context or no citations when expected

answer_relevance (0-3):
  3 = Directly and fully addresses the legal query
  2 = Mostly addresses the query with minor gaps
  1 = Partially addresses the query
  0 = Does not address the query

repetition (0-2):
  0 = No repetition
  1 = Some repeated phrases or sentences
  2 = Significant looping or repeated content blocks

Respond with exactly this JSON structure:
{
  "faithfulness": <int>,
  "context_leakage": <int>,
  "citation_accuracy": <int>,
  "answer_relevance": <int>,
  "repetition": <int>
}"""


def build_cached_context_block(query, context):
    """Build the cacheable user turn prefix: query + context. Marked for caching."""
    return {
        "type": "text",
        "text": f"QUERY:\n{query}\n\nRETRIEVED CONTEXT:\n{context}\n\n",
        "cache_control": {"type": "ephemeral"},
    }


def build_memo_block(memo):
    return {
        "type": "text",
        "text": f"GENERATED MEMO:\n{memo}",
    }


def load_completed(csv_path):
    """Returns set of (model, precision, query_num, variant, run) already scored."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((
                row["model"], row["precision"],
                row["query_num"], row["variant"], row["run"]
            ))
    return completed


def load_queries(base_dir):
    """Load queries.txt into a dict: query_num (str) -> query text."""
    queries_path = os.path.join(base_dir, "queries.txt")
    queries = {}
    if os.path.exists(queries_path):
        with open(queries_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    queries[f"{i+1:02d}"] = line
    return queries


def score_run(client, cached_context_block, memo):
    """Call Haiku and return scores dict or raise on failure.
    cached_context_block is reused across all 25 runs for a query/variant,
    hitting the prompt cache after the first call.
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    cached_context_block,
                    build_memo_block(memo),
                ],
            }
        ],
    )
    text = response.content[0].text.strip()
    # strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    queries = load_queries(BASE_DIR)
    completed = load_completed(OUTPUT_CSV)

    print(f"Loaded {len(queries)} queries.")
    print(f"Already scored: {len(completed)} runs.")

    write_header = not os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for model in MODELS:
            for precision in PRECISIONS:
                base = os.path.join(BASE_DIR, model, precision)
                if not os.path.isdir(base):
                    continue

                for query_dir in sorted(os.listdir(base)):
                    if not query_dir.startswith("query_"):
                        continue
                    query_num = query_dir.replace("query_", "")
                    query_text = queries.get(query_num, "")

                    for variant in VARIANTS:
                        variant_dir = os.path.join(base, query_dir, variant)
                        if not os.path.isdir(variant_dir):
                            continue

                        context_file = os.path.join(variant_dir, "context.txt")
                        context_text = ""
                        if os.path.exists(context_file):
                            with open(context_file) as f:
                                context_text = f.read()

                        # build once, reused across all 25 runs -- hits cache after first call
                        cached_context_block = build_cached_context_block(query_text, context_text)

                        run_files = sorted([
                            f for f in os.listdir(variant_dir)
                            if f.startswith("run_") and f.endswith(".txt")
                        ])

                        for run_file in run_files:
                            run_num = run_file.replace("run_", "").replace(".txt", "")
                            key = (model, precision, query_num, variant, run_num)

                            if key in completed:
                                continue

                            run_path = os.path.join(variant_dir, run_file)
                            with open(run_path) as f:
                                memo = f.read()

                            print(f"Scoring {model}/{precision}/query_{query_num}/{variant}/{run_file}...", end=" ", flush=True)

                            row = {
                                "model": model,
                                "precision": precision,
                                "query_num": query_num,
                                "variant": variant,
                                "run": run_num,
                                "error": "",
                            }

                            try:
                                scores = score_run(client, cached_context_block, memo)
                                row.update(scores)
                                print(f"F={scores['faithfulness']} CR={scores['citation_accuracy']} AR={scores['answer_relevance']} R={scores['repetition']} CL={scores['context_leakage']}")
                            except Exception as e:
                                row.update({
                                    "faithfulness": "",
                                    "context_leakage": "",
                                    "citation_accuracy": "",
                                    "answer_relevance": "",
                                    "repetition": "",
                                    "error": str(e),
                                })
                                print(f"ERROR: {e}")

                            writer.writerow(row)
                            csvfile.flush()

                            # rate limit: ~50 req/min safe for Haiku
                            time.sleep(1.2)

    print(f"\nDone. Results at {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
