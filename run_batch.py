# run_batch.py
# Runs a single query 25 times for both prompt variants at a given precision
# Outputs to mistral/<precision>/query_<n>/unconstrained/ and /constrained/

import argparse
import hashlib
import os
import subprocess


def output_hash(text):
    return hashlib.md5(text.encode()).hexdigest()[:8]


def run_single(model, precision, collection, prompt_variant, query, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # count existing files to avoid overwriting
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".txt")])
    if existing >= 25:
        print(f"  Skipping {output_dir}, already has {existing} runs")
        return ""
    outfile = os.path.join(output_dir, f"run_{existing+1:02d}.txt")

    result = subprocess.run(
        [
            "python",
            "pipeline.py",
            "--model",
            model,
            "--precision",
            precision,
            "--collection",
            collection,
            "--prompt",
            prompt_variant,
            "--query",
            query,
        ],
        capture_output=True,
        text=True,
    )
    output = ""
    capture = False
    for line in result.stdout.split("\n"):
        if "--- Output ---" in line:
            capture = True
            continue
        if line.startswith("Latency:"):
            capture = False
        if capture:
            output += line + "\n"

    with open(outfile, "w") as f:
        f.write(output.strip())

    print(f"  Wrote {outfile}")
    return output.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-num", type=int, required=True, help="query number e.g. 1"
    )
    parser.add_argument("--query", required=True, help="the legal query string")
    parser.add_argument(
        "--precision", required=True, choices=["fp16", "int8", "int4", "gptq", "awq"]
    )
    parser.add_argument("--collection", default="federal")
    parser.add_argument("--runs", type=int, default=25)
    args = parser.parse_args()

    model = "mistral"

    for variant in ["constrained", "unconstrained"]:
        output_dir = os.path.join(
            "mistral",
            args.precision,
            f"query_{args.query_num:02d}",
            variant,
        )
        existing = (
            len([f for f in os.listdir(output_dir) if f.endswith(".txt")])
            if os.path.exists(output_dir)
            else 0
        )
        remaining = args.runs - existing

        if remaining <= 0:
            print(f"  Skipping {variant}, already has {existing} runs")
            continue

        print(f"\nRunning {remaining} runs: {variant} | {
              args.precision} | query_{args.query_num:02d}")

        for i in range(remaining):
            print(f"  Run {existing+i+1}/{args.runs}...")
            run_single(
                model=model,
                precision=args.precision,
                collection=args.collection,
                prompt_variant=variant,
                query=args.query,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()
