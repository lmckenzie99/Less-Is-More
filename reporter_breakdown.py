from datasets import load_dataset
from collections import Counter

ds = load_dataset("common-pile/caselaw_access_project_filtered", split="train", streaming=True)
ds = ds.shuffle(seed=42, buffer_size=10_000)

all_prefixes = Counter()
total = 0

for i, example in enumerate(ds):
    prefix = example["id"].split("/")[0]
    # strip trailing volume number (last underscore-separated chunk if it's numeric)
    parts = prefix.rsplit("_", 1)
    reporter = parts[0] if len(parts) == 2 and parts[1].isdigit() else prefix
    all_prefixes[reporter] += 1
    total += 1
    if i >= 50_000:
        break

federal = {"f-supp", "f-supp-2d", "f-supp-3d", "f2d", "f3d", "us", "us-app-dc"}
fed_count = sum(c for r, c in all_prefixes.items() if r in federal)

print(f"Total sampled: {total}")
print(f"Federal subset: {fed_count} ({100*fed_count/total:.1f}%)")
print(all_prefixes.most_common(20))
