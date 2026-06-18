from datasets import load_dataset
from collections import Counter

ds = load_dataset("common-pile/caselaw_access_project_filtered", split="train", streaming=True)
ds = ds.shuffle(seed=42, buffer_size=10_000)

prefixes = Counter()
for i, example in enumerate(ds):
    prefix = example["id"].split("_")[0]
    prefixes[prefix] += 1
    if i > 20_000:
        break


print(prefixes.most_common(20))
