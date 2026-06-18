from datasets import load_dataset
from collections import Counter

ds = load_dataset("common-pile/caselaw_access_project_filtered", split="train", streaming=True)

prefixes = Counter()
for i, example in enumerate(ds):
    prefix = example["id"].split("/")[0]
    prefixes[prefix] += 1
    if i > 5000:
        break


print(prefixes.most_common(20))
