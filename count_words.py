import sys

from collections import Counter


fname = sys.argv[1]
counter = Counter()

with open(fname) as f:
    for line in f:
        toks = line.split()
        for tok in toks:
            counter[tok] += 1

print(len(counter))
print(counter.most_common(10))
print(counter.most_common()[-10:])
