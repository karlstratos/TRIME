bin_file = 'data-bin/toy/train.bin'
idx_file = 'data-bin/toy/train.idx'

with open(bin_file, 'rb') as f:
    for line in f:
        print(line)

print()
with open(idx_file, 'rb') as f:
    for line in f:
        print(line)
