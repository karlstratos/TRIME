# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter

from fairseq.tokenizer import tokenize_line
import torch


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Binarizer:
    @staticmethod

    # filename=data-bin/wikitext-2/raw_data/wikitext-2/wiki.train.tokens
    # dict=vocab=src_dic
    # consumer= lambda t: ds.add_item(t),  # ds will write to the binary file train.bin (add_item)
    #def tokenize_line(line):
    #line = SPACE_NORMALIZER.sub(" ", line)
    #line = line.strip()
    #return line.split()
    # offset=0
    # end=offsets[1]=0
    def binarize(
        filename,
        dict,
        consumer,
        tokenize=tokenize_line,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
        already_numberized=False,
    ):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                if already_numberized:
                    id_strings = line.strip().split()
                    id_list = [int(id_string) for id_string in id_strings]
                    if reverse_order:
                        id_list.reverse()
                    if append_eos:
                        id_list.append(dict.eos())
                    ids = torch.IntTensor(id_list)
                else:
                    ids = dict.encode_line(
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,  # Counter above
                        append_eos=append_eos,  # The only addition: append </s>
                        reverse_order=reverse_order,
                    )
                    #print('line:', line)  # "The series has also been criticized for its release model in contrast to ... on which songs could be made forward @-@ compatible ."
                    #print('tokenize:', tokenize)  # <function tokenize_line at 0x7f39137d90e0>
                    #print('append_eos:', append_eos)  # True
                    #print('reverse_order:', reverse_order)  # False

                    # print('ids:', ids)  # int32 tensor([   15,   109,    52,    46,    53,  1760,    21...])
                    # ids have varying lengths (2, 87, 328) for different lines
                    # print(' '.join([dict.symbols[id] for id in ids]))  # The series has also been criticized for its release model in contrast ... on which songs could be made forward @-@ compatible . </s>

                nseq += 1
                ntok += len(ids)
                consumer(ids)  # This writes to train.bin
                line = f.readline()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }

    @staticmethod
    def binarize_alignments(filename, alignment_parser, consumer, offset=0, end=-1):
        nseq = 0

        with open(filename, "r") as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
                line = f.readline()
        return {"nseq": nseq}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets
