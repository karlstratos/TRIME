# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn

from typing import List


class AdaptiveInput(nn.Module):

    def __init__(
            self,
            vocab_size: int,  # 33280  (wiki-2)
            padding_idx: int,  # 1
            initial_dim: int,  # 410
            factor: float,  # 1.0
            output_dim: int,  # 410
            cutoff: List[int],  # [2000, 6000]
    ):
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]  # now cutoff = [2000, 6000, 33280]
        else:
            assert vocab_size == cutoff[
                -1], 'cannot specify cutoff larger than vocab size'

        self.cutoff = cutoff
        self.embedding_dim = output_dim
        self.padding_idx = padding_idx

        self.embeddings = nn.ModuleList()  # Each embeddings[i] used to embed the i-th partition of vocab (i=0: most frequent+special toks, i=1: next frequent, i=2: the rest)
        for i in range(len(self.cutoff)):
            prev = self.cutoff[i - 1] if i > 0 else 0
            size = self.cutoff[i] - prev
            dim = int(initial_dim // (factor ** i))
            seq = nn.Sequential(
                nn.Embedding(size, dim, self.padding_idx),
                nn.Linear(dim, output_dim, bias=False)
            )
            self.embeddings.append(seq)
            self.padding_idx = None
        self.padding_idx = padding_idx

        #print(self.embeddings)
        #ModuleList((0): Sequential((0): Embedding(2000, 410, padding_idx=1) (1): Linear(in_features=410, out_features=410, bias=False))
        #           (1): Sequential((0): Embedding(4000, 410) (1): Linear(in_features=410, out_features=410, bias=False))
        #           (2): Sequential((0): Embedding(27280, 410) (1): Linear(in_features=410, out_features=410, bias=False)))

        def init_weights(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=m.weight.shape[1] ** -0.5)
                nn.init.constant_(m.weight[padding_idx], 0)
            elif hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def weights_for_band(self, band: int):
        return self.embeddings[band][0].weight, self.embeddings[band][1].weight

    def forward(self, input: torch.Tensor):  # input: (60, 150) == (bsz, len)
        # result: constant (60, 150, 410)
        result = self._float_tensor.new(input.shape + (self.embedding_dim,))
        for i in range(len(self.cutoff)):  # i=0, 1, 2
            mask = input.lt(self.cutoff[i])  # Word rank < 2000, 6000, 33280?
            if i > 0:
                mask.mul_(input.ge(self.cutoff[i - 1]))  # Rank >= 2000, 6000?
                chunk_input = input[mask] - self.cutoff[i - 1]  # Align embedding indices
            else:
                chunk_input = input[mask]

            if mask.any():
                result[mask] = self.embeddings[i](chunk_input)

            # i=0: mask contains 6919 (out of 9000=60*150) trues
            #          chunk_input (6919,)
            #          result[rank < 2000] = embeddings[0]([corresponding words])  embeds to 2000
            #
            # i=1: mask contains 1055 trues
            #          chunk_input (1055,)
            #          result[rank < 6000 ^ rank > 2000] = embeddings[1]([corresponding words] - 2000)  embeds to middle 4000
            #
            # i=2: mask contains 1026 trues
            #          chunk_input (1055,)
            #          result[rank < vocab_size ^ rank > 6000] = embeddings[2]([corresponding words] - 6000)  embeds to last 4000
        return result
