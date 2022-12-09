# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import operator
import functools

import torch
import torch.nn.functional as F
from torch import nn


class TiedLinear(nn.Module):
    def __init__(self, weight, transpose):
        super().__init__()
        self.weight = weight
        self.transpose = transpose

    def forward(self, input):
        return F.linear(input, self.weight.t() if self.transpose else self.weight)


class TiedHeadModule(nn.Module):
    def __init__(self, weights,  # (tensor(2000,410), tensor(410,410))
                 input_dim,  # 410
                 num_classes):  # 2
        super().__init__()
        tied_emb, _ = weights  # tensor(2000,410) (1st 2000 embs)
        self.num_words, emb_dim = tied_emb.size()

        self.word_proj = TiedLinear(tied_emb, transpose=False)
        #print(input_dim, emb_dim)  # 410=410
        if input_dim != emb_dim:
            self.word_proj = nn.Sequential(
                nn.Linear(input_dim, emb_dim, bias=False),
                self.word_proj,
            )



        self.class_proj = nn.Linear(input_dim, num_classes, bias=False)  # 410 -> 2
        self.out_dim = self.num_words + num_classes  # 2002
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def forward(self, input):  # (9000, 410) == (# toks, dim)
        inp_sz = functools.reduce(operator.mul, input.shape[:-1], 1)  # 9000
        out = self._float_tensor.new(inp_sz, self.out_dim)  # (9000, 410)

        # 1st input embs used in word_proj to make dimension 2000
        out[:, :self.num_words] = self.word_proj(input.view(inp_sz, -1))

        # Linear layer to make dimension 2
        out[:, self.num_words:] = self.class_proj(input.view(inp_sz, -1))

        return out  # dimension 2002


class AdaptiveSoftmax(nn.Module):
    """
    This is an implementation of the efficient softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax
    approximation for GPUs" (http://arxiv.org/abs/1609.04309).
    """

    def __init__(self, vocab_size, input_dim, cutoff, dropout, factor=4., adaptive_inputs=None, tie_proj=False):
        # vocab_size=33280
        # input_dim=410
        # cutoff=[2000, 6000]
        # dropout=0
        # adaptive_inputs=embed_tokens (AdaptiveInput module)
        # factor=1.0
        # tie_proj=False
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]  # now cutoff = [2000, 6000, 33280]
        else:
            assert vocab_size == cutoff[
                -1], 'cannot specify cutoff larger than vocab size'

        output_dim = cutoff[0] + len(cutoff) - 1  # 2002

        self.vocab_size = vocab_size
        self.cutoff = cutoff
        self.dropout = dropout
        self.input_dim = input_dim
        self.factor = factor

        self.lsm = nn.LogSoftmax(dim=1)

        if adaptive_inputs is not None:
            # adaptive_inputs.weights_for_band(0) = (tensor(2000,410), tensor(410,410))
            #                                         1st embs           1st linear
            self.head = TiedHeadModule(adaptive_inputs.weights_for_band(0), input_dim, len(cutoff) - 1)
            # ^ dims: 410 -> 2002, first 2000 uses "1st embs", last 2 dimensions new for classes
        else:
            self.head = nn.Linear(input_dim, output_dim, bias=False)

        self._make_tail(adaptive_inputs, tie_proj)  # tie_proj=False

        def init_weights(m):
            if hasattr(m, 'weight') and not isinstance(m, TiedLinear) and not isinstance(m, TiedHeadModule):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer('version', torch.LongTensor([1]))

    def _make_tail(self, adaptive_inputs=None, tie_proj=False):
        self.tail = nn.ModuleList()
        for i in range(len(self.cutoff) - 1):
            dim = int(self.input_dim // self.factor ** (i + 1))

            tied_emb, tied_proj = adaptive_inputs.weights_for_band(i + 1) \
                if adaptive_inputs is not None else (None, None)
            #print(tied_emb.size())   # i+1=1: (4000, 410), i+1=2: (27280, 410)
            #print(tied_proj.size())  # i+1=1: (410, 410),  i+1=2: (410, 410)

            if tied_proj is not None:
                if tie_proj:  # This is False
                    proj = TiedLinear(tied_proj, transpose=True)
                else:  # So we will just create a new Linear with dimensions (410, 410)
                    proj = nn.Linear(tied_proj.size(0), tied_proj.size(1), bias=False)
            else:
                proj = nn.Linear(self.input_dim, dim, bias=False)

            m = nn.Sequential(
                proj,  # new Linear 410->410
                nn.Dropout(self.dropout),
                nn.Linear(
                    dim, self.cutoff[i + 1] - self.cutoff[i], bias=False,
                ) if tied_emb is None else TiedLinear(tied_emb, transpose=False),  # tied_emb: (4000, 410), (27280, 410)
            )
            self.tail.append(m)

        #print(self.tail)
        #ModuleList((0): Sequential((0): Linear(in_features=410, out_features=410, bias=False) (1): Dropout(p=0, inplace=False) (2): TiedLinear(4000, 410))
        #           (1): Sequential((0): Linear(in_features=410, out_features=410, bias=False) (1): Dropout(p=0, inplace=False) (2): TiedLinear(27280, 410)))


    def upgrade_state_dict_named(self, state_dict, name):
        version_name = name + '.version'
        if version_name not in state_dict:
            raise Exception('This version of the model is no longer supported')

    def adapt_target(self, target):
        """
        In order to be efficient, the AdaptiveSoftMax does not compute the
        scores for all the word of the vocabulary for all the examples. It is
        thus necessary to call the method adapt_target of the AdaptiveSoftMax
        layer inside each forward pass.
        """

        target = target.view(-1)  # (9000,)
        new_target = [target.clone()]
        target_idxs = []

        for i in range(len(self.cutoff) - 1):  # 0, 1
            # i=0: 2000 <= t_rank < 6000  ----> t <- 2000
            # i=1: 6000 <= t_rank < 33280 ----> t <- 2001
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i

            if mask.any():
                target_idxs.append(mask.nonzero().squeeze(1))  # i=0: 1055, i=1: 1023

                # i=0: Shifting 2nd class vocab by 2000
                # i=1: Shifting 3rd class vocab by 6000
                new_target.append(target[mask].add(-self.cutoff[i]))  # "4913 -> 2913"
            else:
                target_idxs.append(None)
                new_target.append(None)

        # new_target = [all target (9000) w/ 2nd class toks reset as 2000 and 3rd as 2001,
        #               subset of target (1055) containing 2nd class, idx shifted by 2000
        #               subset of target (1023) containing 3rd class, idx shifted by 6000]
        #
        # target_idxs = [idxs b/t 2000, 6000 (1055 of them), idxs b/t 6000, 33280 (1023 of them)]
        return new_target, target_idxs

    def forward(self, input, target, return_target_idx=False):
        """
        Args:
            input: (b x t x d)
            target: (b x t)
        Returns:
            2 lists: output for each cutoff section and new targets by cut off
        """

        # [60, 150, 410] -> [9000, 410]
        input = input.contiguous().view(-1, input.size(-1))
        input = F.dropout(input, p=self.dropout, training=self.training)

        # new_target = [all target (9000) w/ 2nd class toks reset as 2000 and 3rd as 2001,
        #               subset target (1055) containing 2nd class, idx shifted by 2000
        #               subset target (1023) containing 3rd class, idx shifted by 6000]
        # target_idxs = [2nd class idxs (1055), 3rd class idxs (1023)]
        new_target, target_idxs = self.adapt_target(target)
        output = [self.head(input)]  # [9000, 410] -> [9000, 2002], ties 1st class embs

        # tail[0]: Linear(410->410), Drop, TiedLinear(410->4000, 2nd class embs)
        # tail[1]: Linear(410->410), Drop, TiedLinear(410->27280, 3rd class embs)
        for i in range(len(target_idxs)):
            if target_idxs[i] is not None:
                #                          ________inputs for 2nd/3rd preds____
                output.append(self.tail[i](input.index_select(0, target_idxs[i])))
            else:
                output.append(None)

        #print(output[0].size())   # (9000, 2002)
        #print(output[1].size())   # (1055, 4000)
        #print(output[2].size())   # (1023, 27280)
        if return_target_idx:
            return output, new_target, target_idxs
        return output, new_target

    def get_log_prob(self, input, target):
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """

        bsz, length, dim = input.size()
        input = input.contiguous().view(-1, dim)

        if target is not None:
            _, target_idxs = self.adapt_target(target)
        else:
            target_idxs = None

        head_y = self.head(input)
        log_probs = head_y.new_zeros(input.size(0), self.vocab_size)

        head_sz = self.cutoff[0] + len(self.tail)
        log_probs[:, :head_sz] = self.lsm(head_y)
        tail_priors = log_probs[:, self.cutoff[0]: head_sz].clone()

        for i in range(len(self.tail)):
            start = self.cutoff[i]
            end = self.cutoff[i + 1]

            assert(target_idxs is not None)
            if target_idxs is None:
                tail_out = log_probs[:, start:end]
                tail_out.copy_(self.tail[i](input))
                log_probs[:, start:end] = self.lsm(tail_out).add_(tail_priors[:, i, None])
            elif target_idxs[i] is not None:
                idxs = target_idxs[i]
                tail_out = self.tail[i](input[idxs])
                log_probs[idxs, start:end] = self.lsm(tail_out) + tail_priors[idxs, i, None]

        log_probs = log_probs.view(bsz, length, -1)
        return log_probs

    def get_vocab_logits(self, input, target=None):

        bsz, length, dim = input.size()
        input = input.contiguous().view(-1, dim)

        if target is not None:
            _, target_idxs = self.adapt_target(target)
        else:
            target_idxs = None

        head_y = self.head(input)
        logits = head_y.new_ones(input.size(0), self.vocab_size) * -10000.0

        head_sz = self.cutoff[0] + len(self.tail)
        logits[:, :head_sz] = head_y
        tail_priors = logits[:, self.cutoff[0]: head_sz].clone()

        for i in range(len(self.tail)):
            start = self.cutoff[i]
            end = self.cutoff[i + 1]

            if target_idxs is None:
                tail_log_probs = self.lsm(self.tail[i](input))
                logits[:, start:end] = tail_priors[:, i, None] + tail_log_probs
            elif target_idxs[i] is not None:
                idxs = target_idxs[i]
                tail_log_probs = self.lsm(self.tail[i](input[idxs]))
                logits[idxs, start:end] = tail_priors[idxs, i, None] + tail_log_probs

        # log_probs = self.lsm(logits)
        # log_probs = log_probs.view(bsz, length, -1)
        return logits
