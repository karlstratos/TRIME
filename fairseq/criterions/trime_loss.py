import math

import torch.distributed as dist
import torch.nn.functional as F
import torch

import random

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

def compute_in_bacth_logits(reps, labels):
    B = reps.shape[0]  # 60
    L = reps.shape[1]  # 150
    reps = reps.contiguous().view(-1, reps.shape[-1])  # (9000, 410)

    bsz = reps.shape[0]

    # compute scaled IP
    inbatch_logits = torch.mm(reps, reps.T) / math.sqrt(reps.shape[-1])  # (9000, 9000)
    # inbatch_logits(i, j) = <ffn(i), ffn(j)> / sqrt(d)
    inbatch_labels = labels

    local_mask = torch.ones((L, L), device=reps.device)
    #      1      1      1      1      1                 (150, 150)
    #      1      1      1      1      1
    #      1      1      1      1      1
    #      1      1      1      1      1
    #      1      1      1      1      1

    local_mask = torch.triu(local_mask, diagonal=0) * -10000.0  # "If diagonal = 0, all elements on and above the main diagonal are retained. "
    # -10000 -10000 -10000 -10000 -10000                 (150, 150)
    #      0 -10000 -10000 -10000 -10000
    #      0      0 -10000 -10000 -10000
    #      0      0      0 -10000 -10000
    #      0      0      0      0 -10000

    local_mask[local_mask == 0] = 1
    # -10000 -10000 -10000 -10000 -10000                 (150, 150)
    #      1 -10000 -10000 -10000 -10000
    #      1      1 -10000 -10000 -10000
    #      1      1      1 -10000 -10000
    #      1      1      1      1 -10000

    local_mask = torch.block_diag(*((local_mask, ) * B))
    # -10000 -10000 -10000 -10000 -10000      0      0      0      0      0 ...                  (9000, 9000): 60 blocks of (150, 150)
    #      1 -10000 -10000 -10000 -10000      0      0      0      0      0
    #      1      1 -10000 -10000 -10000      0      0      0      0      0
    #      1      1      1 -10000 -10000      0      0      0      0      0
    #      1      1      1      1 -10000      0      0      0      0      0
    #      0      0      0      0      0 -10000 -10000 -10000 -10000 -10000
    #      0      0      0      0      0      1 -10000 -10000 -10000 -10000
    #      0      0      0      0      0      1      1 -10000 -10000 -10000
    #      0      0      0      0      0      1      1      1 -10000 -10000
    #      0      0      0      0      0      1      1      1      1 -10000 ...
    #        ...

    local_mask[local_mask == 0] = -10000.0
    local_mask[local_mask == 1] = 0
    # -10000 -10000 -10000 -10000 -10000 -10000 -10000 -10000 -10000 -10000 ...                  (9000, 9000): 60 blocks of (150, 150)
    #      0 -10000 -10000 -10000 -10000 -10000 -10000 -10000 -10000 -10000
    #      0      0 -10000 -10000 -10000 -10000 -10000 -10000 -10000 -10000
    #      0      0      0 -10000 -10000 -10000 -10000 -10000 -10000 -10000
    #      0      0      0      0 -10000 -10000 -10000 -10000 -10000 -10000
    # -10000 -10000 -10000 -10000 -10000 -10000 -10000 -10000 -10000 -10000
    # -10000 -10000 -10000 -10000 -10000      0 -10000 -10000 -10000 -10000
    # -10000 -10000 -10000 -10000 -10000      0      0 -10000 -10000 -10000
    # -10000 -10000 -10000 -10000 -10000      0      0      0 -10000 -10000
    # -10000 -10000 -10000 -10000 -10000      0      0      0      0 -10000 ...
    #        ...
    assert(local_mask.shape[0] == bsz)
    inbatch_mask = local_mask

    # inbatch_logits before
    #       7     10      4      6      1     11      7     ...                                   (9000, 9000): score(i,j) between every token pair i,j in batch
    #      10      3      2      9      8      8      5     ...
    #       4      6      1     16      3      4      2     ...
    inbatch_logits = inbatch_logits + inbatch_mask

    # inbatch_logits after
    #   -9993  -9990  -9996  -9994  -9999  -9989  -9993     ...                                   (9000, 9000): p(j|i) = 0 for j >= i or j in previous sent
    #      10  -9997  -9998  -9991  -9992  -9992  -9995     ...
    #       4      6  -9999  -9984  -9997  -9996  -9998     ...

    return inbatch_logits, inbatch_labels


@register_criterion('trime_loss')
class TrimeLoss(FairseqCriterion):
    """
    This is an implementation of the Trime loss.
    In this function, only local memory will be used to compute the loss.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

        self.dist = args.dist
        self.temp = args.temp  # 1.0

        self.return_ce_loss = False

        if args.ddp_backend == 'c10d':
            raise Exception(
                'AdaptiveLossConloss is not compatible with the c10d '
                'version of DistributedDataParallel. Please use '
                '`--ddp-backend=no_c10d` instead.'
            )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # sample['id'] tensor (60,)
        # sample['nsentences'] int 60
        # sample['ntokens'] int 9000
        # sample['net_input']['src_tokens'] tensor (60,150) [[303,7..],[93,18..]..]
        # sample['net_input']['src_lengths'] tensor (60,)  [150..150]
        # sample['target'] tensor (60,150) [[7..1435],[18..28]..]

        # model: TransformerLanguageModel
        # https://github.com/facebookresearch/fairseq/blob/b5a039c292facba9c73f59ff34621ec131d82341/fairseq/models/transformer_lm.py#L230
        #print(model)

        # model.decoder.dictionary.indices: {..'Fashion': 11488, 'Ffordd': 11489..}
        #print(len(model.decoder.dictionary.indices))  # 33280 =  33277 + 3 (<s>, <pad>, </s>; <unk> already there)

        assert hasattr(model.decoder, 'adaptive_softmax') and model.decoder.adaptive_softmax is not None
        adaptive_softmax = model.decoder.adaptive_softmax

        # TransformerDecoder <- FairseqLanguageModel <- TransformerLanguageModel
        #
        # net_output[0] (x): (60, 150, 410)  == (bsz, len, dim)
        # net_output[1] (extra):
        #      net_output[1]['attn']: [None]
        #      net_output[1]['inner_states']: [17 tensors size (150, 60, 410)]
        #      net_output[1]['last_ffn_input']: (150, 60, 410)
        #      net_output[1]['model_output']: (150, 60, 410)
        net_output = model(**sample['net_input'])
        orig_target = model.get_targets(sample, net_output)  # (60, 150), this is just sample["target"]

        nsentences = orig_target.size(0)
        orig_target = orig_target.view(-1)  # Now orig_target shape (9000,)

        bsz = orig_target.size(0)

        # logits: (logits for 1st class + 2 classes predicted for all targets (9000, 2002) - tied(1st embs),
        #          logits for 2nd class predicted only for 2nd class targets (1055, 4000) - tied(2nd embs),
        #          logits for 3rd class predicted only for 3rd class targets (1023, 27280) - tied(3rd embs))
        # target: (all target (9000) w/ 2nd class toks reset as 2000 and 3rd as 2001,
        #          subset target (1055) containing 2nd class, idx shifted by 2000
        #          subset target (1023) containing 3rd class, idx shifted by 6000]
        # target_idxs = [2nd class idxs (1055), 3rd class idxs (1023)]
        logits, target, target_idxs = adaptive_softmax(net_output[0], orig_target, return_target_idx=True)
        assert len(target) == len(logits)
        norm_t = torch.logsumexp(logits[0], dim=-1)  # (9000,): norm_t[t] = log sum_{x:1st class} p_t(x)

        #print(target[0][target[0] == 2000].numel())  # 1055
        #print(target[0][target[0] == 2001].numel())  # 1023
        # I'm making the model predict 2000 for 2nd class and 2001 for 3rd class!
        #                          (9000, 2002)   (9000)
        token_loss = F.cross_entropy(logits[0], target[0], ignore_index=self.padding_idx, reduction='none')

        for i in range(len(target_idxs)):
            if target_idxs[i] is not None:
                # loss(t) = loss_1st(t) + [[2nd]] loss_2nd(t) + [[3rd]] loss_3rd(t)
                # I'm also making the model multitask predict 2nd/3rd indices for those steps
                #                                     i=0:     (1055, 4000)     (1055)
                #                                     i=1:     (1023, 27280)    (1023)
                token_loss[target_idxs[i]] += F.cross_entropy(logits[i + 1], target[i + 1], ignore_index=self.padding_idx, reduction='none')

        ori_loss = token_loss.sum(-1).view(-1)

        # While epoch <= 3 ("--ce-warmup-epoch 3"), return_ce_loss is True
        if self.return_ce_loss:
            loss = ori_loss
            orig = utils.strip_pad(orig_target, self.padding_idx)
            ntokens = orig.numel()
            sample_size = sample['target'].size(0) if self.args.sentence_avg else ntokens
            logging_output = {
                'loss': loss.data.double(),
                'ori_loss': ori_loss.data.double(),
                'norm_loss': 0.0,
                'ntokens': ntokens,
                'nsentences': nsentences,
                'sample_size': sample_size,
            }
            return loss, sample_size, logging_output

        #      net_output[1]['last_ffn_input']: (150, 60, 410)
        # reps: (60, 150, 410) == (bsz, len, dim)
        reps = net_output[1][self.args.knn_keytype].permute(1, 0, 2)  # knn_keytype: last_ffn_input

        # in_batch_logits: (9000, 9000), in_batch_labels = orig_target
        # in_batch_logits[i, j] = score between token i and j, masked so that p(j|i) = 0 if j >= i or j in outside sent of i
        in_batch_logits, in_batch_labels = compute_in_bacth_logits(reps, orig_target)

        norm_c = torch.logsumexp(in_batch_logits, dim=-1)

        in_batch_logs = F.log_softmax(in_batch_logits, dim=-1)

        #print((orig_target.view(-1, 1) != in_batch_labels.view(-1, 1)).sum())  # 0: these two are identical int vectors

        # negatives (9000, 9000): [i, j] False iff t[i] = t[j]
        #    - This is (9000, 1) x (1, 9000): so comparing each tok with every tok in the batch!
        #                 v (9000, 1)                 v (1, 9000)                            v True
        negatives = (orig_target.view(-1, 1) != (in_batch_labels.view(1, -1) if in_batch_labels.dim() == 1 else in_batch_labels))
        ctx_loss = -torch.logsumexp(in_batch_logs + negatives * -10000.0, dim=-1)  # negatives only used for ctx_loss, not norm_c


        # normalize token loss and ctx loss
        norm_tpc = torch.logsumexp(torch.stack((norm_t, norm_c), dim=-1), dim=-1)
        norm_loss = -torch.logsumexp(torch.stack((-token_loss + norm_t - norm_tpc, -ctx_loss + norm_c - norm_tpc), dim=-1), dim=-1)
        norm_loss = norm_loss.sum(-1).view(-1)

        loss = norm_loss

        orig = utils.strip_pad(orig_target, self.padding_idx)
        ntokens = orig.numel()
        sample_size = sample['target'].size(0) if self.args.sentence_avg else ntokens
        logging_output = {
            'loss': loss.data.double(),
            'ori_loss': ori_loss.data.double(),
            'norm_loss': norm_loss.data.double(),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ori_loss_sum = utils.item(sum(log.get('ori_loss', 0) for log in logging_outputs))
        norm_loss_sum = utils.item(sum(log.get('norm_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        if sample_size == 0:
            metrics.log_scalar('loss', 0.0, 0, round=3)
            metrics.log_scalar('ori_loss', 0.0, 0, round=3)
            metrics.log_scalar('norm_loss', 0.0, 0, round=3)
            return

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ori_loss', ori_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('norm_loss', norm_loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
