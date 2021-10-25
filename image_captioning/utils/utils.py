from torch.nn.modules.linear import Identity
from trainer.start import *
from scipy.optimize import linear_sum_assignment
from transformers.utils.dummy_tokenizers_objects import PreTrainedTokenizerFast
from collections import Counter


class RealValidatePredictor(ValidatePredictor):
    """call eval_forward_pass"""
    @torch.no_grad()
    def forward_pass(self, data, **kwargs) -> Dict:
        with set_mode(self.trainer.net, 'eval'):
            return self.trainer.eval_forward_pass(data, **kwargs)


def weight_equal_by_instance(lengths):
    """
    weight such that each instance in a batch has an equal weight
    Return: weight (n,)
    """
    weight = 1 / lengths
    weight = weight[lengths_to_idx(lengths)]
    weight = weight / weight.sum()
    return weight


def mean_equal_by_instance(x, lengths):
    """
    Args:
        x: (n, hid)
        lengths: (bs, )
    """
    assert x.dim() == 2
    weight = weight_equal_by_instance(lengths)
    _, hid = x.shape
    x = (x * weight.unsqueeze(-1) / hid).sum()
    return x


def lengths_to_idx(lengths: Tensor):
    """
    [1, 2] into [0, 1, 1]
    """
    idx = []
    for i, length in enumerate(lengths):
        idx += [i] * length
    return torch.LongTensor(idx).to(lengths.device)


def mean_vary_lengths(x, lengths):
    """
    Args:
        x: (n, hid)
        lengths: (bs, )
    """
    # (bs, t, hid)
    x = chunk_pad_by_lengths(x, lengths, batch_first=True)
    # (bs, t)
    pad = make_pad_mask(lengths, batch_first=True)
    x[pad] = 0
    # (bs, hid)
    dev = x.device
    # (bs, hid)
    x = x.sum(dim=1) / lengths.to(dev).unsqueeze(-1)
    return x


def max_vary_lengths(x, lengths):
    """
    Args:
        x: (n, hid)
        lengths: (bs, )
    """
    # (bs, t, hid)
    x = chunk_pad_by_lengths(x, lengths, batch_first=True)
    # (bs, t)
    pad = make_pad_mask(lengths, batch_first=True)
    x[pad] = float('-inf')
    # (bs, hid)
    x, _ = x.max(dim=1)
    return x


def index_dedupe(idx: Tensor):
    """
    returns the deduped indices.

    Args:
        idx: (N, K) where K = number of dims
    
    Returns:
        idx: (<= N, K) deduped
    """
    counter = Counter()

    idx = idx.cpu().numpy()
    idx_new = []

    for each in idx:
        key = tuple(each)
        if key not in counter:
            idx_new.append(each)
        counter[key] += 1

    cnt = []
    for each in idx_new:
        cnt.append(counter[tuple(each)])

    idx_new = torch.tensor(idx_new)
    cnt = torch.tensor(cnt)
    return idx_new, cnt


class MultiplyGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, mult: Tensor):
        assert len(x) == len(mult)
        ctx.save_for_backward(mult)
        return x

    @staticmethod
    def backward(ctx, grad):
        mult, = ctx.saved_tensors
        return (grad * mult.to(grad.device), None)


def lengths_to_arange_chunk(lengths, chunk_sizes):
    """
    chunk_sizes = [5, 5]
    lengths = [2, 4]
    => [0, 1, 5, 6, 7, 8]
    """
    arange = []
    j = 0
    for length, chunk_size in zip(lengths, chunk_sizes):
        arange += list(range(j, j + length))
        j += chunk_size
    arange = torch.LongTensor(arange).to(lengths.device)
    return arange


def expand_by_lengths(x, lengths):
    """
    Args:
        x: (bs, d)
        lengths: (bs, )
    Returns: (n*t, d)
    """
    # (n*t, )
    idx = lengths_to_idx(lengths)
    # (n*t, d)
    out = x[idx]
    return out


def chunk_by_lengths(x, lengths):
    """
    Args: 
        x: (n*t, d)
    """
    out = list(x.split(list(lengths), 0))
    return out


def chunk_pad_by_lengths(x, lengths, batch_first: bool = False):
    """
    Args: 
        x: (n*t, d)
    Returns: 
        (t, n, d) if not batch_first
        (n, t, d) if batch_first
    """
    x = x.split(list(lengths), 0)
    x = nn.utils.rnn.pad_sequence(x, batch_first=batch_first)
    return x


def make_pad_mask(lengths, batch_first, max_t=None):
    """
    square matrix with row vectors of [False] * length[i]
    """
    dev = lengths.device
    bs = lengths.shape[0]
    if max_t is None:
        max_t = lengths.max()
    pad_mask = torch.arange(0, max_t).expand(bs, max_t).to(dev)
    pad_mask = pad_mask >= lengths.unsqueeze(-1)
    if not batch_first:
        pad_mask = pad_mask.permute([1, 0])
    return pad_mask


def lengths_to_selection_mask(lengths, batch_first, max_t=None):
    return ~make_pad_mask(lengths, batch_first, max_t)


def flat_by_lengths(x, lengths, batch_first):
    """
    Args:
        x: (n, t, c)
        batch_first: whether x is (n, t, c)
    Returns: (n*t, c)
    """
    assert x.dim() in (2, 3)
    if not batch_first:
        # t, n, c => n, t, c
        if x.dim() == 3:
            x = x.permute([1, 0, 2])
        elif x.dim() == 2:
            x = x.permute([1, 0])
        else:
            raise NotImplementedError()

    mask = ~make_pad_mask(lengths, batch_first=True)
    assert x.shape[:2] == mask.shape
    # (n, t, c) => (n*t, c)
    x = x.flatten(0, 1)
    # (n, t) => (n*t)
    mask = mask.flatten()
    out = x[mask].contiguous()
    return out


def make_mlp_n_layers(n_hid,
                      n_out,
                      n_layer,
                      activation,
                      has_last_linear=True,
                      has_last_norm=False,
                      dropout=0):
    layers = []
    for i in range(n_layer):
        layers.append(nn.Linear(n_hid, n_hid))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.LayerNorm(n_hid))
        if activation == 'relu':
            layers.append(nn.ReLU())
        else:
            raise NotImplementedError()
    if has_last_linear:
        layers.append(nn.Linear(n_hid, n_out))
    if has_last_norm:
        layers.appned(nn.LayerNorm(n_out))
    if len(layers) == 0:
        return nn.Identity()
    else:
        return nn.Sequential(*layers)


def make_mlp_n_layers_v2(n_hid,
                         n_layer,
                         activation='relu',
                         last_activation=False,
                         dropout=0):
    layers = []
    for i in range(n_layer):
        layers.append(nn.Linear(n_hid, n_hid))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.LayerNorm(n_hid))
        if i < n_layer - 1 or last_activation:
            if activation == 'relu':
                layers.append(nn.ReLU())
            else:
                raise NotImplementedError()
    if len(layers) == 0:
        return nn.Identity()
    else:
        return nn.Sequential(*layers)


class MLPLNReLU(nn.Sequential):
    def __init__(self,
                 n_hid: int,
                 n_layer: int,
                 last_relu: bool = True,
                 last_norm: bool = True,
                 n_inp: int = None,
                 n_out: int = None):
        if n_inp is None:
            n_inp = n_hid
        if n_out is None:
            n_out = n_hid

        layers = []
        for i in range(n_layer):
            if i == 0 and i == n_layer - 1:
                layers.append(nn.Linear(n_inp, n_out))
                n = n_out
            elif i == 0:
                layers.append(nn.Linear(n_inp, n_hid))
                n = n_hid
            elif i == n_layer - 1:
                layers.append(nn.Linear(n_hid, n_out))
                n = n_out
            else:
                layers.append(nn.Linear(n_hid, n_hid))
                n = n_hid

            if i != n_layer - 1 or last_norm:
                layers.append(nn.LayerNorm(n))
            if i != n_layer - 1 or last_relu:
                layers.append(nn.ReLU(inplace=False))

        if len(layers) == 0:
            layers.append(Identity())

        super().__init__(*layers)


class ResidualMLPPreNormReLU(nn.Module):
    def __init__(self,
                 n_hid: int,
                 n_layer: int,
                 last_relu: bool = False,
                 last_norm: bool = False):
        super().__init__()
        self.last_relu = last_relu
        self.last_norm = last_norm

        linears = []
        norms = []
        for i in range(n_layer):
            linears.append(nn.Linear(n_hid, n_hid))
            norms.append(nn.LayerNorm(n_hid))

        if last_norm:
            norms.append(nn.LayerNorm(n_hid))

        self.linears = nn.ModuleList(linears)
        self.norms = nn.ModuleList(norms)

    def forward(self, x):
        mid = x
        for i in range(len(self.linears)):
            mid = mid + self.linears[i](F.relu((self.norms[i](x))))
        if self.last_norm:
            mid = self.norms[-1](mid)
        if self.last_relu:
            mid = F.relu(mid)
        return mid


def make_opt(lr, eps=1e-8):
    name = f'adam{lr}'
    if eps != 1e-8:
        name += f'eps{eps}'

    @rename(name)
    def fn(net):
        return optim.Adam(net.parameters(), lr=lr, eps=eps)

    return fn


def infer_sequence_lengths(x, pad_token_id):
    """create a tensor [0, 1, 2, ... N] multiplied it with the eligible mask, then take the argmax + 1"""
    dev = x.device
    idx = torch.arange(x.shape[1], device=dev).expand_as(x)
    out = ((x != pad_token_id) * idx).argmax(dim=1) + 1
    return out.cpu()


def arg_first_zero(x):
    """
    [
        [1, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ] 
    => [2, 1, 0]
    """
    ones = torch.ones_like(x).float()
    neginf = torch.empty_like(x).float().fill_(float('-inf'))
    x = torch.where(x.bool(), ones, neginf)
    x, _ = x.cumsum(dim=-1).max(dim=-1)
    x.clamp_(min=0)
    return x.long()


def convert_ids_to_tokens_and_split_report(ids,
                                           tokenizer: PreTrainedTokenizerFast,
                                           sentence_begin_token='<s>',
                                           report_end_token='</s>'):
    """split batch of report ids into a batch of sentences."""
    bs = len(ids)
    batch = []
    for i in range(bs):
        tokens = tokenizer.convert_ids_to_tokens(ids[i])
        out = []
        sentence = []
        assert tokens[
            0] == sentence_begin_token, f'sentence does not begin with {sentence_begin_token}.'
        for each in tokens:
            if each == sentence_begin_token:
                out.append(sentence)
                sentence = []
            elif each == report_end_token:
                # early stop
                # any pad tokens will be ignored automatically!
                out.append(sentence)
                sentence = []
                break
            else:
                sentence.append(each)

        # something left due to no </s>
        if len(sentence) > 0:
            out.append(sentence)

        out = out[1:]  # ignore the first <s>
        batch.append(out)
    return batch


def convert_ids_to_tokens(ids,
                          tokenizer: PreTrainedTokenizerFast,
                          sentence_begin_token='<s>',
                          report_end_token='</s>'):
    bs = len(ids)
    batch = []
    for i in range(bs):
        tokens = tokenizer.convert_ids_to_tokens(ids[i],
                                                 skip_special_tokens=True)
        batch.append(tokens)
    return batch


class Evaluator(ValidatePredictor):
    def forward_pass(self, data, **kwargs) -> Dict:
        out = self.trainer.eval_forward_pass(data)
        assert 'reference' in out
        assert 'hypothesis' in out
        self.buffer['reference'] += out['reference']
        self.buffer['hypothesis'] += out['hypothesis']
        self.buffer['ref_text'] += data['ref_text']
        return out

    def predict(self, loader: DataLoader):
        self.looper.loop(loader, n_max_itr=len(loader))

        out = {}
        out.update(
            best_sentence_level_bleu(self.buffer['reference'],
                                     self.buffer['hypothesis']))
        return out


class ExactMatchEvaluator(Evaluator):
    def __init__(self, trainer: BaseTrainer, callbacks):
        super(BasePredictor, self).__init__()
        self.trainer = trainer
        self.callbacks = callbacks
        self.looper = Looper(base=self, callbacks=self.callbacks)

    def predict(self, loader: DataLoader):
        out = super().predict(loader)

        # tokenizer
        tokenizer = self.trainer.conf.tokenizer

        # string predictions
        report_ref = []
        report_pred = []
        # for each report
        pred_lengths = []
        true_lengths = []
        for ref, hyp in zip(self.buffer['reference'],
                            self.buffer['hypothesis']):
            pred_lengths.append(len(hyp))
            true_lengths.append(len(ref))
            refs = []
            for each in ref:
                # each[0] = the first and only reference sentence
                refs.append(tokenizer.convert_tokens_to_string(each[0]))
            hyps = []
            for each in hyp:
                hyps.append(tokenizer.convert_tokens_to_string(each))
            report_ref.append(refs)
            report_pred.append(hyps)
        # lengths
        pred_lengths = np.array(pred_lengths)
        true_lengths = np.array(true_lengths)
        acc_length = (pred_lengths == true_lengths).mean()
        abs_err_length = np.abs(pred_lengths - true_lengths).mean()

        out.update(best_exact_match_score(report_ref, report_pred))
        out.update(best_ref_match_score(self.buffer['ref_text'], report_pred))
        out.update({
            'acc_length': acc_length,
            'abs_err_length': abs_err_length,
        })
        return out


def best_sentence_level_bleu(ref_reports, hyp_reports):
    """
    Args:
        references: list of list of list of sentences [report[refs[sentence]]]
        hypotheses: list of list of sentences [report[sentence]]
    """
    scores_macro = []  # each report has equal weight
    scores_weighted = []  # each sentence has equal weight
    for i in range(len(ref_reports)):
        ref_report = ref_reports[i]
        hyp_report = hyp_reports[i]
        scores, row, col = best_report_bleu(ref_report, hyp_report)
        scores_macro.append(np.array(scores).mean())
        scores_weighted += scores

    return {
        'bleu4_macro': np.array(scores_macro).mean(),
        'bleu4_weighted': np.array(scores_weighted).mean(),
    }


def best_report_bleu(references, predicts):
    """best match report bleu score"""
    dists = []
    for ref in references:
        row = []
        for pred in predicts:
            dist = sentence_bleu(ref, pred, weights=(0.25, 0.25, 0.25, 0.25))
            row.append(dist)
        dists.append(row)
    dists = np.array(dists)

    # empty report
    if len(references) == 0:
        row, col = None, None
        if len(predicts) == 0:
            # predicted as empty = correct
            scores = [1.]
        else:
            # predicted something = no score
            scores = []
    else:
        row, col = linear_sum_assignment(dists, maximize=True)
        scores = list(dists[row, col])

    # 0 score for the differences
    scores += [0] * abs(len(references) - len(predicts))
    # row = idx of refs
    # col = idx of predicts
    return scores, row, col


def best_exact_match_score(ref_reports, hyp_reports):
    """
    Args:
        references: list of list of sentences [report[sentence]]
        hypotheses: list of list of sentences [report[sentence]]
    """
    scores_macro = []  # each report has equal weight
    scores_weighted = []  # each sentence has equal weight
    for i in range(len(ref_reports)):
        ref_report = ref_reports[i]
        hyp_report = hyp_reports[i]
        scores, row, col = best_exact_match(ref_report, hyp_report)
        # 0 score for the differences
        scores_macro.append(np.array(scores).mean())
        scores_weighted += scores

    return {
        'exact_macro': np.array(scores_macro).mean(),
        'exact_weighted': np.array(scores_weighted).mean(),
    }


def best_exact_match(references, predicts):
    """best match report bleu score
    Args:
        references: list of strings
        predicts: list of strings
    """
    # all-pair matching scores
    dists = []
    for ref in references:
        row = []
        for pred in predicts:
            row.append(float(ref == pred))
        dists.append(row)
    dists = np.array(dists)

    # empty report
    if len(references) == 0:
        row, col = None, None
        if len(predicts) == 0:
            # predicted as empty = correct
            scores = [1.]
        else:
            scores = []
    else:
        row, col = linear_sum_assignment(dists, maximize=True)
        scores = list(dists[row, col])

    # differences in number of sentences
    scores += [0] * abs(len(references) - len(predicts))
    # row = idx of refs
    # col = idx of predicts
    return scores, row, col


def best_ref_match_score(ref_texts, hyp_reports):
    """
    Args:
        references: list of list of sentences [report[sentence]]
        hypotheses: list of list of sentences [report[sentence]]
    """
    scores_macro = []  # each report has equal weight
    scores_weighted = []  # each sentence has equal weight
    for i in range(len(ref_texts)):
        ref_text = ref_texts[i]
        hyp_report = hyp_reports[i]
        scores, row, col = best_ref_match(ref_text, hyp_report)
        # 0 score for the differences
        scores_macro.append(np.array(scores).mean())
        scores_weighted += scores

    return {
        'ref_macro': np.array(scores_macro).mean(),
        'ref_weighted': np.array(scores_weighted).mean(),
    }


def best_ref_match(ref_text, predicts):
    """best match from reference text
    Args:
        ref_texts: ref string including "\n"
        predicts: list of strings
    """
    ref_text = ref_text.split('\n')
    # (size, color, material, shape)
    attrs = [each.split(' ') for each in ref_text]

    def attr_match(attr, pred):
        # whether each attr is in prediction
        assert isinstance(pred, str)
        score = []
        for each in attr:
            score.append(float(each in pred))
        return np.array(score).mean()

    # all-pair matching scores
    dists = []
    for attr in attrs:
        row = []
        for pred in predicts:
            row.append(attr_match(attr, pred))
        dists.append(row)
    dists = np.array(dists)

    row, col = linear_sum_assignment(dists, maximize=True)
    scores = list(dists[row, col])

    # differences in number of sentences
    scores += [0] * abs(len(attrs) - len(predicts))
    # row = idx of refs
    # col = idx of predicts
    return scores, row, col


class PairStraightThrough(torch.autograd.Function):
    """returns the first argument, but backprops to both ends."""
    @staticmethod
    def forward(ctx, a, b):
        return a

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


pair_staright_through = PairStraightThrough.apply
