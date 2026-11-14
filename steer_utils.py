"""Shared helpers for the thesis experiments.

Steering vectors to deactivate ROME knowledge edits in GPT-2 XL.

Conventions
-----------
* ``hidden_states[i]`` is the output of transformer block ``i - 1``
  (index 0 is the embedding output). The output of the ROME-edited block 17
  is therefore ``hidden_states[18]``. All functions below take a
  *hidden-state index*, not a block number.
* A steering vector ``v`` is defined as ``mean(h_original - h_edited)``,
  so steering means *adding* ``alpha * v`` (same convention as CAA).
* ``positions='last'`` intervenes from the last prompt token onwards
  (this includes answer tokens when an answer is scored);
  ``positions='all'`` intervenes at every position.
"""
import contextlib
import io
import json
import logging
import os
import urllib.request

import torch

COUNTERFACT_URL = 'https://memit.baulab.info/data/dsets/counterfact.json'


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_counterfact(path, url=COUNTERFACT_URL):
    """Download CounterFact once and return it as a list of records."""
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    with open(path) as f:
        return json.load(f)


def relation_records(data, relation):
    """All CounterFact records of one relation, flattened into simple dicts."""
    out = []
    for rec in data:
        rr = rec['requested_rewrite']
        if rr.get('relation_id') != relation:
            continue
        out.append(dict(
            case_id=rec['case_id'],
            relation=relation,
            template=rr['prompt'],
            prompt=rr['prompt'].format(rr['subject']),
            subject=rr['subject'],
            true=rr['target_true']['str'],
            new=rr['target_new']['str'],
            paraphrase=rec.get('paraphrase_prompts', []),
            neighborhood=rec.get('neighborhood_prompts', []),
            attribute=rec.get('attribute_prompts', []),
        ))
    return out


# --------------------------------------------------------------------------
# Model and editor
# --------------------------------------------------------------------------
def load_editor(easyedit_dir, model_name='openai-community/gpt2-xl',
                hparams_path='./hparams/ROME/gpt2-xl'):
    """Load EasyEdit's ROME editor. Returns (editor, model, tok, hparams)."""
    os.chdir(easyedit_dir)
    from easyeditor import BaseEditor, ROMEHyperParams
    hparams = ROMEHyperParams.from_hparams(hparams_path)
    hparams.model_name = model_name
    editor = BaseEditor.from_hparams(hparams)
    return editor, editor.model, editor.tok, hparams


def rewrite_weight(model, hparams):
    """The weight matrix that ROME changes (block 17 MLP c_proj for GPT-2 XL)."""
    name = f'{hparams.rewrite_module_tmp.format(hparams.layers[0])}.weight'
    return model.get_parameter(name)


def rome_edit(editor, rec, quiet=True):
    """Apply one ROME edit in place (the model stays edited afterwards)."""
    kwargs = dict(prompts=[rec['prompt']], ground_truth=[rec['true']],
                  target_new=[rec['new']], subject=[rec['subject']],
                  sequential_edit=True, verbose=False)
    if not quiet:
        return editor.edit(**kwargs)
    level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return editor.edit(**kwargs)
    finally:
        logging.disable(level)


def rank_one_factors(dW):
    """Write the (rank-one) weight change dW as outer(u, v).

    Returns (u, v, relative_error). A relative error close to 0 means the
    two vectors reproduce dW exactly, so we only need to store u and v.
    """
    dW = dW.float()
    U, S, V = torch.svd_lowrank(dW, q=3, niter=4)
    u, v = U[:, 0] * S[0], V[:, 0]
    err = (torch.outer(u, v) - dW).norm() / dW.norm()
    return u.cpu(), v.cpu(), err.item()


@torch.no_grad()
def set_edit(W, W_orig, u=None, v=None):
    """Load one stored edit into W, or restore the original weights (u=None)."""
    if u is None:
        W.copy_(W_orig)
    else:
        W.copy_(W_orig + torch.outer(u, v).to(W.device, W.dtype))


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------
def _ids(tok, text):
    return tok(text)['input_ids']


@torch.no_grad()
def hidden_states(model, tok, prompt):
    """All hidden states for one prompt: tensor [n_layers + 1, seq_len, d] on CPU."""
    ids = torch.tensor([_ids(tok, prompt)], device=model.device)
    out = model(ids, output_hidden_states=True)
    return torch.stack(out.hidden_states)[:, 0].float().cpu()


# --------------------------------------------------------------------------
# Interventions
# --------------------------------------------------------------------------
def add_steer(vec, hidden_index, positions='all'):
    """Steering spec: add `vec` to hidden_states[hidden_index]."""
    return dict(kind='add', vec=vec, hidden_index=hidden_index, positions=positions)


def proj_steer(directions, hidden_index, positions='all'):
    """Steering spec: remove the span of `directions` (list of vectors or [k, d])."""
    D = torch.stack(list(directions)) if not torch.is_tensor(directions) else directions
    if D.dim() == 1:
        D = D[None]
    Q, _ = torch.linalg.qr(D.float().T)           # [d, k], orthonormal columns
    return dict(kind='proj', Q=Q, hidden_index=hidden_index, positions=positions)


def _make_hook(steer, start):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        # During text generation with a cache, later steps only see the new token(s):
        # then the whole chunk lies after the prompt, so intervene on all of it.
        s = start if h.shape[1] > start else 0
        part = h[:, s:, :]
        if steer['kind'] == 'add':
            part += steer['vec'].to(h.device, h.dtype)
        else:
            Q = steer['Q'].to(h.device, h.dtype)
            part -= (part @ Q) @ Q.T
        return output
    return hook


@contextlib.contextmanager
def steering(model, steer, prompt_len):
    """Context manager: run the model with `steer` active (None = no steering)."""
    if steer is None:
        yield
        return
    if steer['hidden_index'] < 1:
        raise ValueError('hidden_index must be >= 1 (index 0 is the embedding output)')
    start = 0 if steer['positions'] == 'all' else prompt_len - 1
    block = model.transformer.h[steer['hidden_index'] - 1]
    handle = block.register_forward_hook(_make_hook(steer, start))
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@torch.no_grad()
def answer_logprob(model, tok, prompt, answer, steer=None):
    """Mean log-probability per token of ' ' + answer after the prompt."""
    p_ids, a_ids = _ids(tok, prompt), _ids(tok, ' ' + answer)
    ids = torch.tensor([p_ids + a_ids], device=model.device)
    with steering(model, steer, len(p_ids)):
        logits = model(ids).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    pos = len(p_ids) - 1
    return sum(logp[pos + i, t].item() for i, t in enumerate(a_ids)) / len(a_ids)


@torch.no_grad()
def next_token_logprobs(model, tok, prompt, steer=None):
    """Log-probabilities of the next token after the prompt (vector over the vocabulary)."""
    p_ids = _ids(tok, prompt)
    ids = torch.tensor([p_ids], device=model.device)
    with steering(model, steer, len(p_ids)):
        logits = model(ids).logits[0, -1].float()
    return torch.log_softmax(logits, dim=-1).cpu()


def top1(model, tok, prompt, steer=None):
    """The single most likely next token, as text."""
    return tok.decode(int(next_token_logprobs(model, tok, prompt, steer).argmax()))


@torch.no_grad()
def generate(model, tok, prompt, n_tokens=8, steer=None):
    """Greedy continuation of the prompt (only the new text), with optional steering."""
    p_ids = _ids(tok, prompt)
    ids = torch.tensor([p_ids], device=model.device)
    with steering(model, steer, len(p_ids)):
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, len(p_ids):].tolist())


def score_record(model, tok, rec, steer=None):
    """Original vs. new answer for one record, plus the model's top-1 next token."""
    lt = answer_logprob(model, tok, rec['prompt'], rec['true'], steer)
    ln = answer_logprob(model, tok, rec['prompt'], rec['new'], steer)
    return dict(logp_true=lt, logp_new=ln, prefers_true=lt > ln,
                top1=top1(model, tok, rec['prompt'], steer))


def kl(base_logp, new_logp):
    """KL(base || new) between two next-token distributions given as log-probs."""
    return (base_logp.exp() * (base_logp - new_logp)).sum().item()


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def unit(x, dim=-1):
    return torch.nn.functional.normalize(x.float(), dim=dim)


def cos_matrix(X, Y=None):
    """Cosine similarity between the rows of X and the rows of Y (default: X)."""
    X = unit(X)
    Y = X if Y is None else unit(Y)
    return X @ Y.T


def mean_offdiag(C):
    """Mean of a square matrix without its diagonal."""
    n = C.shape[0]
    return (C.sum() - C.diagonal().sum()).item() / (n * n - n)
