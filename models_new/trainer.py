import torch
import torch.nn.functional as F
from transformers import Trainer


class RecTrainer(Trainer):
    def __init__(
        self,
        ldpo_alpha: float = 1.0,
        ldpo_beta: float = 1.0,
        ldpo_only: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ldpo_alpha = ldpo_alpha
        self.ldpo_beta = ldpo_beta
        self.ldpo_only = ldpo_only

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        ldpo_item_index = inputs.pop("ldpo_item_index", None)
        ldpo_item_groups = inputs.pop("ldpo_item_groups", None)
        ldpo_num_items = inputs.pop("ldpo_num_items", None)
        sample_type = inputs.pop("sample_type", None)
        ldpo_only = self.ldpo_only

        outputs = model(**inputs)
        logits = outputs.logits

        bsz, _, vocab_size = logits.shape
        logits_next = logits[:, :-1, :]
        target = labels[:, 1:]

        valid = target.ne(-100)
        if sample_type is not None:
            # [B] -> [B, L-1]
            st_next = sample_type.unsqueeze(1).expand_as(target)
            if ldpo_only:
                ce_valid = valid & st_next.eq(0)
            else:
                ce_valid = valid
        else:
            ce_valid = torch.zeros_like(valid, dtype=torch.bool) if ldpo_only else valid

        if ce_valid.any():
            ce_loss = F.cross_entropy(
                logits_next[ce_valid],
                target[ce_valid],
                reduction="mean",
            )
        else:
            ce_loss = logits.new_zeros(())

        if ldpo_item_index is None or ldpo_item_groups is None:
            loss = ce_loss
            return (loss, outputs) if return_outputs else loss

        alpha = float(self.ldpo_alpha)
        beta = float(self.ldpo_beta)
        if alpha == 0.0:
            loss = ce_loss
            return (loss, outputs) if return_outputs else loss

        log_probs_next = torch.log_softmax(logits_next, dim=-1)

        token_logp = torch.zeros_like(target, dtype=log_probs_next.dtype)
        gathered = log_probs_next.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        token_logp[valid] = gathered[valid]

        item_index_next = ldpo_item_index[:, 1:]
        bsz2, nmax = ldpo_item_groups.shape
        if bsz2 != bsz:
            raise ValueError(f"ldpo_item_groups batch size mismatch: {bsz2} vs logits batch size {bsz}")
        if nmax == 0:
            loss = ce_loss
            return (loss, outputs) if return_outputs else loss

        if sample_type is not None:
            ldpo_rows = sample_type.eq(1)
            if not ldpo_rows.any():
                loss = ce_loss
                return (loss, outputs) if return_outputs else loss
            token_logp = token_logp[ldpo_rows]
            valid = valid[ldpo_rows]
            item_index_next = item_index_next[ldpo_rows]
            ldpo_item_groups = ldpo_item_groups[ldpo_rows]
            if ldpo_num_items is not None:
                ldpo_num_items = ldpo_num_items[ldpo_rows]
            bsz = token_logp.size(0)

        pi = torch.zeros((bsz, nmax), dtype=token_logp.dtype, device=token_logp.device)
        for k in range(nmax):
            mask_k = item_index_next.eq(k) & valid
            if mask_k.any():
                pi[:, k] = (token_logp * mask_k).sum(dim=-1)

        groups = ldpo_item_groups.clone()
        if ldpo_num_items is not None:
            for b in range(bsz):
                n = int(ldpo_num_items[b].item())
                if n < nmax:
                    groups[b, n:] = 0

        max_group = int(groups.max().item())
        if max_group <= 1:
            loss = ce_loss
            return (loss, outputs) if return_outputs else loss

        s = beta * pi
        exp_s = torch.exp(s.clamp(min=-50.0, max=50.0))

        loss_ldpo = logits.new_zeros(())
        for j in range(1, max_group):
            higher = groups.eq(j + 1)
            lower = groups.ge(1) & groups.le(j)
            num_higher = higher.sum()
            if num_higher == 0:
                continue
            denom_lower = (exp_s * lower).sum(dim=-1, keepdim=True)
            frac = exp_s / (exp_s + denom_lower + 1e-12)
            term = -(torch.log(frac + 1e-12) * higher)
            loss_ldpo = loss_ldpo + term.sum() / (num_higher + 1e-12)

        loss = ce_loss + alpha * loss_ldpo
        return (loss, outputs) if return_outputs else loss

