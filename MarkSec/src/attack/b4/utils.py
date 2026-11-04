import torch

def optimize_distribution(batch_p_scores: torch.FloatTensor, batch_q_scores: torch.FloatTensor, epsilon: float):
    """
    Args:
        batch_p_scores (torch.FloatTensor): [bsz, vocab] i.e. log(p(x))
        batch_q_scores (torch.FloatTensor): [bsz, vocab] i.e. log(q(x))
        epsilon (float): the constraint on the KL divergence between the optimized distribution h(x) and the target distribution p(x)
    Returns:
        alpha (torch.FloatTensor): [bsz, 1]
        batch_log_f (torch.FloatTensor): [bsz, vocab]
    """
    # DEBUG
    assert torch.exp(batch_p_scores).sum(-1).allclose(torch.ones_like(batch_p_scores.sum(-1))), "batch_p_scores is not a valid distribution"
    assert torch.exp(batch_q_scores).sum(-1).allclose(torch.ones_like(batch_q_scores.sum(-1))), "batch_q_scores is not a valid distribution"

    default_alpha = epsilon # set default alpha to epsilon if the root does not exist

    p_inf_mask = batch_p_scores == -torch.inf # [bsz, vocab], mask for inf values in batch_p_scores
    q_inf_mask = batch_q_scores == -torch.inf # [bsz, vocab], mask for inf values in batch_q_scores
    inf_mask = p_inf_mask | q_inf_mask # [bsz, vocab], mask for inf values in both batch_p_scores and batch_q_scores

    bsz = batch_p_scores.size(0)
    batch_log_pq = batch_p_scores - batch_q_scores # [bsz, vocab]
    batch_log_pq_square = batch_log_pq ** 2 # [bsz, vocab]
    def F(alpha):
        batch_log_f = (alpha + 1) * batch_p_scores - alpha * batch_q_scores # [bsz, vocab]
        batch_log_f.masked_fill_(q_inf_mask, -torch.inf) # in case of inf values in batch_q_scores
        batch_log_Z = torch.logsumexp(batch_log_f, 1, keepdim=True)  # [bsz, 1]
        """Unstable"""
        # batch_F = (alpha * torch.exp(batch_log_f - batch_log_Z) * batch_log_pq).sum(-1, keepdim=True) - batch_log_Z - epsilon

        """Stable"""
        batch_F = (alpha * torch.exp(batch_log_f - batch_log_Z) * batch_log_pq).masked_fill(inf_mask, 0).sum(-1, keepdim=True) - batch_log_Z - epsilon

        return batch_F
    
    lower = torch.ones(bsz, 1, device=batch_p_scores.device) * 0.000000001
    F_lower = F(lower)
    step_size = 0.05
    upper = lower
    finish_mask = torch.zeros(bsz, 1, device=batch_p_scores.device, dtype=torch.bool)
    while (F_lower * F(upper) > 0).any() and step_size < 100:
        step_size *= 2
        # step_size = torch.min(step_size, torch.ones_like(step_size) * 100)
        tmp_upper = upper + step_size
        old_upper = upper
        upper = torch.where(finish_mask == 0, tmp_upper, old_upper) # for the ones that not finished before this turn, update upper bound to new upper bound
        tmp_finish_mask = F_lower * F(tmp_upper) < 0 # calculate finish mask for this turn
        finish_mask = finish_mask | tmp_finish_mask # update finish mask
        lower = torch.where(~finish_mask, old_upper, lower) # for the ones that not finished, update lower bound to old upper bound

    for _ in range(10):
        mid = lower + (upper - lower) / 2
        F_mid = F(mid)
        F_lower = F(lower)
        F_upper = F(upper)
        lower = torch.where(F_mid * F_lower > 0, mid, lower)
        upper = torch.where(F_mid * F_upper > 0, mid, upper)

    valid_mask = finish_mask
    avg_alpha = torch.masked_select(mid, valid_mask).mean()
    mid = torch.where(~valid_mask, avg_alpha.unsqueeze(0), mid)
    alpha = mid

    step = 0
    while step < 40:
        batch_log_f = (alpha + 1) * batch_p_scores - alpha * batch_q_scores # [bsz, vocab]
        batch_log_f.masked_fill_(q_inf_mask, -torch.inf) # in case of inf values in batch_q_scores
        batch_log_Z = torch.logsumexp(batch_log_f, 1, keepdim=True)  # [bsz, 1]
        batch_f = torch.exp(batch_log_f)

        # unstable
        # batch_Z_prime = (batch_f * batch_log_pq).sum(-1, keepdim=True) # [bsz, 1]
        # stabl
        batch_Z_prime = (batch_f * batch_log_pq).masked_fill(inf_mask, 0).sum(-1, keepdim=True) # [bsz, 1]

        # unstable
        # batch_F = (alpha * torch.exp(batch_log_f - batch_log_Z) * batch_log_pq).sum(-1, keepdim=True) - batch_log_Z - epsilon
        # stable
        batch_F = (alpha * torch.exp(batch_log_f - batch_log_Z) * batch_log_pq).masked_fill(inf_mask, 0).sum(-1, keepdim=True) - batch_log_Z - epsilon

        batch_Z = batch_f.sum(-1, keepdim=True) # [bsz, 1]

        # unstable
        # batch_F_prime = (alpha * (batch_log_pq_square * batch_f).sum(-1, keepdim=True) / batch_Z) - (alpha * ((batch_Z_prime / batch_Z) ** 2))
        # stable
        batch_F_prime = (alpha * (batch_log_pq_square * batch_f).masked_fill(inf_mask, 0).sum(-1, keepdim=True) / batch_Z) - (alpha * ((batch_Z_prime / batch_Z) ** 2))
        

        if torch.abs(batch_F).max().item() < 1e-6:
            # print(f"[INFO] optimize_distribution(): converged!")
            return alpha.squeeze(-1), batch_log_f - batch_log_Z

        alpha = alpha - batch_F / batch_F_prime
        step += 1
        # res.append({
        #     'step': step,
        #     'loss': torch.abs(batch_F).max().item()
        # })
        # print(f"Step {step}, loss: {torch.abs(batch_F).max().item()}")

    if alpha.isnan().any() or (alpha < 0).any():
        valid_mask = (~alpha.isnan()) & (alpha >= 0)
        # print(f"[WARNING] optimize_distribution(): too large epsilon, only {valid_mask.sum().item()} out of {bsz} converged")
        # if all nan, set to default_alpha
        if not valid_mask.any():
            alpha = torch.ones_like(alpha) * default_alpha
        # if some nan, set to average of valid ones
        else:
            avg_alpha = torch.masked_select(alpha, valid_mask).mean()
            alpha = torch.where(~valid_mask, avg_alpha.unsqueeze(0), alpha)

        batch_log_f = (alpha + 1) * batch_p_scores - alpha * batch_q_scores # [bsz, vocab]
        batch_log_f.masked_fill_(q_inf_mask, -torch.inf) # in case of inf values in batch_q_scores
        batch_log_Z = torch.logsumexp(batch_log_f, 1, keepdim=True) # [bsz, 1]
        return alpha.masked_select(valid_mask), batch_log_f - batch_log_Z
    else:
        # print("[WARNING] optimize_distribution(): did not converge")
        return alpha.squeeze(-1), batch_log_f - batch_log_Z