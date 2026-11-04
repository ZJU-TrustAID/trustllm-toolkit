import torch

def optimize_distribution(batch_p_scores: torch.FloatTensor, batch_q_scores: torch.FloatTensor, epsilon: float):
    """
    Numerically solves for the Lagrange multiplier (alpha) to mix distributions p and q
    such that KL(h || p) <= epsilon, where h is the optimized distribution.
    
    Used in B4's closed-form solution to balance efficacy and fidelity dynamically.

    Args:
        batch_p_scores (torch.FloatTensor): Logits of the fidelity distribution P_f (paraphrase model). [bsz, vocab]
        batch_q_scores (torch.FloatTensor): Logits of the watermark distribution P_w (proxy model). [bsz, vocab]
        epsilon (float): The KL divergence constraint threshold.

    Returns:
        tuple: (alpha, batch_log_f)
            alpha (torch.FloatTensor): The calculated mixing coefficient [bsz].
            batch_log_f (torch.FloatTensor): The resulting fused logits [bsz, vocab].
    """
    # Stability checks (optional, can be computationally expensive)
    # assert torch.exp(batch_p_scores).sum(-1).allclose(torch.ones_like(batch_p_scores.sum(-1))), "p not valid"
    
    default_alpha = epsilon
    bsz = batch_p_scores.size(0)
    device = batch_p_scores.device

    # Handling Infs
    p_inf_mask = batch_p_scores == -float('inf')
    q_inf_mask = batch_q_scores == -float('inf')
    inf_mask = p_inf_mask | q_inf_mask

    batch_log_pq = batch_p_scores - batch_q_scores
    batch_log_pq_square = batch_log_pq ** 2

    def F(alpha):
        # Calculate the function value whose root we want to find
        # Corresponds to the gradient of the Lagrangian or similar condition
        # h(x) ~ p(x)^(alpha+1) / q(x)^alpha
        batch_log_f = (alpha + 1) * batch_p_scores - alpha * batch_q_scores
        batch_log_f.masked_fill_(q_inf_mask, -float('inf'))
        batch_log_Z = torch.logsumexp(batch_log_f, 1, keepdim=True)
        
        # KL(h || p) - epsilon
        # Stable calculation masking infs
        batch_F = (alpha * torch.exp(batch_log_f - batch_log_Z) * batch_log_pq).masked_fill(inf_mask, 0).sum(-1, keepdim=True) - batch_log_Z - epsilon
        return batch_F

    # --- 1. Bisection Search for Initialization ---
    lower = torch.ones(bsz, 1, device=device) * 1e-9
    F_lower = F(lower)
    step_size = 0.05
    upper = lower.clone()
    finish_mask = torch.zeros(bsz, 1, device=device, dtype=torch.bool)
    
    # Exponential search to find upper bound
    while (F_lower * F(upper) > 0).any() and step_size < 100:
        step_size *= 2
        tmp_upper = upper + step_size
        old_upper = upper.clone()
        
        # Update upper only for unfinished
        upper = torch.where(~finish_mask, tmp_upper, upper)
        
        # Check which are finished (sign change detected)
        tmp_finish_mask = (F_lower * F(tmp_upper)) < 0
        finish_mask = finish_mask | tmp_finish_mask
        
        # If not finished, lower bound moves up to previous upper
        lower = torch.where(~finish_mask, old_upper, lower)

    # Binary search refinement
    for _ in range(10):
        mid = lower + (upper - lower) / 2
        F_mid = F(mid)
        F_l = F(lower)
        F_u = F(upper)
        lower = torch.where(F_mid * F_l > 0, mid, lower)
        upper = torch.where(F_mid * F_u > 0, mid, upper)

    # Handle non-converged items
    valid_mask = finish_mask
    if valid_mask.any():
        avg_alpha = torch.masked_select(mid, valid_mask).mean()
        # Fill invalid with average or default
        alpha = torch.where(~valid_mask, avg_alpha.unsqueeze(0), mid)
    else:
        alpha = mid

    # --- 2. Newton-Raphson Optimization ---
    step = 0
    while step < 40:
        batch_log_f = (alpha + 1) * batch_p_scores - alpha * batch_q_scores
        batch_log_f.masked_fill_(q_inf_mask, -float('inf'))
        batch_log_Z = torch.logsumexp(batch_log_f, 1, keepdim=True)
        batch_f = torch.exp(batch_log_f)
        
        batch_Z_prime = (batch_f * batch_log_pq).masked_fill(inf_mask, 0).sum(-1, keepdim=True)
        batch_F = (alpha * torch.exp(batch_log_f - batch_log_Z) * batch_log_pq).masked_fill(inf_mask, 0).sum(-1, keepdim=True) - batch_log_Z - epsilon
        
        batch_Z = batch_f.sum(-1, keepdim=True)
        
        # Derivative F'
        term1 = (batch_log_pq_square * batch_f).masked_fill(inf_mask, 0).sum(-1, keepdim=True) / batch_Z
        term2 = (batch_Z_prime / batch_Z) ** 2
        batch_F_prime = alpha * (term1 - term2)
        
        # Check convergence
        if torch.abs(batch_F).max().item() < 1e-6:
            return alpha.squeeze(-1), batch_log_f - batch_log_Z
        
        # Newton step
        # Avoid division by zero if F_prime is too small
        alpha = alpha - batch_F / (batch_F_prime + 1e-9)
        step += 1

    # Final fallback checks
    if alpha.isnan().any() or (alpha < 0).any():
        valid_mask = (~alpha.isnan()) & (alpha >= 0)
        if not valid_mask.any():
            alpha = torch.ones_like(alpha) * default_alpha
        else:
            avg = torch.masked_select(alpha, valid_mask).mean()
            alpha = torch.where(~valid_mask, avg.unsqueeze(0), alpha)

    # Recalculate final logits
    batch_log_f = (alpha + 1) * batch_p_scores - alpha * batch_q_scores
    batch_log_f.masked_fill_(q_inf_mask, -float('inf'))
    batch_log_Z = torch.logsumexp(batch_log_f, 1, keepdim=True)
    
    return alpha.squeeze(-1), batch_log_f - batch_log_Z