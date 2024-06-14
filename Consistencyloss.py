import torch

def kl_for_log_probs(log_p, prob_q):
    """
    Compute KL divergence between log probabilities and probabilities.
    Args:
        log_p: Tensor of log probabilities (predicted distribution, log P).
        prob_q: Tensor of probabilities (target distribution, Q).

    Returns:
        KL divergence for each example in the batch.
    """
    p = torch.exp(log_p)  # Convert log P to P
    eps = torch.finfo(prob_q.dtype).eps  # Machine epsilon for numerical stability
    kl = torch.sum(p * (log_p - torch.log(prob_q + eps)), dim=-1)
    return kl

def kl_div_loss(tgt_ori_log_probs, aug_probs):
    """
    Calculate the mean KL divergence loss for a batch of data in PyTorch.
    Args:
        tgt_ori_log_probs: Tensor of log probabilities from the original target data (log P).
        aug_probs: Tensor of probabilities from the augmented data (Q).

    Returns:
        Mean KL divergence loss.
    """
    per_example_kl_loss = kl_for_log_probs(tgt_ori_log_probs, aug_probs)
    mean_kl_loss = torch.mean(per_example_kl_loss)
    return mean_kl_loss
