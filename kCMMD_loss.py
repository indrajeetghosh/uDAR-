#Classwise Alignmnet without gaussian kernel based:- Regularization is used because CMMD loss essentially measures the squared norm of the difference between the mean features of corresponding classes from source and target dataset i.e. if the means are very close to each other, the squared norm might become very small/neglible, potentially leading to numerical instability during optimization. This regularization is done direclt on feature space.

#Regularization based CMMD loss:- Kernel (regularization added directly to the kernel matrix, impacting the kernelized feature space - motivation is prevent overfitting and to enhance the stability - singularity or ill-conditioning during during matrix inversion or eigenvalue decomposition.

import torch
import torch.nn.functional as F

def rbf_kernel(X1, X2, gamma=1.0):
    # Compute squared Euclidean distances between each pair of vectors
    sq_dist = torch.cdist(X1, X2, p=2) ** 2
    # Compute the RBF kernel
    K = torch.exp(-gamma * sq_dist)
    return K

def cmmd_loss(source_features, source_labels, target_features, target_pseudo_labels, num_classes, gamma=1.0, lambda_reg=1e-3):
    """
    Compute the CMMD loss excluding self-similarity bias and including regularization.
    
    Parameters:
    - source_features: Tensor of source domain features [n_source_samples, feature_dim]
    - source_labels: Tensor of source domain labels [n_source_samples]
    - target_features: Tensor of target domain features [n_target_samples, feature_dim]
    - target_pseudo_labels: Tensor of target domain pseudo labels [n_target_samples]
    - num_classes: Total number of classes (C)
    - gamma: Kernel bandwidth parameter
    - lambda_reg: Regularization parameter (lambda)
    
    Returns:
    - Scalar tensor representing the CMMD loss
    """
    device = source_features.device
    total_loss = 0.0
    classes_present = 0  # Counts classes present in both domains

    for class_idx in range(num_classes):
        # Extract features for class class_idx
        source_mask = (source_labels == class_idx)
        target_mask = (target_pseudo_labels == class_idx)

        source_class_features = source_features[source_mask]
        target_class_features = target_features[target_mask]

        n_s = source_class_features.size(0)
        n_t = target_class_features.size(0)

        # Skip if either class has too few samples
        if n_s < 100 or n_t < 100:
            continue

        # Compute K_ss (source-source) and exclude self-similarity
        K_ss = rbf_kernel(source_class_features, source_class_features, gamma)
        # Exclude self-similarity by zeroing the diagonal
        K_ss_no_diag = K_ss - torch.diag(torch.diag(K_ss))
        # Add regularization to diagonal
        K_ss_reg = K_ss_no_diag + lambda_reg * torch.eye(n_s, device=device)

        # Compute K_tt (target-target) and exclude self-similarity
        K_tt = rbf_kernel(target_class_features, target_class_features, gamma)
        # Exclude self-similarity by zeroing the diagonal
        K_tt_no_diag = K_tt - torch.diag(torch.diag(K_tt))
        # Add regularization to diagonal
        K_tt_reg = K_tt_no_diag + lambda_reg * torch.eye(n_t, device=device)

        # Compute K_st (source-target)
        K_st = rbf_kernel(source_class_features, target_class_features, gamma)

        # Compute mean values excluding self-similarity
        mean_K_ss = K_ss_no_diag.sum() / (n_s * (n_s - 1))
        mean_K_tt = K_tt_no_diag.sum() / (n_t * (n_t - 1))
        mean_K_st = K_st.sum() / (n_s * n_t)

        # Compute CMMD loss for class class_idx
        loss_c = mean_K_ss + mean_K_tt - 2 * mean_K_st

        total_loss += loss_c
        classes_present += 1

    if classes_present == 0:
        # Return zero loss if no classes are present in both domains
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Average the loss over the number of classes present
    cmmd_loss = total_loss / classes_present
    return cmmd_loss
