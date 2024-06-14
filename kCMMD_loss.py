#Classwise Alignmnet without gaussian kernel based:- Regularization is used because CMMD loss essentially measures the squared norm of the difference between the mean features of corresponding classes from source and target dataset i.e. if the means are very close to each other, the squared norm might become very small/neglible, potentially leading to numerical instability during optimization. This regularization is done direclt on feature space.

#Regularization based CMMD loss:- Kernel (regularization added directly to the kernel matrix, impacting the kernelized feature space - motivation is prevent overfitting and to enhance the stability - singularity or ill-conditioning during during matrix inversion or eigenvalue decomposition.

import torch
import torch.nn.functional as F

def rbf_kernel(X1, X2, gamma=1.0):
    """
    Compute the RBF (Gaussian) kernel between X1 and X2.
    :param X1: A tensor of size [n_samples_1, n_features].
    :param X2: A tensor of size [n_samples_2, n_features].
    :param gamma: Kernel coefficient.
    """
    sq_dist = torch.cdist(X1, X2)**2
    return torch.exp(-gamma * sq_dist)
    

def cmmd_loss(source_features, source_labels, target_features, target_pseudo_labels, num_classes, gamma=1.0, lambda_reg=0.6):
    cmmd_loss = 0.0
    classes = 0

    for class_idx in range(num_classes):
        source_class_features = source_features[source_labels == class_idx]
        target_class_features = target_features[target_pseudo_labels == class_idx]

        # Skip if either class has too few samples
        if len(source_class_features) < 100 or len(target_class_features) < 100:
            continue

        # Compute the kernel matrices with regularization
        K_ss = rbf_kernel(source_class_features, source_class_features, gamma) + lambda_reg * torch.eye(len(source_class_features)).to(source_features.device)
        K_tt = rbf_kernel(target_class_features, target_class_features, gamma) + lambda_reg * torch.eye(len(target_class_features)).to(target_features.device)
        K_st = rbf_kernel(source_class_features, target_class_features, gamma)

        # Compute mean discrepancy in kernel space
        mean_discrepancy = torch.mean(K_ss) + torch.mean(K_tt) - 2 * torch.mean(K_st)

        cmmd_loss += mean_discrepancy
        classes += 1

    if classes == 0:
        return torch.tensor(0.0).to(source_features.device)

    return cmmd_loss / classes
