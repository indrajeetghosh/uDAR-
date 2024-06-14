import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


best_model_state_dict = None
best_accuracy = 0.0
weight_cmmd = 1.0
weight_kl = 1.0
num_epochs = 128
num_classes = classes
alpha = 0.6
total_loss_list = []
total_acc_list = []

save_folder = 'Model_checkpoints'
os.makedirs(save_folder, exist_ok=True)
# Initialize placeholders for temporal ensembling
ensemble_predictions = torch.zeros(len(target_dataloader.dataset), num_classes).to(device)
ensemble_counts = torch.zeros(len(target_dataloader.dataset)).to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    source_iter = iter(source_dataloader)
    target_iter = iter(target_dataloader)

    for _ in range(min(len(source_dataloader), len(target_dataloader))):
        source_x, source_y = next(source_iter)
        target_x, target_indices = next(target_iter)

        source_x, source_y, target_x = source_x.to(device), source_y.to(device), target_x.to(device)

        # Apply augmentation to both source and target data
        source_x_augmented = augment_data(source_x).to(device)
        target_x_augmented = augment_data(target_x).to(device)

        optimizer.zero_grad()

        # Original source prediction and loss
        source_pred = model(source_x)
        loss_source = classification_loss_fn(source_pred, source_y)
        
        
        # Forward pass on augmented source data
        source_pred_augmented = model(source_x_augmented)
        loss_source_aug = kl_divergence_loss(source_pred_augmented, F.softmax(model(source_x), dim=1))

        correct_predictions += (source_pred.argmax(1) == source_y).sum().item()
        total_predictions += source_y.size(0)

        # Original target prediction and ensemble update
        target_pred = model(target_x)
        target_pred_softmax = F.softmax(target_pred, dim=1)
        
        ensemble_predictions[target_indices] *= alpha
        ensemble_predictions[target_indices] += (1 - alpha) * target_pred_softmax.detach()
        ensemble_counts[target_indices] += 1

        ensemble_avg = ensemble_predictions[target_indices] / ensemble_counts[target_indices].unsqueeze(1)
        target_pseudo_labels = torch.argmax(ensemble_avg, dim=1)

        # CMMD loss for original predictions
        loss_cmmd = cmmd_loss(source_pred, source_y, target_pred, target_pseudo_labels, num_classes)
        
                # Forward pass on augmented target data
        target_pred_augmented = model(target_x_augmented)
        loss_target_aug = kl_divergence_loss(target_pred_augmented, F.softmax(model(target_x), dim=1))
        
     
        loss = loss_source + weight_cmmd * loss_cmmd + weight_kl * loss_source_aug + weight_kl * loss_target_aug

        total_loss += loss.item()

        # Backward and optimize
        loss.backward()
        optimizer.step()

    epoch_loss = total_loss / min(len(source_dataloader), len(target_dataloader))
    epoch_accuracy = correct_predictions / total_predictions
    total_loss_list.append(epoch_loss)
    total_acc_list.append(epoch_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Update best model based on accuracy
    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        best_model_state_dict = model.state_dict()

        # Save the best model checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_model_{timestamp}_accuracy{best_accuracy:.4f}.pt"
        save_path = os.path.join(save_folder, filename)
        torch.save(best_model_state_dict, save_path)
