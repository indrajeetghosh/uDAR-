import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from collections import Counter 

def calculate_entropy(predictions):
    predictions = torch.clamp(predictions, 1e-9, 1)  # Prevent log(0)
    entropy = (-predictions * torch.log(predictions)).sum(dim=1).mean().item()
    return entropy
    
def train_and_evaluate(model, source_dataloader, target_dataloader, optimizer, device, alpha, num_epochs=128, num_classes=12):
    ensemble_predictions = torch.zeros(len(target_dataloader.dataset), num_classes).to(device)
    ensemble_counts = torch.zeros(len(target_dataloader.dataset)).to(device)
    best_accuracy = 0.0
    total_loss_list = []
    total_acc_list = []
    average_entropy_list = []

    save_folder = 'Model_checkpoints'
    os.makedirs(save_folder, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        epoch_entropy = []

        source_iter = iter(source_dataloader)
        target_iter = iter(target_dataloader)

        for _ in range(min(len(source_dataloader), len(target_dataloader))):
            source_x, source_y = next(source_iter)
            target_x, target_indices = next(target_iter)

            source_x, source_y, target_x = source_x.to(device), source_y.to(device), target_x.to(device)
            optimizer.zero_grad()

            # Forward pass on source data
            source_pred = model(source_x)
            loss_source = classification_loss_fn(source_pred, source_y)
            correct_predictions += (source_pred.argmax(1) == source_y).sum().item()
            total_predictions += source_y.size(0)

            # Forward pass on target data
            target_pred = model(target_x)
            target_pred_softmax = F.softmax(target_pred, dim=1)

            # Temporal ensembling update
            ensemble_predictions[target_indices] *= alpha
            ensemble_predictions[target_indices] += (1 - alpha) * target_pred_softmax.detach()
            ensemble_counts[target_indices] += 1

            # Calculate entropy for the batch and accumulate
            batch_entropy = calculate_entropy(ensemble_predictions[target_indices] / ensemble_counts[target_indices].unsqueeze(1))
            epoch_entropy.append(batch_entropy)

            # Combine losses and optimize
            loss = loss_source
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Record epoch-wise metrics
        epoch_loss = total_loss / min(len(source_dataloader), len(target_dataloader))
        epoch_accuracy = correct_predictions / total_predictions
        total_loss_list.append(epoch_loss)
        total_acc_list.append(epoch_accuracy)
        average_entropy_list.append(np.mean(epoch_entropy))

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Entropy: {np.mean(epoch_entropy):.4f}")

    return total_loss_list, total_acc_list, average_entropy_list