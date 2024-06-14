import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


weight_kl = 0.3
num_epochs = 50
total_loss_list = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    min_len = min(len(source_dataloader), len(target_dataloader))

    for _ in range(min_len):
        source_data = next(iter(source_dataloader))
        target_data = next(iter(target_dataloader))

    # Unpack and convert to tensor if necessary
        source_x, source_y = source_data if isinstance(source_data, tuple) else (source_data[0], source_data[1])
        target_x = target_data if isinstance(target_data, torch.Tensor) else target_data[0]

    # Move to device
        source_x, source_y, target_x = source_x.to(device), source_y.to(device), target_x.to(device)
        optimizer.zero_grad()

        # Process source labels
        if source_y.ndim > 1 and source_y.shape[1] > 1:
            source_y = torch.argmax(source_y, dim=1)

        # Forward pass on source and target data
        source_pred = model(source_x)
        target_pred = model(target_x)

        # Classification loss on source domain
        loss_source = classification_loss_fn(source_pred, source_y)
        correct_predictions += (source_pred.argmax(1) == source_y).sum().item()
        total_predictions += source_y.size(0)

        # KL Divergence loss for domain adaptation
        loss_kl = kl_divergence_loss(source_pred, target_pred)

        # Combine losses
        loss = loss_source + weight_kl * loss_kl
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    epoch_loss = total_loss / min_len
    epoch_accuracy = correct_predictions / total_predictions
    total_loss_list.append(epoch_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")