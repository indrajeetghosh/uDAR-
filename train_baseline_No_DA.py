import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

total_loss_list = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    min_len = len(source_dataloader)#, len(target_dataloader))

    for i in range(min_len):
        source_x, source_y = next(iter(source_dataloader))
        #target_x = next(iter(target_dataloader))

        source_x, source_y = source_x.to(device), source_y.to(device)
        optimizer.zero_grad()

        if source_y.ndim > 1 and source_y.shape[1] > 1:
            source_y = torch.argmax(source_y, dim=1)

        source_pred = model(source_x)
        loss_source = classification_loss_fn(source_pred, source_y)
        _, predicted_labels = torch.max(source_pred, 1)
        correct_predictions += (predicted_labels == source_y).sum().item()
        total_predictions += source_y.size(0)

        # loss
        loss = loss_source 
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    epoch_loss = total_loss / min_len
    epoch_accuracy = correct_predictions / total_predictions
    total_loss_list.append(epoch_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")