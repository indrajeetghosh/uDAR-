import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

ensemble_predictions = torch.zeros(len(target_dataset), num_classes).to(device)
ensemble_counts = torch.zeros(len(target_dataset)).to(device)

best_model_state_dict = None
best_accuracy = 0.0
weight_cmmd = 1.0
weight_kl = 1.0
num_epochs = 128
num_classes = 12
alpha = 0.6
total_loss_list = []
total_acc_list = []

save_folder = 'Model_checkpoints'
os.makedirs(save_folder, exist_ok=True) 

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

        # Forward pass on source data
        optimizer.zero_grad()
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

        # Generate pseudo labels from ensemble
        ensemble_avg = ensemble_predictions[target_indices] / ensemble_counts[target_indices].unsqueeze(1)
        target_pseudo_labels = torch.argmax(ensemble_avg, dim=1)
        
        
        # CMMD loss
        loss_cmmd = cmmd_loss(source_pred, source_y, target_pred, target_pseudo_labels, num_classes)

        
        target_x_augmented = augment_data(target_x).to(device)

        target_pred_augmented = model(target_x_augmented)
        loss_kl = kl_div_loss(F.log_softmax(target_pred_augmented, dim=1), F.softmax(model(target_x), dim=1)) #target_pred_augmented, target_pred_softmax)
        

        loss = loss_source + weight_cmmd * loss_cmmd + weight_kl * loss_kl
        
        total_loss += loss.item()

        # Backward and optimize
        loss.backward()
        optimizer.step()

    epoch_loss = total_loss / min(len(source_dataloader), len(target_dataloader))
    epoch_accuracy = correct_predictions / total_predictions
    total_loss_list.append(epoch_loss)
    total_acc_list.append(epoch_accuracy)
    

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    
    if  epoch_accuracy  > best_accuracy:
        best_accuracy = epoch_accuracy 
        best_model_state_dict = model.state_dict()

# Save the best model checkpoint
        if best_model_state_dict is not None:

    # Save the best model checkpoint with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"best_model_{timestamp}_accuracy{best_accuracy:.4f}.pt"
            save_path = os.path.join(save_folder, filename)
            #torch.save(best_model_state_dict, save_path)