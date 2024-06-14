import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import seaborn as sns
from sklearn.metrics import accuracy_score
from pytorch_metric_learning import losses
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

model.eval()  # Set the model to evaluation mode
true_labels = []
pred_labels = []
embeddings = []

with torch.no_grad():  # No need to track gradients
    for source_batch, target_batch in zip(eval_source_dataloader, eval_target_dataloader):

        target_inputs, target_labels = target_batch
        target_inputs = target_inputs.to(device)
        target_labels = target_labels.to(device)
        target_outputs = model(target_inputs)
        target_features = target_outputs  
        _, predicted = torch.max(target_outputs, 1)
        true_labels.extend(target_labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())
        embeddings.append(target_features.detach().cpu().numpy())
        
    # Calculate the macro F1 score
    
#classes = ['Forehand Service','Backhand Service','Clear Lob Overhead Forehand','Clear Lob Overhead Backhand',
 #          'Clear Lob Underarm Forehand','Clear Lob Underarm Backhand','Net Shot Underarm Forehand',
  #         'Net Shot Underarm Backhand','Drop Shot Overhead Forehand','Drop Shot Overhead Backhand',
   #       'Smash Overhead Forehand','Smash Overhead Backhand']

classes = ['FS','BS','CLOF','CLOB','CLUF','CLUB','NSUF','NSUB','DSOF','DSOB','SOF','SOB']
    

# Confusion matrix
cf_matrix = confusion_matrix(true_labels, pred_labels)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,10))
sns.heatmap(df_cm, annot=True)
plt.xlabel('Class Labels', fontweight='semibold')
plt.ylabel('Class Labels', fontweight='semibold')
#plt.savefig("Images/New_Confusion_Matrix_BAR_Gender_Variations.png", format='png',bbox_inches='tight', pad_inches=0,dpi=300)
plt.show()

macro_f1 = f1_score(true_labels, pred_labels, average='macro')
precision = precision_score(true_labels, pred_labels, average = 'macro')
recall = recall_score(true_labels, pred_labels, average = 'macro')
accuracy = accuracy_score(true_labels, pred_labels)
print(macro_f1, precision, recall, accuracy)
# Calculating AUROC

# Create a DataFrame to hold the metrics
data = {
    'Macro F1-Score': [macro_f1],
    'Precision': [precision],
    'Recall': [recall],
    'Accuracy': [accuracy]
}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
#df.to_csv('model_metrics_only_UDA.csv', index=False)

print("Metrics saved to model_metrics.csv")
