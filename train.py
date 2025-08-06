import os  
import time  
import torch  
import pickle  
import numpy as np  
import torch.nn.functional as F  
import matplotlib.pyplot as plt  
import matplotlib  
matplotlib.use('Agg')  
from tqdm import tqdm  
from torch.utils import data  
from torch.optim.adadelta import Adadelta  
from models.stgcn import *  
from torch.utils.data import DataLoader, TensorDataset  
from collections import OrderedDict  
import logging  
import yaml  
from dataloader.dataset import processing_data  
import datetime  
  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
print(f"Using device: {device}")  
  
if torch.cuda.is_available():  
    try:  
        torch.cuda.empty_cache()  
        torch.cuda.reset_peak_memory_stats()  
    except AttributeError:  
        print("Warning: CUDA memory functions not available in this PyTorch version")  
  
# Get parameter - Load config.yaml  
with open("/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/config.yaml", "r") as stream:  
    config_data = yaml.safe_load(stream)  
  
# Updated parameters for 4 classes dataset  
input_dataset_train = config_data['dataset-path-train']  # Should point to train_4classes.pkl  
input_dataset_test = config_data['dataset-path-test']    # Should point to test_4classes.pkl  
epochs = config_data['epochs']  
batch_size = config_data['batch-size']  
input_size = config_data['img-size']  
num_frame = config_data['num-frame']  
path_save_model = config_data['project']  
  
print(f"Configuration loaded:")  
print(f"  Train dataset: {input_dataset_train}")  
print(f"  Test dataset: {input_dataset_test}")  
print(f"  Epochs: {epochs}")  
print(f"  Batch size: {batch_size}")  
  

def label_smoothing(labels, smoothing=0.1):  
    """Apply label smoothing to one-hot labels"""  
    num_classes = labels.shape[1]  
    return labels * (1 - smoothing) + smoothing / num_classes  
  
# Load dataset train - UPDATED for 4 classes with bbox data  
print("Loading training dataset...")  
with open(input_dataset_train, 'rb') as f:  
    fts, lbs, bbox_fts = pickle.load(f)  
    features = [fts]  
    labels = [lbs]  
    # bbox_features = [bbox_fts]  # Load but don't use for TwoStreamSpatialTemporalGraph  
del fts, lbs, bbox_fts  
print("Training dataset loaded successfully")  
  
# Data preprocessing  
labels = np.concatenate(labels, axis=0)  
features = np.concatenate(features, axis=0)  # 30x17x3  
  
# Frame sampling: 30 -> 15 frames  
features = features[:, ::2, :, :]  
  
# Pose normalization  
features[:, :, :, :2] = processing_data(features[:, :, :, :2])  
  
# FIXED: Apply label smoothing to prevent extreme values  
labels = label_smoothing(labels, smoothing=0.1)  
  
x_train = features  
y_train = labels  
  
print(" --------- Number class train---------")  
for i in range(4):
    print(f"class {i}: {np.argmax(labels, axis=1).tolist().count(i)}")  
  
# Load test dataset  
print("Loading test dataset...")  
with open(input_dataset_test, 'rb') as f:  
    fts, lbs, bbox_fts = pickle.load(f)  
    features = [fts]  
    labels = [lbs]  

del fts, lbs, bbox_fts  
print("Test dataset loaded successfully")  
  
labels = np.concatenate(labels, axis=0)  
features = np.concatenate(features, axis=0)  
  
features = features[:, ::2, :, :]  
features[:, :, :, :2] = processing_data(features[:, :, :, :2])  
  
# FIXED: Apply label smoothing to test labels too  
labels = label_smoothing(labels, smoothing=0.1)  
  
x_valid = features  
y_valid = labels  
  
print(" --------- Number class test---------")  
for i in range(4):  # Changed from 5 to 4  
    print(f"class {i}: {np.argmax(labels, axis=1).tolist().count(i)}")  
  
del features, labels  
  
print("Creating tensor datasets...")  
train_dataset = TensorDataset(  
    torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),  
    torch.tensor(y_train, dtype=torch.float32)  
)  
val_dataset = TensorDataset(  
    torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),  
    torch.tensor(y_valid, dtype=torch.float32)  
)  
  
del x_train, x_valid, y_train, y_valid  
print("Tensor datasets created successfully")  
  
if not os.path.exists(path_save_model):  
    os.mkdir(path_save_model)  
count = 0  
while os.path.exists(path_save_model + f'/exp{count}'):  
    count += 1  
path_save_model = path_save_model + f'/exp{count}'  
os.mkdir(path_save_model)  
print(f"Experiment folder created: {path_save_model}")  
  
print("Creating data loaders...")  
train_loader = DataLoader(  
    train_dataset,  
    batch_size=batch_size,  
    shuffle=True,  
    num_workers=0 if device.type == 'cpu' else min(8, batch_size),  
    pin_memory=torch.cuda.is_available()  
)  
  
val_loader = DataLoader(  
    val_dataset,  
    batch_size=batch_size,  
    shuffle=False,  
    num_workers=0 if device.type == 'cpu' else min(8, batch_size),  
    pin_memory=torch.cuda.is_available()  
)  
  
del train_dataset, val_dataset  
print("Data loaders created successfully")  
  
def set_training(model, mode=True):  
    for p in model.parameters():  
        p.requires_grad = mode  
    model.train(mode)  
    return model  
  
 
classes_name = ['Vandalism', 'LookingAround', 'UnauthorizedFilming', 'ClimbOverFence']  
print("Class name:", classes_name)  
  
print("Initializing model...")  
graph_args = {'strategy': 'spatial'}  
# Changed from ThreeStreamSpatialTemporalGraph to TwoStreamSpatialTemporalGraph  
model = TwoStreamSpatialTemporalGraph(graph_args, len(classes_name)).to(device)  
print("Model initialized successfully")  
  
if torch.cuda.device_count() > 1:  
    print(f"Using {torch.cuda.device_count()} GPUs")  
    model = torch.nn.DataParallel(model)  
  
# FIXED: Reduced learning rate and enhanced optimizer  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  
losser = torch.nn.BCELoss()  
print("Optimizer and scheduler initialized")  
  
def train_model(model, losser, optimizer, scheduler, num_epochs):  
    print(f"Starting training for {num_epochs} epochs...")  
      
    best_loss_acc = -1  
    loss_list = {'train': [], 'valid': []}  
    acc_list = {'train': [], 'valid': []}  
      
    for epoch in range(num_epochs):  
        print(f"Epoch {epoch+1}/{num_epochs}")  
          
        # Training phase  
        losses_train = 0.0  
        train_corrects = 0  
        last_time = time.time()  
        model = set_training(model, True)  
          
        pbar_train = tqdm(train_loader, desc=f'Training', unit='batch')  
          
        for batch_idx, (batch_vid, labels) in enumerate(pbar_train):  # Removed bbox_data  
            try:  
                # Move to device  
                mot = batch_vid[:, :2, 1:, :] - batch_vid[:, :2, :-1, :]  
                mot, batch_vid, labels = mot.to(device), batch_vid.to(device), labels.to(device)  
                  
                # Forward pass with two streams (pose + motion) - removed bbox_data  
                outputs = model((batch_vid, mot))  
                  
                # FIXED: Add validation and clipping to prevent CUDA assertion  
                outputs = torch.clamp(outputs, min=1e-7, max=1-1e-7)  
                labels = torch.clamp(labels, min=0.0, max=1.0)  
                  
                # Check for NaN/Inf values  
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():  
                    print(f"Warning: NaN/Inf detected in outputs at batch {batch_idx}, skipping batch")  
                    continue  
                  
                if torch.isnan(labels).any() or torch.isinf(labels).any():  
                    print(f"Warning: NaN/Inf detected in labels at batch {batch_idx}, skipping batch")  
                    continue  
                  
                loss = losser(outputs, labels)  
                  
                # Check for NaN loss  
                if torch.isnan(loss):  
                    print(f"Warning: NaN loss detected at batch {batch_idx}, skipping batch")  
                    continue  
                  
                # Backward pass  
                optimizer.zero_grad()  
                loss.backward()  
                  
                # FIXED: Enhanced gradient clipping  
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  
                optimizer.step()  
                  
                losses_train += loss.item()  
                _, preds = torch.max(outputs, 1)  
                train_corrects += (preds == labels.data.argmax(1)).detach().cpu().numpy().mean()  
                  
                # Memory cleanup  
                del batch_vid, labels, mot  
                if torch.cuda.is_available():  
                    torch.cuda.empty_cache()  
                  
                # Progress monitoring  
                if torch.cuda.is_available():  
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  
                    pbar_train.set_postfix(OrderedDict({  
                        'Loss': f"{loss.item():.4f}",  
                        'GPU Memory': f"{gpu_memory:.2f} GB"  
                    }))  
                else:  
                    pbar_train.set_postfix(OrderedDict({  
                        'Loss': f"{loss.item():.4f}",  
                        'Batch': f"{batch_idx+1}/{len(train_loader)}"  
                    }))  
                  
            except Exception as e:  
                print(f"Error in training batch {batch_idx}: {e}")  
                continue  
          
        # Update learning rate  
        scheduler.step()  
          
        epoch_loss = losses_train / len(train_loader)  
        loss_list['train'].append(epoch_loss)  
        epoch_acc = train_corrects / len(train_loader)  
        acc_list['train'].append(epoch_acc)  
          
        print(f"  Train - Acc: {epoch_acc:.4f}, Loss: {epoch_loss:.4f}")  
          
        # Validation phase  
        losses_val = 0.0  
        val_corrects = 0  
        model = set_training(model, False)  
          
        with torch.no_grad():  
            pbar_val = tqdm(val_loader, desc=f'Validation', unit='batch')  
            for batch_idx, (batch_vid, labels) in enumerate(pbar_val):  # Removed bbox_data  
                try:  
                    mot = batch_vid[:, :2, 1:, :] - batch_vid[:, :2, :-1, :]  
                    mot, batch_vid, labels = mot.to(device), batch_vid.to(device), labels.to(device)  
                      
                    outputs = model((batch_vid, mot))  # Removed bbox_data  
                      
                    # FIXED: Apply same validation for validation  
                    outputs = torch.clamp(outputs, min=1e-7, max=1-1e-7)  
                    labels = torch.clamp(labels, min=0.0, max=1.0)  
                      
                    # Skip batch if NaN/Inf detected  
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any() or torch.isnan(labels).any():  
                        continue  
                      
                    loss = losser(outputs, labels)  
                      
                    if torch.isnan(loss):  
                        continue  
                      
                    losses_val += loss.item()  
                    _, preds = torch.max(outputs, 1)  
                    val_corrects += (preds == labels.data.argmax(1)).detach().cpu().numpy().mean()  
                      
                    del batch_vid, labels, mot  
                      
                except Exception as e:  
                    print(f"Error in validation batch {batch_idx}: {e}")  
                    continue  
          
        epoch_loss = losses_val / len(val_loader)  
        loss_list['valid'].append(epoch_loss)  
        epoch_acc = val_corrects / len(val_loader)  
        acc_list['valid'].append(epoch_acc)  
          
        print(f"  Valid - Acc: {epoch_acc:.4f}, Loss: {epoch_loss:.4f}")  
          
        # Save best model  
        if best_loss_acc == -1 or best_loss_acc <= epoch_acc:  
            best_loss_acc = epoch_acc  
            torch.save(model.state_dict(), path_save_model + '/best.pt')  
            print(f"Saved best model (acc: {best_loss_acc:.4f})")  
          
        # Save plots every 10 epochs  
        if (epoch + 1) % 10 == 0:  
            fig = plt.figure(figsize=(12, 5))  
            plt.subplot(1, 2, 1)  
            plt.plot(acc_list['train'], label="Train Accuracy")  
            plt.plot(acc_list['valid'], label="Val Accuracy")  
            plt.xlabel("Epoch")  
            plt.ylabel("Accuracy")  
            plt.title("Training Progress - Accuracy")  
            plt.legend()  
            plt.grid(True)  
              
            plt.subplot(1, 2, 2)  
            plt.plot(loss_list['train'], label="Train Loss")  
            plt.plot(loss_list['valid'], label="Val Loss")  
            plt.xlabel("Epoch")  
            plt.ylabel("Loss")  
            plt.title("Training Progress - Loss")  
            plt.legend()  
            plt.grid(True)  
              
            plt.tight_layout()  
            fig.savefig(path_save_model + '/result.png', dpi=500)  
            plt.close(fig)  
            del fig  
      
    return model  
  
def main():  
    """  
    function: training model  
    :return:  
    """  
    model_trained = train_model(model, losser, optimizer, scheduler, num_epochs=epochs)  
    torch.save(model_trained.state_dict(), path_save_model + '/last.pt')  
    logging.warning('Saved last model at {}'.format(path_save_model + "/last.pt"))  
    print("Training completed!")  
  
if __name__ == '__main__':  
    main()
