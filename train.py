import os
import time
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
import logging
import yaml
from dataloader.dataset import processing_data
import datetime
import warnings
warnings.filterwarnings('ignore')

from models.stgcn import EnhancedMultimodalGraph
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except AttributeError:
        print("Warning: CUDA memory functions not available in this PyTorch version")

with open("/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/config.yaml", "r") as stream:
    config_data = yaml.safe_load(stream)

input_dataset_train_dir = config_data['dataset-path-train']
input_dataset_test_dir = config_data['dataset-path-test']
epochs = config_data['epochs']
batch_size = config_data['batch-size']
path_save_model = config_data['project']

print(f"Configuration loaded:")
print(f"   Train dataset directory: {input_dataset_train_dir}")
print(f"   Test dataset directory: {input_dataset_test_dir}")
print(f"   Epochs: {epochs}")
print(f"   Batch size: {batch_size}")

def calculate_enhanced_class_weights(labels, smoothing=0.1):
    if labels.ndim > 1:
        class_indices = np.argmax(labels, axis=1)
    else:
        class_indices = labels.astype(int)
    
    class_counts = np.bincount(class_indices, minlength=4)
    total_samples = len(class_indices)
    num_classes = 4
    
    class_weights = (total_samples + smoothing * num_classes) / (num_classes * (class_counts + smoothing))
    
    print("Enhanced class weights calculated:")
    class_names = ['Vandalism', 'UnauthorizedFilming', 'ClimbOverFence', 'Normal']
    for i, weight in enumerate(class_weights):
        print(f"  Class {i} ({class_names[i]}): {weight:.4f} (count: {class_counts[i]})")
        
    return torch.FloatTensor(class_weights)

def load_multimodal_dataset(data_dir):
    print(f"Loading data from directory: {data_dir}...")
    
    file_map = {
        'X': os.path.join(data_dir, f'X.npy'),
        'y': os.path.join(data_dir, f'y.npy'),
        'bbox': os.path.join(data_dir, f'bbox.npy'),
        'visual': os.path.join(data_dir, f'visual.npy')
    }
    
    datasets = {}
    for key, path in file_map.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        print(f"  - Loading {os.path.basename(path)}...")
        datasets[key] = np.load(path)
        
    print("All datasets loaded successfully.")
    return datasets['X'], datasets['y'], datasets['bbox'], datasets['visual']

try:
    train_pose, train_labels_raw, train_bbox, train_visual = load_multimodal_dataset(input_dataset_train_dir)
except FileNotFoundError as e:
    print(e)
    exit()

try:
    test_pose, test_labels_raw, test_bbox, test_visual = load_multimodal_dataset(input_dataset_test_dir)
except FileNotFoundError as e:
    print(e)
    exit()

print("Enhanced preprocessing pose features...")
train_pose = train_pose[:, ::2, :, :]
test_pose = test_pose[:, ::2, :, :]
print("Applying enhanced pose normalization...")
train_pose[:, :, :, :2] = processing_data(train_pose[:, :, :, :2])
test_pose[:, :, :, :2] = processing_data(test_pose[:, :, :, :2])
print(f"Post-normalization pose range: [{train_pose[:, :, :, :2].min():.3f}, {train_pose[:, :, :, :2].max():.3f}]")
train_visual = train_visual[:, ::2, :]
test_visual = test_visual[:, ::2, :]
train_bbox = train_bbox[:, ::2, :]
test_bbox = test_bbox[:, ::2, :]

if train_labels_raw.ndim > 1 and train_labels_raw.shape[1] > 1:
    num_classes = train_labels_raw.shape[1]
    print(f"Detected one-hot labels with {num_classes} classes")
else:
    num_classes = len(np.unique(train_labels_raw))
    print(f"Detected integer labels with {num_classes} classes")

if num_classes != 4:
    print(f"Warning: Expected 4 classes, got {num_classes}")

print(" --------- Enhanced class analysis (detected {num_classes} classes) ---------")
class_names = ['Vandalism', 'UnauthorizedFilming', 'ClimbOverFence', 'Normal']

if train_labels_raw.ndim > 1 and train_labels_raw.shape[1] > 1:
    train_class_counts = np.bincount(np.argmax(train_labels_raw, axis=1), minlength=num_classes)
    print("Using one-hot labels")
else:
    train_class_counts = np.bincount(train_labels_raw.astype(int), minlength=num_classes)
    print("Using integer labels")
for i in range(num_classes):
    class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
    print(f"class {i} ({class_name}): {train_class_counts[i]} ({train_class_counts[i]/len(train_labels_raw)*100:.1f}%)")

print("Calculating enhanced class weights...")

class_weights = calculate_enhanced_class_weights(train_labels_raw)

print("Preparing labels for CrossEntropyLoss...")

if train_labels_raw.ndim > 1 and train_labels_raw.shape[1] > 1:
    train_labels = np.argmax(train_labels_raw, axis=1)
    test_labels = np.argmax(test_labels_raw, axis=1)
    print("Converted one-hot labels to class indices")
else:
    train_labels = train_labels_raw.astype(int)
    test_labels = test_labels_raw.astype(int)
    print("Using integer labels directly")
    
print("Creating optimized multimodal tensor datasets...")
train_dataset = TensorDataset(
    torch.tensor(train_pose, dtype=torch.float32).permute(0, 3, 1, 2),
    torch.tensor(train_visual, dtype=torch.float32),
    torch.tensor(train_bbox, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(test_pose, dtype=torch.float32).permute(0, 3, 1, 2),
    torch.tensor(test_visual, dtype=torch.float32),
    torch.tensor(test_bbox, dtype=torch.float32),
    torch.tensor(test_labels, dtype=torch.long)
)
print("Optimized multimodal tensor datasets created successfully")

if not os.path.exists(path_save_model):
    os.mkdir(path_save_model)
count = 0
while os.path.exists(path_save_model + f'/exp{count}'):
    count += 1
path_save_model = path_save_model + f'/exp{count}'
os.mkdir(path_save_model)
print(f"Experiment folder created: {path_save_model}")

print("Creating enhanced multimodal data loaders...")
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=True, drop_last=False
)
print("Enhanced multimodal data loaders created successfully")

def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model

classes_name = ['Vandalism', 'UnauthorizedFilming', 'ClimbOverFence', 'Normal']
print(f"Class names ({num_classes} classes):", classes_name)

print("Initializing enhanced multimodal model...")
graph_args = {'strategy': 'spatial'}
model = EnhancedMultimodalGraph(graph_args, num_classes).to(device)
print("Enhanced multimodal model initialized successfully")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
losser = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
print("Enhanced optimization settings initialized")

def train_model_multimodal(model, losser, optimizer, scheduler, num_epochs):
    print(f"Starting enhanced multimodal training for {num_epochs} epochs...")
    
    best_val_acc = -1
    loss_list = {'train': [], 'valid': []}
    acc_list = {'train': [], 'valid': []}
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        losses_train = 0.0
        train_corrects = 0
        valid_batches_train = 0
        model.train()
        
        pbar_train = tqdm(train_loader, desc=f'Training', unit='batch')
        
        for batch_idx, data in enumerate(pbar_train):
            try:
                batch_pose, batch_visual, batch_bbox, labels = data
                
                if batch_pose.shape[2] < 2:
                    continue
                    
                mot = batch_pose[:, :2, 1:, :] - batch_pose[:, :2, :-1, :]
                
                min_T = mot.shape[2]
                batch_pose_resized = batch_pose[:, :, :min_T, :]
                batch_visual_resized = batch_visual[:, :min_T, :]
                batch_bbox_resized = batch_bbox[:, :min_T, :]
                
                if min_T == 0:
                     continue
                
                batch_pose_resized = batch_pose_resized.to(device)
                mot = mot.to(device)
                batch_visual_resized = batch_visual_resized.to(device)
                batch_bbox_resized = batch_bbox_resized.to(device)
                labels = labels.to(device)
                
                outputs = model((batch_pose_resized, mot, batch_visual_resized, batch_bbox_resized))
                
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    continue
                
                loss = losser(outputs, labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                losses_train += loss.item()
                
                with torch.no_grad():
                    _, predicted_classes = torch.max(outputs, dim=1)
                    accuracy = (predicted_classes == labels).float().mean()
                    train_corrects += accuracy.item()
                
                valid_batches_train += 1
                
                del batch_pose, batch_visual, batch_bbox, labels, mot, data
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        epoch_loss_train = losses_train / valid_batches_train if valid_batches_train > 0 else float('inf')
        loss_list['train'].append(epoch_loss_train)
        epoch_acc_train = train_corrects / valid_batches_train if valid_batches_train > 0 else 0
        acc_list['train'].append(epoch_acc_train)
        
        print(f"  Train - Acc: {epoch_acc_train:.4f}, Loss: {epoch_loss_train:.4f}")
        
        losses_val = 0.0
        val_corrects = 0
        valid_batches_val = 0
        model.eval()
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f'Validation', unit='batch')
            for batch_idx, data in enumerate(pbar_val):
                try:
                    batch_pose, batch_visual, batch_bbox, labels = data
                    
                    if batch_pose.shape[2] < 2:
                        continue
                    
                    mot = batch_pose[:, :2, 1:, :] - batch_pose[:, :2, :-1, :]
                    
                    min_T = mot.shape[2]
                    batch_pose_resized = batch_pose[:, :, :min_T, :]
                    batch_visual_resized = batch_visual[:, :min_T, :]
                    batch_bbox_resized = batch_bbox[:, :min_T, :]
                    
                    if min_T == 0:
                        continue
                    
                    batch_pose_resized = batch_pose_resized.to(device)
                    mot = mot.to(device)
                    batch_visual_resized = batch_visual_resized.to(device)
                    batch_bbox_resized = batch_bbox_resized.to(device)
                    labels = labels.to(device)
                    
                    outputs = model((batch_pose_resized, mot, batch_visual_resized, batch_bbox_resized))
                    
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        continue
                    
                    loss = losser(outputs, labels)
                    
                    if torch.isnan(loss):
                        continue
                    
                    losses_val += loss.item()
                    
                    _, predicted_classes = torch.max(outputs, dim=1)
                    accuracy = (predicted_classes == labels).float().mean()
                    val_corrects += accuracy.item()
                    valid_batches_val += 1
                    
                    del batch_pose, batch_visual, batch_bbox, labels, mot, data
                    torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        epoch_loss_val = losses_val / valid_batches_val if valid_batches_val > 0 else float('inf')
        loss_list['valid'].append(epoch_loss_val)
        epoch_acc_val = val_corrects / valid_batches_val if valid_batches_val > 0 else 0
        acc_list['valid'].append(epoch_acc_val)
        
        print(f"  Valid - Acc: {epoch_acc_val:.4f}, Loss: {epoch_loss_val:.4f}")
        
        if epoch_acc_val > best_val_acc:
            best_val_acc = epoch_acc_val
            torch.save(model.state_dict(), path_save_model + '/best.pt')
            print(f"Saved best model (acc: {best_val_acc:.4f})")
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            fig = plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(acc_list['train'], label="Train Accuracy", linewidth=2)
            plt.plot(acc_list['valid'], label="Val Accuracy", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Multimodal Training - Accuracy")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(loss_list['train'], label="Train Loss", linewidth=2)
            plt.plot(loss_list['valid'], label="Val Loss", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Multimodal Training - Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(path_save_model + '/result.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            del fig
    
    return model

def main():
    print("Starting enhanced 4-class multimodal training...")
    model_trained = train_model_multimodal(model, losser, optimizer, scheduler, num_epochs=epochs)
    torch.save(model_trained.state_dict(), path_save_model + '/last.pt')
    logging.warning('Saved last model at {}'.format(path_save_model + "/last.pt"))
    print(f"Training completed! Models saved to: {path_save_model}")
    print("Enhanced multimodal model supports 4 classes with pose + visual + bbox features")
    
if __name__ == '__main__':
    main()