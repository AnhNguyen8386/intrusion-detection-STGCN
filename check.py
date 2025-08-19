import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.stgcn import *
from sklearn import metrics
from util.plot import plot_cm
from tqdm import tqdm
from dataloader.dataset import processing_data
import torch.nn as nn
import os
import yaml

# Corrected: Enhanced Multimodal Architecture to match the 672-input checkpoint
class EnhancedMultimodalGraph(nn.Module):
    def __init__(self, graph_args, num_class, edge_importance_weighting=True, **kwargs):
        super().__init__()
        
        self.pts_stream = StreamSpatialTemporalGraph(3, graph_args, None, edge_importance_weighting, **kwargs)
        self.mot_stream = StreamSpatialTemporalGraph(2, graph_args, None, edge_importance_weighting, **kwargs)

        self.visual_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # CORRECTED: Bbox FC with output 32 to match 672 total
        self.bbox_fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # CORRECTED: Final fusion input is 256+256+128+32 = 672
        self.fcn = nn.Sequential(
            nn.Linear(672, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_class)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        pose_data, motion_data, visual_data, bbox_data = inputs
        
        pose_out = self.pts_stream(pose_data)       # (B, 256)
        motion_out = self.mot_stream(motion_data)   # (B, 256)
        pose_motion_concat = torch.cat([pose_out, motion_out], dim=-1) # (B, 512)
        
        visual_pooled = torch.mean(visual_data, dim=1) # (B, 512)
        visual_out = self.visual_fc(visual_pooled)      # (B, 128)
        
        bbox_pooled = torch.mean(bbox_data, dim=1)      # (B, 4)
        bbox_out = self.bbox_fc(bbox_pooled)            # (B, 32)
        
        concat_features = torch.cat([pose_motion_concat, visual_out, bbox_out], dim=-1) # (B, 672)
        out = self.fcn(concat_features)
        
        return out

def load_multimodal_dataset_npy(data_dir):
    """Load multimodal dataset from .npy files"""
    print(f"Loading data from directory: {data_dir}...")
    
    file_map = {
        'X': os.path.join(data_dir, 'X_test.npy'),
        'y': os.path.join(data_dir, 'y_test.npy'),
        'bbox': os.path.join(data_dir, 'bbox_test.npy'),
        'visual': os.path.join(data_dir, 'visual_test.npy')
    }
    
    datasets = {}
    for key, path in file_map.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        print(f"  - Loading {os.path.basename(path)}...")
        datasets[key] = np.load(path)
        
    print("All datasets loaded successfully.")
    return datasets['X'], datasets['y'], datasets['bbox'], datasets['visual']

def test_enhanced_multimodal_confusion_matrix(path_test_dir, path_model, batch_size=64):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    class_names = ['Vandalism', 'UnauthorizedFilming', 'ClimbOverFence', 'Normal']
    num_classes = len(class_names)
    
    # FIXED: Use 'spatial' strategy to resolve the ValueError
    graph_args = {'strategy': 'spatial'}
    model = EnhancedMultimodalGraph(graph_args, num_classes).to(device)
    
    try:
        # Checkpoint loading with strict=True to enforce matching architecture
        checkpoint = torch.load(path_model, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        print("✓ Model loaded successfully with enhanced multimodal architecture.")
    except Exception as e:
        print(f"✗ Error loading model state dict: {e}")
        print("Please ensure the checkpoint was trained with an identical architecture.")
        return None
    
    model.eval()

    try:
        pose_fts, lbs, bbox_fts, visual_fts = load_multimodal_dataset_npy(path_test_dir)
        if lbs.ndim > 1 and lbs.shape[1] > 1:
            labels = np.argmax(lbs, axis=1)
        else:
            labels = lbs.astype(int)
    except FileNotFoundError as e:
        print(e)
        return None

    # Data preprocessing
    pose_fts = pose_fts[:, ::2, :, :]  # Frame sampling 30->15
    pose_fts[:, :, :, :2] = processing_data(pose_fts[:, :, :, :2])
    
    visual_fts = visual_fts[:, ::2, :] # (N, 15, 512)
    bbox_fts = bbox_fts[:, ::2, :]     # (N, 15, 4)
    
    print(" --------- Enhanced Test Set Class Distribution ---------")
    for i in range(num_classes):
        count = np.sum(labels == i)
        print(f"class {i} ({class_names[i]}): {count}")

    # Create dataset with correct tensor order
    test_dataset = TensorDataset(
        torch.tensor(pose_fts, dtype=torch.float32).permute(0, 3, 1, 2),  # Pose: (N, C, T, V)
        torch.tensor(visual_fts, dtype=torch.float32),                     # Visual: (N, T, 512)
        torch.tensor(bbox_fts, dtype=torch.float32),                      # Bbox: (N, T, 4)
        torch.tensor(labels, dtype=torch.long)                             # Labels: (N,)
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    truth = []
    pred = []
    
    with torch.no_grad():
        pbar_test = tqdm(test_loader, desc='Enhanced Testing', unit='batch')
        # Unpack in correct order: pose, visual, bbox, labels
        for batch_pose, batch_visual, batch_bbox, batch_labels in pbar_test:
            if batch_pose.shape[2] < 2:
                continue
            
            # Calculate motion from pose
            mot = batch_pose[:, :2, 1:, :] - batch_pose[:, :2, :-1, :]
            
            # Ensure temporal dimensions match
            min_T = mot.shape[2]
            batch_pose_resized = batch_pose[:, :, :min_T, :]
            batch_visual_resized = batch_visual[:, :min_T, :]
            batch_bbox_resized = batch_bbox[:, :min_T, :]

            # Move to device
            mot = mot.to(device)
            batch_pose_resized = batch_pose_resized.to(device)
            batch_visual_resized = batch_visual_resized.to(device)
            batch_bbox_resized = batch_bbox_resized.to(device)
            
            # Forward pass with 4 inputs in correct order
            outputs = model((batch_pose_resized, mot, batch_visual_resized, batch_bbox_resized))
            _, preds = torch.max(outputs, 1)
            
            truth.extend(batch_labels.tolist())
            pred.extend(preds.cpu().tolist())

    # Calculate metrics
    CM = metrics.confusion_matrix(truth, pred).T
    precision = metrics.precision_score(truth, pred, average=None, zero_division=0)
    recall = metrics.recall_score(truth, pred, average=None, zero_division=0)
    accuracy = metrics.accuracy_score(truth, pred, normalize=True)
    f1_score = metrics.f1_score(truth, pred, average=None, zero_division=0)
    
    # Print results
    print(f"\n{'='*60}")
    print("ENHANCED MULTIMODAL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nPer-class Metrics:")
    for i in range(num_classes):
        print(f'****{class_names[i]}****')
        print(f'Precision: {precision[i]:.4f}')
        print(f'Recall: {recall[i]:.4f}')
        print(f'F1-score: {f1_score[i]:.4f}')
    
    # Save results
    os.makedirs('info_enhanced_multimodal', exist_ok=True)
    
    with open('info_enhanced_multimodal/info_stgcn.txt', 'w') as file:
        file.write(f'{precision} {recall} {f1_score}')
    
    plot_cm(CM, normalize=False, save_dir='info_enhanced_multimodal',
            names_x=class_names, names_y=class_names, show=False)
    
    print('Enhanced confusion matrix saved to info_enhanced_multimodal/confusion_matrix.png')
    return CM, precision, recall, f1_score, accuracy

if __name__ == '__main__':
    path_model = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/runs/exp17/best.pt'
    path_test_dir = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/test_4classes_visual'
    
    test_enhanced_multimodal_confusion_matrix(path_test_dir, path_model, batch_size=64)