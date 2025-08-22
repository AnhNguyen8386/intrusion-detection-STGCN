import os  
import numpy as np  
import torch  
import pickle  
import pandas as pd  
from torch.utils.data import Dataset, DataLoader  
  
def processing_data(features):  
    """  
    Normalize pose points by scale with max/min value of each pose.  
    xy : (frames, parts, xy) or (parts, xy)  
    """  
    def scale_pose(xy):  
        if xy.ndim == 2:  
            xy = np.expand_dims(xy, 0)  
          
        # Calculate min/max across parts for each frame  
        xy_min = np.nanmin(xy, axis=1, keepdims=True)  
        xy_max = np.nanmax(xy, axis=1, keepdims=True)  
          
        # Avoid division by zero for constant values  
        denominator = (xy_max - xy_min)  
        denominator[denominator == 0] = 1e-8  
          
        xy = (xy - xy_min) / denominator * 2 - 1  
        return xy.squeeze()  
      
    features = scale_pose(features)  
    return features  
  
class EnhancedMultiModalDataset(Dataset):  
    def __init__(self, data_dir, mode='train'):  
        """  
        Enhanced dataset compatible with multimodal pipeline  
        mode: 'train' or 'test'  
        """  
        self.data_dir = data_dir  
        self.mode = mode  
          
        # Try to load from enhanced pipeline first (CSV format)  
        csv_file = os.path.join(data_dir, 'pose_visual_bbox_4classes.csv')  
        if os.path.exists(csv_file):  
            self._load_from_csv(csv_file)  
        else:  
            # Fallback to pickle format from traditional pipeline  
            pkl_file = os.path.join(data_dir, f'{mode}_4classes.pkl')  
            if os.path.exists(pkl_file):  
                self._load_from_pickle(pkl_file)  
            else:  
                raise FileNotFoundError(f"Neither CSV nor PKL files found in {data_dir}")  
      
    def _load_from_csv(self, csv_file):  
        """Load data from enhanced pipeline CSV format"""  
        print(f"Loading enhanced multimodal data from {csv_file}")  
        df = pd.read_csv(csv_file)  
          
        # Extract pose features (17 keypoints * 3 = 51 features)  
        pose_cols = [f'pose_{i}_{axis}' for i in range(17) for axis in ['x', 'y', 'score']]  
        self.pose_data = df[pose_cols].values.reshape(-1, 17, 3)  
          
        # Extract bbox features (4 coordinates)  
        bbox_cols = [f'bbox_{i}' for i in range(4)]  
        self.bbox_data = df[bbox_cols].values  
          
        # Extract visual features (512 from ResNet18)  
        visual_cols = [f'visual_{i}' for i in range(512)]  
        self.visual_data = df[visual_cols].values  
          
        # Labels  
        self.labels = df['label'].values  
          
        # Create dummy scene and flow data for compatibility  
        self.scene_data = np.zeros_like(self.visual_data)  # Same shape as visual  
        self.flow_data = self.pose_data[:, :, :2]  # Use pose x,y as motion proxy  
          
        print(f"Loaded {len(self.labels)} samples with enhanced features")  
      
    def _load_from_pickle(self, pkl_file):  
        """Load data from traditional pickle format"""  
        print(f"Loading traditional data from {pkl_file}")  
        with open(pkl_file, 'rb') as f:  
            data = pickle.load(f)  
          
        if len(data) == 3:  # (X, y, bbox) format  
            self.pose_data, self.labels, self.bbox_data = data  
            # Create dummy visual and scene data  
            batch_size = len(self.labels)  
            self.visual_data = np.zeros((batch_size, 512))  # ResNet18 features  
            self.scene_data = np.zeros((batch_size, 512))  
            self.flow_data = self.pose_data[:, :, :, :2].mean(axis=1)  # Average motion  
        else:  
            raise ValueError(f"Unexpected pickle format in {pkl_file}")  
      
    def __len__(self):  
        return len(self.labels)  
      
    def __getitem__(self, idx):  
        # Get pose data and apply normalization  
        pose = self.pose_data[idx]  
        if pose.ndim == 2:  # (17, 3) format  
            pose_xy = pose[:, :2]  # Take only x,y coordinates  
            pose_xy = processing_data(pose_xy)  # Apply ST-GCN normalization  
            # Add confidence scores back  
            confidence = self.pose_data[idx][:, 2:3]  
            pose = np.concatenate([pose_xy, confidence], axis=1)  
          
        # Calculate motion features (difference between consecutive frames)  
        if hasattr(self, 'sequence_length') and self.pose_data.ndim == 4:  
            # For sequence data, calculate motion  
            motion = np.diff(self.pose_data[idx][:, :, :2], axis=0)  
            motion = np.concatenate([motion, motion[-1:]], axis=0)  # Pad last frame  
        else:  
            # For single frame, use flow_data  
            motion = self.flow_data[idx]  
          
        # Convert to tensors  
        pose_tensor = torch.from_numpy(pose).float()  
        motion_tensor = torch.from_numpy(motion).float()  
        bbox_tensor = torch.from_numpy(self.bbox_data[idx]).float()  
        visual_tensor = torch.from_numpy(self.visual_data[idx]).float()  
        scene_tensor = torch.from_numpy(self.scene_data[idx]).float()  
        label_tensor = torch.from_numpy(np.array(self.labels[idx])).long()  
          
        return pose_tensor, motion_tensor, visual_tensor, bbox_tensor, scene_tensor, label_tensor  
  
def get_enhanced_dataloaders(data_dir, batch_size=32, train_test_split=True):  
    """  
    Enhanced dataloader compatible with both CSV and PKL formats  
    """  
    try:  
        if train_test_split:  
            # Try to load separate train/test files  
            train_dataset = EnhancedMultiModalDataset(data_dir, mode='train')  
            test_dataset = EnhancedMultiModalDataset(data_dir, mode='test')  
        else:  
            # Load single dataset and split  
            dataset = EnhancedMultiModalDataset(data_dir, mode='train')  
            train_size = int(0.8 * len(dataset))  
            test_size = len(dataset) - train_size  
            train_dataset, test_dataset = torch.utils.data.random_split(  
                dataset, [train_size, test_size]  
            )  
          
        train_loader = DataLoader(  
            train_dataset,   
            batch_size=batch_size,   
            shuffle=True,  
            num_workers=2  
        )  
        test_loader = DataLoader(  
            test_dataset,   
            batch_size=batch_size,   
            shuffle=False,  
            num_workers=2  
        )  
          
        return train_loader, test_loader  
          
    except FileNotFoundError as e:  
        print(f"Error: {e}")  
        print("Please run the enhanced pipeline (create_dataset_1.py -> create_dataset_2.py) first!")  
        return None, None  
  
# Backward compatibility function  
def get_dataloaders(train_dir, test_dir, batch_size):  
    """Backward compatibility wrapper"""  
    return get_enhanced_dataloaders(train_dir, batch_size)  
  
# Legacy MultiModalDataset for backward compatibility  
class MultiModalDataset(Dataset):  
    def __init__(self, data_dir):  
        # Redirect to enhanced dataset  
        self.enhanced_dataset = EnhancedMultiModalDataset(data_dir, mode='train')  
      
    def __len__(self):  
        return len(self.enhanced_dataset)  
      
    def __getitem__(self, idx):  
        return self.enhanced_dataset[idx]