import pickle  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from concurrent.futures import ProcessPoolExecutor  
import multiprocessing as mp  
  
# Updated class names cho 4 class (gộp Vandalism)  
class_names = ['Vandalism', 'LookingAround', 'UnauthorizedFilming', 'ClimbOverFence']  
  
# Updated main_parts để tương thích với bbox data  
main_parts = ['left_shoulder_x','left_shoulder_y','left_shoulder_s',  
              'right_shoulder_x','right_shoulder_y','right_shoulder_s',  
              'left_hip_x','left_hip_y','left_hip_s',  
              'right_hip_x','right_hip_y','right_hip_s',  
              'left_knee_x','left_knee_y','left_knee_s',  
              'right_knee_x','right_knee_y','right_knee_s']  
main_idx_parts = [5,6,11,12,13,14]  
  
# Updated paths cho 4 classes  
csv_pose_file = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/pose_and_score_4classes.csv'  
save_path_train = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/train_4classes.pkl'  
save_path_test = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/test_4classes.pkl'  
  
# Params  
smooth_labels_step = 8  
n_frames = 30  
skip_frame = 1  
  
def scale_pose(xy):  
    """Normalize pose points by scale with max/min value of each pose."""  
    if xy.ndim == 2:  
        xy = np.expand_dims(xy, 0)  
    xy_min = np.nanmin(xy, axis=1)  
    xy_max = np.nanmax(xy, axis=1)  
    for i in range(xy.shape[0]):  
        xy_range = xy_max[i] - xy_min[i]  
        xy_range = np.where((xy_range == 0) | np.isnan(xy_range), 1, xy_range)  
        xy[i] = np.nan_to_num(((xy[i] - xy_min[i]) / xy_range) * 2 - 1, nan=0.0)  
    return xy.squeeze()  
  
def normalize_bbox_sequence(bbox_data):  
    """Normalize bbox coordinates to [0,1] range"""  
    bbox_normalized = bbox_data.copy().astype(float)  
    bbox_normalized[:, [0, 2]] /= 640.0  
    bbox_normalized[:, [1, 3]] /= 640.0  
    return np.nan_to_num(bbox_normalized, nan=0.0)  
  
def seq_label_smoothing(labels, max_step=10):  
    """Label smoothing function"""  
    if labels.size == 0:  
        return labels  
        
    steps = remain_step = target_label = active_label = start_change = 0  
    max_val = np.max(labels) if labels.size > 0 else 1.0  
    min_val = np.min(labels) if labels.size > 0 else 0.0  
        
    for i in range(labels.shape[0]):  
        if remain_step > 0:  
            if i >= start_change:  
                if steps > 0:  
                    labels[i][active_label] = max_val * remain_step / steps  
                    calculated_value = max_val * (steps - remain_step) / steps  
                    labels[i][target_label] = calculated_value if calculated_value else min_val  
                remain_step -= 1  
            continue  
            
        diff_index = np.where(np.argmax(labels[i:i+max_step], axis=1) - np.argmax(labels[i]) != 0)[0]  
        if len(diff_index) > 0:  
            steps = diff_index[0]  
            remain_step = steps  
            start_change = i + remain_step // 2  
            target_label = np.argmax(labels[i + remain_step])  
            active_label = np.argmax(labels[i])  
    return labels  
  
def process_video_chunk(args):  
    """Process a chunk of videos in parallel - FIXED for multiprocessing"""  
    video_data_chunk, class_columns = args  # Unpack arguments  
    local_feature_set = []  
    local_bbox_set = []  
    local_labels_set = []  
        
    for vid, data in video_data_chunk:  
        if len(data) < n_frames:  
            continue  
            
        # Label Smoothing - FIXED: Use pre-determined class columns  
        esp = 0.1  
        cols = class_columns  # Use passed class columns instead of detecting  
            
        if len(cols) > 1:  
            data[cols] = data[cols] * (1 - esp) + (1 - data[cols]) * esp / (len(cols) - 1)  
        data[cols] = seq_label_smoothing(data[cols].values, smooth_labels_step)  
            
        # Separate continuous frames  
        frames = data['frame'].values  
        frames_set = []  
        fs = [0]  
        for i in range(1, len(frames)):  
            if frames[i] < frames[i-1] + 10:  
                fs.append(i)  
            else:  
                if len(fs) >= n_frames:  
                    frames_set.append(fs)  
                fs = [i]  
        if len(fs) >= n_frames:  
            frames_set.append(fs)  
            
        for fs in frames_set:  
            if len(fs) < n_frames:  
                continue  
                
            # Extract pose data (17 keypoints × 3) - skip bbox columns  
            pose_data = data.iloc[fs, 1:-len(cols)-4].values.reshape(-1, 17, 3)  
                
            # Extract bbox data (4 values: xmin, ymin, xmax, ymax)  
            bbox_data = data.iloc[fs, -len(cols)-4:-len(cols)].values  
                
            # Scale pose normalize  
            pose_data[:, :, :2] = scale_pose(pose_data[:, :, :2])  
                
            # Normalize bbox  
            bbox_normalized = normalize_bbox_sequence(bbox_data)  
                
            # Weighting main parts score  
            scr = pose_data[:, :, -1].copy()  
            scr[:, main_idx_parts] = np.minimum(scr[:, main_idx_parts] * 1.5, 1.0)  
            scr = scr.mean(1)  
                
            # Targets  
            lb = data.iloc[fs, -len(cols):].values  
            lb = lb * scr[:, None]  
                
            # Create 30-frame sequences  
            for i in range(pose_data.shape[0] - n_frames + 1):  
                local_feature_set.append(pose_data[i:i+n_frames])  
                local_bbox_set.append(bbox_normalized[i:i+n_frames])  
                local_labels_set.append(lb[i:i+n_frames].mean(0))  
        
    return local_feature_set, local_bbox_set, local_labels_set  
  
# Load CSV từ create_dataset_2.py  
try:  
    print("Loading pose data...")  
    annot = pd.read_csv(csv_pose_file)  
    print(f"Loaded pose data with {len(annot)} samples")  
    print("Class distribution:", annot['label'].value_counts().sort_index())  
except FileNotFoundError:  
    print(f"Error: Pose file not found at {csv_pose_file}")  
    print("Please run create_dataset_2.py first!")  
    exit()  
  
# Remove NaN - tương thích với bbox columns  
idx = annot[main_parts].isna().sum(1) > 0  
print(f"Removing {idx.sum()} samples with NaN in main parts")  
annot = annot[~idx].reset_index(drop=True)  
  
# One-Hot Labels cho 4 classes  
label_onehot = pd.get_dummies(annot['label'])  
cols = label_onehot.columns.values  
annot = annot.drop('label', axis=1).join(label_onehot)  
  
# FIXED: Ensure column names are consistent for multiprocessing  
annot.columns = [str(col) for col in annot.columns]  
cols = [str(col) for col in cols]  
print(f"One-hot classes: {cols}")  
  
# Process videos with multiprocessing  
vid_list = annot['video'].unique()  
print(f"Processing {len(vid_list)} videos with parallel processing...")  
  
# Prepare video data chunks for parallel processing  
num_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes  
chunk_size = max(1, len(vid_list) // num_processes)  
  
video_chunks = []  
for i in range(0, len(vid_list), chunk_size):  
    chunk_vids = vid_list[i:i+chunk_size]  
    chunk_data = []  
    for vid in chunk_vids:  
        data = annot[annot['video'] == vid].drop(columns='video').reset_index(drop=True)  
        chunk_data.append((vid, data))  
    # FIXED: Pass class columns as argument to avoid serialization issues  
    video_chunks.append((chunk_data, cols))  
  
print(f"Created {len(video_chunks)} chunks for parallel processing")  
  
# Process chunks in parallel  
all_features = []  
all_bboxes = []  
all_labels = []  
  
with ProcessPoolExecutor(max_workers=num_processes) as executor:  
    results = list(executor.map(process_video_chunk, video_chunks))  
  
# Combine results  
for features, bboxes, labels in results:  
    all_features.extend(features)  
    all_bboxes.extend(bboxes)  
    all_labels.extend(labels)  
  
# Convert to numpy arrays  
print("Converting to final arrays...")  
feature_set = np.array(all_features) if all_features else np.empty((0, n_frames, 17, 3))  
bbox_set = np.array(all_bboxes) if all_bboxes else np.empty((0, n_frames, 4))  
labels_set = np.array(all_labels) if all_labels else np.empty((0, len(cols)))  
  
print(f"\nDataset creation completed!")  
print(f"Feature set shape: {feature_set.shape}")  
print(f"Bbox set shape: {bbox_set.shape}")  
print(f"Labels set shape: {labels_set.shape}")  
  
# Train/test split  
X_train, X_test, y_train, y_test = train_test_split(feature_set, labels_set, test_size=0.2, random_state=0)  
bbox_train, bbox_test = train_test_split(bbox_set, test_size=0.2, random_state=0)  
  
# Save training data với cả pose và bbox  
print("Saving training data...")  
with open(save_path_train, 'wb') as f:  
    pickle.dump((X_train, y_train, bbox_train), f)  
  
# Save testing data với cả pose và bbox  
print("Saving testing data...")  
with open(save_path_test, 'wb') as f:  
    pickle.dump((X_test, y_test, bbox_test), f)  
  
print(f"Dataset saved successfully!")  
print(f"Training data: {save_path_train}")  
print(f"Testing data: {save_path_test}")  
print(f"Training samples: {len(X_train)}")  
print(f"Testing samples: {len(X_test)}")  
  
# Show final class distribution  
print("\nFinal class distribution:")  
train_labels = np.argmax(y_train, axis=1)  
for class_id in range(len(class_names)):  
    count = np.sum(train_labels == class_id)  
    print(f"  - Class {class_id} ({class_names[class_id]}): {count} samples")