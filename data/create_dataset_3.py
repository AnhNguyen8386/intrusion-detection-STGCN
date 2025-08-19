import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

class_names = ['Vandalism', 'UnauthorizedFilming', 'ClimbOverFence', 'Normal']

main_parts = ['left_shoulder_x','left_shoulder_y','left_shoulder_s',
              'right_shoulder_x','right_shoulder_y','right_shoulder_s',
              'left_hip_x','left_hip_y','left_hip_s',
              'right_hip_x','right_hip_y','right_hip_s',
              'left_knee_x','left_knee_y','left_knee_s',
              'right_knee_x','right_knee_y','right_knee_s']
main_idx_parts = [5, 6, 11, 12, 13, 14] 

csv_pose_file = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/pose_bbox_visual_4classes.csv'
save_path_train_dir = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/train_4classes_visual'
save_path_test_dir = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/test_4classes_visual'

smooth_labels_step = 8
n_frames = 30

try:
    print("Loading pose, bbox and visual features data...")
    annot = pd.read_csv(csv_pose_file)
    print(f"Loaded dataset with {len(annot)} samples and {len(annot.columns)} columns")
    print("Class distribution:", annot['label'].value_counts().sort_index())
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_pose_file}")
    print("Please run create_dataset_2.py with visual features first!")
    exit()

# Remove NaN values
idx = annot.iloc[:, 2:53][main_parts].isna().sum(1) > 0
print(f"Removing {idx.sum()} samples with NaN in main parts")
annot = annot[~idx].reset_index(drop=True)

# One-Hot Labels
label_onehot = pd.get_dummies(annot['label'])
annot = annot.drop('label', axis=1).join(label_onehot)
cols = label_onehot.columns.values
print(f"One-hot classes: {cols}")

def scale_pose(xy):
    """Normalize pose points by scale with max/min value of each pose."""
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1, keepdims=True)
    xy_max = np.nanmax(xy, axis=1, keepdims=True)
    xy_range = xy_max - xy_min
    xy_range = np.where(xy_range == 0, 1.0, xy_range)
    xy_normalized = ((xy - xy_min) / xy_range) * 2 - 1
    return np.nan_to_num(xy_normalized, nan=0.0).squeeze()

def normalize_visual_features(visual_data):
    """Normalize visual features to [0,1] range"""
    visual_normalized = visual_data.copy().astype(float)
    feat_min = np.min(visual_data, axis=0)
    feat_max = np.max(visual_data, axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    visual_normalized = (visual_data - feat_min) / feat_range
    return visual_normalized

def seq_label_smoothing(labels, max_step=10):
    """Label smoothing function"""
    if labels.size == 0:
        return labels
    
    smooth_labels = labels.copy()
    num_frames = smooth_labels.shape[0]
    num_classes = smooth_labels.shape[1]
    
    max_label_indices = np.argmax(smooth_labels, axis=1)
    
    i = 0
    while i < num_frames:
        diff_indices = np.where(max_label_indices[i:i+max_step] - max_label_indices[i] != 0)[0]
        
        if len(diff_indices) > 0:
            steps = diff_indices[0]
            if steps > 0:
                start_change = i + steps // 2
                end_change = i + steps
                
                for j in range(start_change, end_change):
                    progress = (j - start_change) / (end_change - start_change)
                    current_label_idx = max_label_indices[i]
                    target_label_idx = max_label_indices[end_change]
                    
                    smooth_labels[j, current_label_idx] = (1 - progress) * smooth_labels[j, current_label_idx]
                    smooth_labels[j, target_label_idx] = smooth_labels[j, target_label_idx] + (labels[j, target_label_idx] * progress)
            i += steps
        else:
            i += max_step
    
    return smooth_labels

vid_list = annot['video'].unique()
labels_vid = [annot[annot['video'] == vid][cols].values[0].argmax() for vid in vid_list]

print(f"Total videos to split: {len(vid_list)}")
print("Performing stratified train/test split at video level...")

train_vids, test_vids, _, _ = train_test_split(
    vid_list, labels_vid,
    test_size=0.2,
    random_state=0,
    stratify=labels_vid
)

print(f"Number of videos in train set: {len(train_vids)}")
print(f"Number of videos in test set: {len(test_vids)}")

def process_and_save_data(video_list, data_annot, save_dir):
    feature_set_list = []
    labels_set_list = []
    bbox_set_list = []
    visual_set_list = []

    os.makedirs(save_dir, exist_ok=True)
    
    for vid in tqdm(video_list, desc=f"Processing videos for {os.path.basename(save_dir)}"):
        data = data_annot[data_annot['video'] == vid].reset_index(drop=True).drop(columns='video')

        if len(data) < n_frames:
            print(f"  Skipping {vid}: insufficient frames ({len(data)} < {n_frames})")
            continue

        esp = 0.1
        if len(cols) > 1:
            data[cols] = data[cols] * (1 - esp) + (1 - data[cols]) * esp / (len(cols) - 1)
        data[cols] = seq_label_smoothing(data[cols].values, smooth_labels_step)
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

            pose_data = data.iloc[fs, 1:52].values.reshape(-1, 17, 3)
            bbox_data = data.iloc[fs, 52:56].values
            visual_data = data.iloc[fs, 56:568].values
            pose_data[:, :, :2] = scale_pose(pose_data[:, :, :2])
            visual_normalized = normalize_visual_features(visual_data)
            scr = pose_data[:, :, -1].copy()
            scr[:, main_idx_parts] = np.minimum(scr[:, main_idx_parts] * 1.5, 1.0)
            scr = scr.mean(1)
            lb = data.iloc[fs, -len(cols):].values
            lb = lb * scr[:, None]
            num_sequences = pose_data.shape[0] - n_frames + 1
            
            for i in range(num_sequences):
                feature_set_list.append(pose_data[i:i+n_frames])
                labels_set_list.append(lb[i:i+n_frames].mean(0))
                bbox_set_list.append(bbox_data[i:i+n_frames])
                visual_set_list.append(visual_normalized[i:i+n_frames])

    if len(feature_set_list) > 0:
        np.save(os.path.join(save_dir, 'X.npy'), np.array(feature_set_list))
        np.save(os.path.join(save_dir, 'y.npy'), np.array(labels_set_list))
        np.save(os.path.join(save_dir, 'bbox.npy'), np.array(bbox_set_list))
        np.save(os.path.join(save_dir, 'visual.npy'), np.array(visual_set_list))
        print(f"Saved {len(feature_set_list)} samples to {save_dir}")
    else:
        print(f"No samples generated for {save_dir}")

process_and_save_data(train_vids, annot, save_path_train_dir)
process_and_save_data(test_vids, annot, save_path_test_dir)

print("\nMultimodal dataset creation completed!")