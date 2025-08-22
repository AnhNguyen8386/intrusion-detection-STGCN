import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from tqdm import tqdm  
import os  
import sys  

class_names = ['Vandalism', 'UnauthorizedFilming', 'ClimbOverFence', 'Normal']  
CSV_OUTPUT_FILE = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/pose_visual_bbox_4classes.csv'  
SAVE_PATH_TRAIN_DIR = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/train'  
SAVE_PATH_TEST_DIR = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/test'  
SMOOTH_LABELS_STEP = 0.1  
SEQUENCE_LENGTH = 30  
TEST_SIZE = 0.1  
RANDOM_STATE = 42  
CLASS_NAMES = class_names  
  
def smooth_labels(labels_one_hot, alpha):  
    if labels_one_hot.ndim < 2:  
        return labels_one_hot  
    num_classes = labels_one_hot.shape[1]  
    uniform_distribution = alpha / num_classes  
    smoothed = (1 - alpha) * labels_one_hot + uniform_distribution  
    return smoothed  
  
def normalize_features(features, feat_min, feat_max):  
    feat_range = feat_max - feat_min  
    feat_range = np.where(feat_range == 0, 1e-8, feat_range)  
    normalized_features = (features - feat_min) / feat_range  
    normalized_features = np.nan_to_num(normalized_features, nan=0.0)  
    return normalized_features  
  
def scale_pose(xy_data, xy_min, xy_max):  
    xy_data = np.nan_to_num(xy_data, nan=0.0)  
    if xy_data.ndim == 2:  
        xy_data = np.expand_dims(xy_data, 0)  
    xy_range = xy_max - xy_min  
    xy_range = np.where(xy_range == 0, 1.0, xy_range)  
    xy_normalized = ((xy_data - xy_min) / xy_range) * 2 - 1  
    xy_normalized = np.nan_to_num(xy_normalized, nan=0.0)  
    return xy_normalized.squeeze()  
  
def process_and_save_data(video_list, data_annot, save_dir,   
                         pose_min, pose_max, visual_min, visual_max):  
    """Enhanced processing for multimodal data"""  
      
    feature_set_list, labels_set_list, bbox_set_list, visual_set_list, scene_set_list = [], [], [], [], []  
    os.makedirs(save_dir, exist_ok=True)  
      
    num_keypoints = 17  
    num_kpt_features = num_keypoints * 3  
    num_bbox_features = 4  
    num_visual_features = 512  # FIX: ResNet18 outputs 512 features   
    pose_cols_start = 3  
    pose_cols_end = pose_cols_start + num_kpt_features  
    bbox_cols_start = pose_cols_end  
    bbox_cols_end = bbox_cols_start + num_bbox_features  
    visual_cols_start = bbox_cols_end  
    visual_cols_end = visual_cols_start + num_visual_features  
  
    for vid in tqdm(video_list, desc=f"Processing videos for {os.path.basename(save_dir)}"):  
        video_data = data_annot[data_annot['video'] == vid].reset_index(drop=True)  
        if len(video_data) < SEQUENCE_LENGTH:  
            print(f"   Bỏ qua {vid}: không đủ khung hình ({len(video_data)} < {SEQUENCE_LENGTH})")  
            continue  
        labels = video_data['label'].values  
          
        try:  
            pose_data = video_data.iloc[:, pose_cols_start:pose_cols_end].values.reshape(-1, num_keypoints, 3)  
            if pose_data.size == 0:   
                pose_data = np.full((len(video_data), num_keypoints, 3), np.nan)  
        except (IndexError, ValueError):   
            pose_data = np.full((len(video_data), num_keypoints, 3), np.nan)  
              
        try:  
            bbox_data = video_data.iloc[:, bbox_cols_start:bbox_cols_end].values  
            if bbox_data.size == 0:   
                bbox_data = np.full((len(video_data), num_bbox_features), np.nan)  
        except (IndexError, ValueError):   
            bbox_data = np.full((len(video_data), num_bbox_features), np.nan)  
              
        try:  
            visual_data = video_data.iloc[:, visual_cols_start:visual_cols_end].values  
            if visual_data.size == 0:   
                visual_data = np.full((len(video_data), num_visual_features), np.nan)  
        except (IndexError, ValueError):   
            visual_data = np.full((len(video_data), num_visual_features), np.nan)  
  
        # FIX: Create dummy scene data (same as visual for compatibility)  
        scene_data = visual_data.copy()   
        pose_data_interpolated = pd.DataFrame(pose_data.reshape(len(pose_data), -1)).interpolate(  
            method='linear', limit_direction='both').ffill().bfill().values.reshape(-1, num_keypoints, 3)  
        bbox_data_interpolated = pd.DataFrame(bbox_data).interpolate(  
            method='linear', limit_direction='both').ffill().bfill().values  
        visual_data_interpolated = pd.DataFrame(visual_data).interpolate(  
            method='linear', limit_direction='both').ffill().bfill().values  
        scene_data_interpolated = visual_data_interpolated.copy()   
        pose_data_interpolated[:, :, :2] = scale_pose(pose_data_interpolated[:, :, :2], pose_min, pose_max)  
        visual_normalized = normalize_features(visual_data_interpolated, visual_min, visual_max)  
        scene_normalized = visual_normalized.copy()   
        labels_one_hot = np.zeros((len(labels), len(CLASS_NAMES)))  
        for i, label_val in enumerate(labels):  
            labels_one_hot[i, int(label_val)] = 1  
          
        labels_smoothed = smooth_labels(labels_one_hot, SMOOTH_LABELS_STEP)   
        for i in range(len(video_data) - SEQUENCE_LENGTH + 1):  
            pose_seq = pose_data_interpolated[i:i + SEQUENCE_LENGTH]  
            bbox_seq = bbox_data_interpolated[i:i + SEQUENCE_LENGTH]  
            visual_seq = visual_normalized[i:i + SEQUENCE_LENGTH]  
            scene_seq = scene_normalized[i:i + SEQUENCE_LENGTH]  
            label_seq = labels_smoothed[i:i + SEQUENCE_LENGTH]  
              
            feature_set_list.append(pose_seq)  
            bbox_set_list.append(bbox_seq)  
            visual_set_list.append(visual_seq)  
            scene_set_list.append(scene_seq)  
            labels_set_list.append(np.argmax(label_seq[len(label_seq) // 2]))  
  
    # Save data  
    if feature_set_list:  
        np.save(os.path.join(save_dir, 'X.npy'), np.array(feature_set_list))  
        np.save(os.path.join(save_dir, 'y.npy'), np.array(labels_set_list))  
        np.save(os.path.join(save_dir, 'bbox.npy'), np.array(bbox_set_list))  
        np.save(os.path.join(save_dir, 'visual.npy'), np.array(visual_set_list))  
        np.save(os.path.join(save_dir, 'scene.npy'), np.array(scene_set_list))  

        flow_data = []  
        for pose_seq in feature_set_list:  
            motion = np.diff(pose_seq[:, :, :2], axis=0)  
            motion = np.concatenate([motion, motion[-1:]], axis=0)    
            flow_data.append(motion)  
        np.save(os.path.join(save_dir, 'flow.npy'), np.array(flow_data))  
          
        print(f"Đã lưu {len(feature_set_list)} mẫu vào {save_dir}")  
        print(f"  - Pose sequences: {np.array(feature_set_list).shape}")  
        print(f"  - Visual features: {np.array(visual_set_list).shape}")  
        print(f"  - Bbox data: {np.array(bbox_set_list).shape}")  
    else:  
        print(f"Không có mẫu nào được tạo cho {save_dir}")  
  
def main():  
    try:  
        annot = pd.read_csv(CSV_OUTPUT_FILE)  
        print(f"Đã tải enhanced multimodal dataset với {len(annot)} mẫu và {len(annot.columns)} cột")  
        print("Phân bố lớp ban đầu:", annot['label'].value_counts().sort_index())  
    except FileNotFoundError:  
        print(f"Lỗi: Không tìm thấy tệp CSV tại {CSV_OUTPUT_FILE}")  
        print("Vui lòng chạy enhanced pipeline (create_dataset_1.py -> create_dataset_2.py) trước!")  
        sys.exit(1)  
  
    vid_list = annot['video'].unique()  
      
    video_labels_map = {  
        vid: annot[annot['video'] == vid]['label'].mode()[0]  
        for vid in vid_list  
    }  
      
    labels_vid = [video_labels_map[vid] for vid in vid_list]  
      
    print("Đang thực hiện phân chia train/test theo video...")  
    train_vids, test_vids = train_test_split(  
        vid_list, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels_vid  
    )  
    print(f"Số video trong tập train: {len(train_vids)}")  
    print(f"Số video trong tập test: {len(test_vids)}")  
      
    print("Đang tính toán các tham số chuẩn hóa từ dữ liệu huấn luyện...")  
    all_train_data = annot[annot['video'].isin(train_vids)]  
  
    num_keypoints = 17  
    num_kpt_features = num_keypoints * 3  
    num_visual_features = 512  
      
    pose_cols_start = 3  
    pose_cols_end = pose_cols_start + num_kpt_features  
    bbox_cols_start = pose_cols_end  
    visual_cols_start = bbox_cols_start + 4  
    visual_cols_end = visual_cols_start + num_visual_features  
      
    pose_data_train = all_train_data.iloc[:, pose_cols_start:pose_cols_end].values.reshape(-1, num_keypoints, 3)  
    visual_data_train = all_train_data.iloc[:, visual_cols_start:visual_cols_end].values  
  
    pose_min = np.nanmin(pose_data_train[:, :, :2], axis=(0, 1), keepdims=True) if pose_data_train.size > 0 else np.zeros((1, 2))  
    pose_max = np.nanmax(pose_data_train[:, :, :2], axis=(0, 1), keepdims=True) if pose_data_train.size > 0 else np.ones((1, 2))  
    visual_min = np.nanmin(visual_data_train, axis=0, keepdims=True) if visual_data_train.size > 0 else np.zeros((1, num_visual_features))  
    visual_max = np.nanmax(visual_data_train, axis=0, keepdims=True) if visual_data_train.size > 0 else np.ones((1, num_visual_features))  
  
    print("Đang xử lý và lưu dữ liệu train")  
    process_and_save_data(train_vids, annot, SAVE_PATH_TRAIN_DIR, pose_min, pose_max, visual_min, visual_max)  
    print("Đang xử lý và lưu dữ liệu test")  
    process_and_save_data(test_vids, annot, SAVE_PATH_TEST_DIR, pose_min, pose_max, visual_min, visual_max)  
      
    print("\nEnhanced pipeline Stage 3 completed!")  
    print("Created multimodal data files compatible with EnhancedMultiModalDataset")  
  
if __name__ == '__main__':  
    main()