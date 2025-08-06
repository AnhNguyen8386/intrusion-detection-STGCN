import os  
import cv2  
import time  
import pandas as pd  
import numpy as np  
from ultralytics import YOLO  
import glob  
  
# Load YOLOv8 pose model  
model = YOLO('/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/result_model/yolov7-pose/yolov8x-pose.pt')  
  
# Updated paths cho 4 classes  
save_path = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/pose_and_score_4classes.csv'  
annot_file = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/annotation_4classes.csv'  
video_folder = '/mnt/myhdd/anhnv/intrustion_ditection/data/Train/extracted_images/intrusion'  
  
# Updated columns để include bbox data  
columns = ['video', 'frame',     
           'nose_x','nose_y','nose_s','left_eye_x','left_eye_y','left_eye_s',    
           'right_eye_x','right_eye_y','right_eye_s','left_ear_x','left_ear_y','left_ear_s',    
           'right_ear_x','right_ear_y','right_ear_s','left_shoulder_x','left_shoulder_y','left_shoulder_s',    
           'right_shoulder_x','right_shoulder_y','right_shoulder_s','left_elbow_x','left_elbow_y','left_elbow_s',    
           'right_elbow_x','right_elbow_y','right_elbow_s','left_wrist_x','left_wrist_y','left_wrist_s',    
           'right_wrist_x','right_wrist_y','right_wrist_s','left_hip_x','left_hip_y','left_hip_s',    
           'right_hip_x','right_hip_y','right_hip_s','left_knee_x','left_knee_y','left_knee_s',    
           'right_knee_x','right_knee_y','right_knee_s','left_ankle_x','left_ankle_y','left_ankle_s',    
           'right_ankle_x','right_ankle_y','right_ankle_s',    
           'bbox_xmin','bbox_ymin','bbox_xmax','bbox_ymax','label']  
  
# Updated class names cho 4 class (gộp Vandalism)  
class_names = ['Vandalism', 'LookingAround', 'UnauthorizedFilming', 'ClimbOverFence']  
frame_size = [640, 640]  
  
# GIỚI HẠN VIDEOS PER CLASS  
MAX_VIDEOS_PER_CLASS = 200  
  
def normalize_points_with_size(points_xy, width, height, flip=False):  
    """Normalize keypoints to [0,1] range"""  
    points_xy[:, 0] /= width  
    points_xy[:, 1] /= height  
    if flip:  
        points_xy[:, 0] = 1 - points_xy[:, 0]  
    return points_xy  
  
def process_yolov8_results(results, thresh=0.01):  
    """Process YOLOv8 pose results và convert sang format tương thích với YL2XY"""  
    if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints.data) == 0:  
        return np.zeros([17, 3]), False, [0, 0, 640, 640]  
        
    # Get first detection  
    keypoints = results[0].keypoints.data[0].cpu().numpy()  # Shape: (17, 3)  
        
    # Get bbox if available  
    if results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:  
        bbox = results[0].boxes.xyxy[0].cpu().numpy()  # [xmin, ymin, xmax, ymax]  
    else:  
        bbox = np.array([0, 0, 640, 640])  
        
    # Convert to expected format tương thích với YL2XY logic  
    result = np.zeros([17, 3])  
    cf = True  
        
    for i in range(17):  
        if i < len(keypoints):  
            x, y, conf = keypoints[i]  
            # Check if coordinates are valid (not at image borders)  
            if not (x % 640 == 0 or y % 640 == 0):  
                result[i, 0] = x  
                result[i, 1] = y  
                result[i, 2] = conf  
                    
                # Check confidence for main body parts như trong YL2XY  
                if (conf < thresh) and (i in [5, 6, 11, 12, 13, 14]):  
                    cf = False  
        
    return result, cf, bbox  
  
def get_balanced_video_list_by_source(annot, max_per_class=200):  
    """Pre-filter video list để cân bằng theo folder nguồn cho Vandalism"""  
    class_video_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 4 classes  
    vandalism_sources = {'damage-to-facilities': 0, 'damage-to-fence': 0}  
    selected_videos = []  
        
    print(f"Selecting maximum {max_per_class} videos per class...")  
        
    for vid in annot['video'].unique():  
        video_class = annot[annot['video'] == vid]['label'].iloc[0]  
          
        # Đặc biệt xử lý class Vandalism (class 0)  
        if video_class == 0:  
            # Xác định nguồn folder từ video path  
            video_pattern = os.path.join(video_folder, '**', vid)  
            video_matches = glob.glob(video_pattern, recursive=True)  
              
            if video_matches:  
                video_path = video_matches[0]  
                if 'damage-to-facilities' in video_path:  
                    source = 'damage-to-facilities'  
                elif 'damage-to-fence' in video_path:  
                    source = 'damage-to-fence'  
                else:  
                    continue  
                  
                # Lấy tối đa 100 video từ mỗi nguồn (tổng 200 cho Vandalism)  
                if vandalism_sources[source] < max_per_class // 2:  
                    selected_videos.append(vid)  
                    class_video_counts[video_class] += 1  
                    vandalism_sources[source] += 1  
        else:  
            # Các class khác xử lý bình thường  
            if class_video_counts[video_class] < max_per_class:  
                selected_videos.append(vid)  
                class_video_counts[video_class] += 1  
                
        # Stop early nếu đã đủ videos cho tất cả classes  
        if len(selected_videos) >= max_per_class * 4:  # 200 * 4 classes  
            break  
        
    print("Selected videos per class:")  
    for class_id, count in class_video_counts.items():  
        if class_id == 0:  
            print(f"  - Class {class_id} (Vandalism): {count} videos")  
            print(f"    + damage-to-facilities: {vandalism_sources['damage-to-facilities']} videos")  
            print(f"    + damage-to-fence: {vandalism_sources['damage-to-fence']} videos")  
        else:  
            print(f"  - Class {class_id} ({class_names[class_id]}): {count} videos")  
        
    return selected_videos  
  
# Load annotation file từ create_dataset_1.py  
try:  
    annot = pd.read_csv(annot_file)  
    print(f"Loaded annotation file with {len(annot)} samples")  
    print("Original class distribution:", annot['label'].value_counts().sort_index())  
except FileNotFoundError:  
    print(f"Error: Annotation file not found at {annot_file}")  
    print("Please run create_dataset_1.py first!")  
    exit()  
  
# Pre-filter video list để cân bằng videos per class và folder nguồn  
vid_list = get_balanced_video_list_by_source(annot, max_per_class=MAX_VIDEOS_PER_CLASS)  
print(f"\nProcessing {len(vid_list)} selected videos (instead of {len(annot['video'].unique())} total)...")  
  
processed_videos = 0  
total_frames_processed = 0  
class_video_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 4 classes  
  
for vid in vid_list:  
    # Get class của video hiện tại  
    video_class = annot[annot['video'] == vid]['label'].iloc[0]  
        
    print(f'Process on: {vid} ({processed_videos + 1}/{len(vid_list)}) - Class {video_class} ({class_names[video_class]})')  
    df = pd.DataFrame(columns=columns)  
    cur_row = 0  
  
    # Get labels cho video hiện tại  
    frames_label = annot[annot['video'] == vid].reset_index(drop=True)  
        
    # Tìm video file trong nested structure  
    video_pattern = os.path.join(video_folder, '**', vid)  
    video_matches = glob.glob(video_pattern, recursive=True)  
        
    if not video_matches:  
        print(f"Error: Cannot find video file {vid}")  
        continue  
            
    video_path = video_matches[0]  
    cap = cv2.VideoCapture(video_path)  
        
    if not cap.isOpened():  
        print(f"Error: Cannot open video {video_path}")  
        continue  
            
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    print(f"Video {vid}: {frames_count} frames")  
        
    i = 1  
    frames_with_pose = 0  
        
    while True:  
        ret, frame = cap.read()  
        if ret:  
            # Resize frame  
            frame = cv2.resize(frame, frame_size)  
                
            # Get label cho frame hiện tại  
            frame_data = frames_label[frames_label['frame'] == i]  
            if not frame_data.empty:  
                cls_idx = int(frame_data['label'].iloc[0])  
            else:  
                cls_idx = 0  # Default to Vandalism  
                
            # YOLOv8 inference  
            try:  
                results = model(frame, verbose=False)  
                result, cf, bbox = process_yolov8_results(results)  
                    
                if cf:  
                    # Normalize keypoints  
                    pt_norm = normalize_points_with_size(result.copy(), frame_size[0], frame_size[1])  
                        
                    # Extract bbox coordinates  
                    xmin, ymin, xmax, ymax = bbox  
                        
                    # Create row with keypoints + bbox + label  
                    row = [vid, i, *pt_norm.flatten().tolist(), xmin, ymin, xmax, ymax, cls_idx]  
                    scr = result[:, 2].mean()  
                    frames_with_pose += 1  
                else:  
                    # No valid pose detected  
                    row = [vid, i, *[np.nan] * (17 * 3), 0, 0, 0, 0, cls_idx]  
                    scr = 0.0  
                        
            except Exception as e:  
                print(f"Error processing frame {i}: {e}")  
                # No keypoints detected  
                row = [vid, i, *[np.nan] * (17 * 3), 0, 0, 0, 0, cls_idx]  
                scr = 0.0  
  
            # Add row to dataframe  
            df.loc[cur_row] = row  
            cur_row += 1  
            i += 1  
                
            # Progress indicator mỗi 100 frames  
            if i % 100 == 0:  
                print(f"  Processed {i}/{frames_count} frames, poses detected: {frames_with_pose}")  
                    
        else:  
            break  
  
    cap.release()  
    total_frames_processed += len(df)  
  
    # Save to CSV  
    if os.path.exists(save_path):  
        df.to_csv(save_path, mode='a', header=False, index=False)  
    else:  
        df.to_csv(save_path, mode='w', index=False)  
        
    processed_videos += 1  
    class_video_counts[video_class] += 1  
        
    print(f"Completed {vid}: {len(df)} frames saved, {frames_with_pose} with valid poses")  
    print(f"Progress: {processed_videos}/{len(vid_list)} videos, {total_frames_processed} total frames processed")  
  
print(f"\nBalanced dataset processing completed!")  
print(f"Total videos processed: {processed_videos}")  
print(f"Total frames processed: {total_frames_processed}")  
print(f"Videos per class:")  
for class_id, count in class_video_counts.items():  
    print(f"  - Class {class_id} ({class_names[class_id]}): {count} videos")  
print(f"Output saved to: {save_path}")