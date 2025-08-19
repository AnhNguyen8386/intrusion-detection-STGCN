import os            
import cv2            
import time            
import pandas as pd            
import numpy as np            
from ultralytics import YOLO            
import glob    
import torch    
import torchvision.transforms as transforms    
import torchvision.models as models    
from PIL import Image    
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
print(f"Using device: {device}")  
print(f"GPU available: {torch.cuda.is_available()}")  
if torch.cuda.is_available():  
    print(f"GPU name: {torch.cuda.get_device_name(0)}")  
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")  
  
model = YOLO('/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/result_model/yolov7-pose/yolo11n-pose.pt')  
if torch.cuda.is_available():  
    model.to(device)  

print("Loading CNN feature extractor...")    
feature_extractor = models.resnet18(pretrained=True)    
feature_extractor.fc = torch.nn.Identity()  
feature_extractor = feature_extractor.to(device) 
feature_extractor.eval()    
 
transform = transforms.Compose([    
    transforms.Resize((224, 224)),    
    transforms.ToTensor(),    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])    
     
save_path = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/pose_bbox_visual_4classes.csv'            
annot_file = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/annotation_4classes.csv'            
video_folder = '/mnt/myhdd/anhnv/intrustion_ditection/data/Train/extracted_images/intrusion'            
         
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
           'bbox_xmin','bbox_ymin','bbox_xmax','bbox_ymax',    
           # ADDED: 512 visual features from ResNet18    
           *[f'visual_feat_{i}' for i in range(512)],    
           'label']            
  
class_names = ['Vandalism', 'UnauthorizedFilming', 'ClimbOverFence', 'Normal']            
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
  
def clear_gpu_cache():  
    """Clear GPU cache to prevent memory issues"""  
    if torch.cuda.is_available():  
        torch.cuda.empty_cache()  
  
def extract_bbox_visual_features(frame, bbox, feature_extractor, transform, device):  
    """Extract visual features from bbox region using CNN on GPU"""  
    try:  
        xmin, ymin, xmax, ymax = bbox  
        h, w = frame.shape[:2]  
        xmin = max(0, min(int(xmin), w-1))  
        ymin = max(0, min(int(ymin), h-1))  
        xmax = max(xmin+1, min(int(xmax), w))  
        ymax = max(ymin+1, min(int(ymax), h))  
        bbox_region = frame[ymin:ymax, xmin:xmax]  
          
        if bbox_region.size == 0 or bbox_region.shape[0] < 10 or bbox_region.shape[1] < 10:  
            return np.zeros(512, dtype=np.float32)  

        bbox_rgb = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2RGB)  
        bbox_pil = Image.fromarray(bbox_rgb)  
        bbox_tensor = transform(bbox_pil).unsqueeze(0).to(device)  
          
        with torch.no_grad():  
            visual_features = feature_extractor(bbox_tensor)  
            visual_features = visual_features.cpu().squeeze().numpy()  
          
        return visual_features.astype(np.float32)  
          
    except Exception as e:  
        print(f"Error extracting visual features: {e}")  
        return np.zeros(512, dtype=np.float32)  
  
def process_yolov8_results(results, thresh=0.01):            
    """Process YOLOv8 pose results và convert sang format tương thích với YL2XY"""            
    if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints.data) == 0:            
        return np.zeros([17, 3]), False, [0, 0, 640, 640]            
          
    keypoints = results[0].keypoints.data[0].cpu().numpy()  # Shape: (17, 3)            
         
    if results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:            
        bbox = results[0].boxes.xyxy[0].cpu().numpy()  # [xmin, ymin, xmax, ymax]            
    else:            
        bbox = np.array([0, 0, 640, 640])            
           
    result = np.zeros([17, 3])            
    cf = True            
                  
    for i in range(17):            
        if i < len(keypoints):            
            x, y, conf = keypoints[i]            
            if not (x % 640 == 0 or y % 640 == 0):            
                result[i, 0] = x            
                result[i, 1] = y            
                result[i, 2] = conf            
         
                if (conf < thresh) and (i in [5, 6, 11, 12, 13, 14]):            
                    cf = False            
                  
    return result, cf, bbox            
  
def get_balanced_video_list_by_source(annot, max_per_class=200):                    
    class_video_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 4 classes            
    vandalism_sources = {'damage-to-facilities': 0, 'damage-to-fence': 0}            
    selected_videos = []            
                  
    print(f"Selecting maximum {max_per_class} videos per class...")            
    print(f"Target classes: {class_names}")          
                  
    for vid in annot['video'].unique():            
        video_class = annot[annot['video'] == vid]['label'].iloc[0]            
     
        if video_class == 0:                   
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
         
                if vandalism_sources[source] < max_per_class // 2:            
                    selected_videos.append(vid)            
                    class_video_counts[video_class] += 1            
                    vandalism_sources[source] += 1            
        else:                   
            if class_video_counts[video_class] < max_per_class:            
                selected_videos.append(vid)            
                class_video_counts[video_class] += 1            
       
        if len(selected_videos) >= max_per_class * 4:         
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
  
try:            
    annot = pd.read_csv(annot_file)            
    print(f"Loaded annotation file with {len(annot)} samples")            
    print("Class distribution:", annot['label'].value_counts().sort_index())                
    print("Using direct class mapping from create_dataset_1.py:")      
    print("  - Class 0: Vandalism (damage-to-facilities + damage-to-fence)")      
    print("  - Class 1: UnauthorizedFilming (facility-filming)")        
    print("  - Class 2: ClimbOverFence (climb-over-fence)")      
    print("  - Class 3: Normal (normal)")      
              
except FileNotFoundError:            
    print(f"Error: Annotation file not found at {annot_file}")            
    print("Please run create_dataset_1.py first!")            
    exit()            
           
vid_list = get_balanced_video_list_by_source(annot, max_per_class=MAX_VIDEOS_PER_CLASS)            
print(f"\nProcessing {len(vid_list)} selected videos (instead of {len(annot['video'].unique())} total)...")            
  
processed_videos = 0            
total_frames_processed = 0            
class_video_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 4 classes            
  
for vid in vid_list:                       
    video_class = annot[annot['video'] == vid]['label'].iloc[0]            
                  
    print(f'Process on: {vid} ({processed_videos + 1}/{len(vid_list)}) - Class {video_class} ({class_names[video_class]})')            
    df = pd.DataFrame(columns=columns)            
    cur_row = 0            
         
    frames_label = annot[annot['video'] == vid].reset_index(drop=True)                       
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
                          
            # Get label
            frame_data = frames_label[frames_label['frame'] == i]            
            if not frame_data.empty:            
                cls_idx = int(frame_data['label'].iloc[0])            
            else:            
                cls_idx = 0  # Default to Vandalism            
                                  
            try:            
                results = model(frame, verbose=False)            
                result, cf, bbox = process_yolov8_results(results)            
                              
                if cf:            
                    # Normalize keypoints            
                    pt_norm = normalize_points_with_size(result.copy(), frame_size[0], frame_size[1])            
                                  
                    # Extract bbox coordinates            
                    xmin, ymin, xmax, ymax = bbox            
                    visual_features = extract_bbox_visual_features(frame, bbox, feature_extractor, transform, device)            
                    row = [vid, i, *pt_norm.flatten().tolist(),     
                           xmin, ymin, xmax, ymax,    
                           *visual_features.tolist(),  # Add 512 visual features    
                           cls_idx]            
                    scr = result[:, 2].mean()            
                    frames_with_pose += 1            
                else:                    
                    row = [vid, i, *[np.nan] * (17 * 3), 0, 0, 0, 0,     
                           *[0.0] * 512,    
                           cls_idx]            
                    scr = 0.0            
                                  
            except Exception as e:            
                print(f"Error processing frame {i}: {e}")                       
                row = [vid, i, *[np.nan] * (17 * 3), 0, 0, 0, 0,    
                       *[0.0] * 512,   
                       cls_idx]            
                scr = 0.0                    
            df.loc[cur_row] = row            
            cur_row += 1            
            i += 1                    
            if i % 100 == 0:            
                print(f"  Processed {i}/{frames_count} frames, poses detected: {frames_with_pose}")  
                if i % 500 == 0:  
                    clear_gpu_cache()  
                              
        else:            
            break            
          
    cap.release()            
    total_frames_processed += len(df)            
          
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