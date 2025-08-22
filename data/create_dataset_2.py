import os    
import sys    
import pandas as pd    
import numpy as np    
import torch    
import cv2    
import pickle    
import concurrent.futures    
from tqdm import tqdm    
from pathlib import Path    
from ultralytics import YOLO    
from torchvision.models import resnet18, ResNet18_Weights    
from PIL import Image    
import torchvision.transforms as transforms    

class_names = ['Vandalism', 'UnauthorizedFilming', 'ClimbOverFence', 'Normal']    
video_folder = '/mnt/myhdd/anhnv/intrustion_ditection/data/Train/extracted_images/intrusion'    
annot_file = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/annotation_4classes.csv'    
metadata_file = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/video_metadata.pkl'    
output_csv = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/pose_visual_bbox_4classes.csv'    
pose_model_path = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/result_model/yolov7-pose/yolo11n-pose.pt'    
    
NUM_THREADS = 4    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

def load_video_metadata():    
    """Load and filter video metadata for selected videos only"""    
    try:    
        with open(metadata_file, 'rb') as f:    
            all_metadata = pickle.load(f)    

        if not os.path.exists(annot_file):  
            print(f"Error: Annotation file not found at {annot_file}")  
            sys.exit(1)  
              
        labels_df = pd.read_csv(annot_file)  
        selected_video_names = set(labels_df['video'].apply(os.path.basename))  
        filtered_metadata = {  
            name: meta for name, meta in all_metadata.items()   
            if name in selected_video_names  
        }  
          
        print(f"Loaded metadata for {len(filtered_metadata)} videos (filtered from {len(all_metadata)} total)")  
        return filtered_metadata  
          
    except FileNotFoundError:    
        print(f"Error: Metadata file not found at {metadata_file}")    
        print("Please run create_dataset_1.py first!")    
        sys.exit(1)    
    
@torch.no_grad()    
def get_feature_models(device):    
    """Loads all models for a single process - Updated for ResNet18"""    
    # Load YOLO model for pose estimation    
    yolo_model = YOLO(pose_model_path)    
    yolo_model.to(device)    
    
    # Load ResNet18 for visual feature extraction (512 features)    
    weights = ResNet18_Weights.DEFAULT    
    visual_model = resnet18(weights=weights).to(device).eval()    
    visual_model.fc = torch.nn.Identity()  # Remove final FC layer    
        
    # Enhanced transforms for better feature extraction    
    visual_transforms = transforms.Compose([    
        transforms.Resize((224, 224)),    
        transforms.ToTensor(),    
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
    ])    
    
    return yolo_model, visual_model, visual_transforms    
    
def get_video_labels(video_labels_file):    
    """Reads the annotation CSV and creates a dictionary mapping video paths to labels"""    
    labels_df = pd.read_csv(video_labels_file)    
    labels_df['video_path'] = labels_df['video'].apply(    
        lambda x: os.path.normpath(os.path.join(video_folder, x))    
    )    
    labels_dict = dict(zip(labels_df['video_path'], labels_df['label']))    
    return labels_dict, labels_df    
    
def expand_bbox_with_metadata(bbox, video_name, metadata, frame_shape):    
    """Expand bbox using metadata-driven approach"""    
    if video_name in metadata:    
        expansion_pixels = metadata[video_name].get('bbox_expansion_pixels', 10)    
    else:    
        expansion_pixels = 10      
        
    if len(bbox) == 4:    
        xmin, ymin, xmax, ymax = bbox    
        h_frame, w_frame = frame_shape[:2]       
        xmin_new = max(0, int(xmin - expansion_pixels))    
        ymin_new = max(0, int(ymin - expansion_pixels))    
        xmax_new = min(w_frame, int(xmax + expansion_pixels))    
        ymax_new = min(h_frame, int(ymax + expansion_pixels))    
            
        return [xmin_new, ymin_new, xmax_new, ymax_new]    
    return bbox    
    
def process_yolov8_results(results, thresh=0.25):    
    """Process YOLOv8 pose results - Enhanced version"""    
    if not results or not results[0].boxes or len(results[0].boxes.xyxy) == 0:    
        return np.zeros([17, 3]), np.zeros(4), False     
    best_person_idx = 0    
    if len(results[0].boxes.xyxy) > 1:    
        best_person_idx = torch.argmax(results[0].boxes.conf).item()    
        
    bbox_xyxy = results[0].boxes.xyxy[best_person_idx].cpu().numpy()    
    keypoints_data = results[0].keypoints.data[best_person_idx].cpu().numpy()    
    
    bbox = np.array([bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]])    
    pose = np.zeros([17, 3])    
    is_pose_valid = True    
    
    for i in range(len(keypoints_data)):    
        x, y, conf = keypoints_data[i]    
        if conf > thresh:    
            pose[i] = [x, y, conf]    
        # Check main body parts for pose validation    
        if (conf < thresh) and (i in [5, 6, 11, 12, 13, 14]):    
            is_pose_valid = False    
    
    if np.sum(pose[:, 2]) < 0.5 * 17 * thresh:    
        is_pose_valid = False    
        
    return pose, bbox, is_pose_valid    
    
def process_single_video(video_path, labels_dict, metadata):      
    yolo_model, visual_model, visual_transforms = get_feature_models(device)            
    vid_path_norm = os.path.normpath(video_path)    
    video_label = labels_dict.get(vid_path_norm)    
    video_name = os.path.basename(video_path)    
        
    if video_label is None:    
        print(f"Warning: No label found for {video_path}. Skipping.")    
        return []    
    
    video_data = []    
    cap = cv2.VideoCapture(video_path)    
    if not cap.isOpened():    
        print(f"Error: Could not open video file {video_path}")    
        return []    
    
    frame_idx = 0    
    frames_with_pose = 0    
        
    while cap.isOpened():    
        ret, frame = cap.read()    
        if not ret:    
            break    
  
        pose_data = np.full(17 * 3, np.nan)    
        bbox_data = np.full(4, np.nan)    
        visual_data = np.full(512, np.nan)  # ResNet18 outputs 512 features    
            
        try:       
            frame_resized = cv2.resize(frame, [640, 640])      
            yolo_results = yolo_model(frame_resized, verbose=False)    
            pose, bbox, is_pose_valid = process_yolov8_results(yolo_results)    
                
            if is_pose_valid:    
                pose_normalized = pose.copy()    
                pose_normalized[:, 0] /= 640    
                pose_normalized[:, 1] /= 640     
                pose_data = pose_normalized.flatten()     
                bbox_data = bbox       
                expanded_bbox = expand_bbox_with_metadata(bbox, video_name, metadata, frame.shape)    
                xmin, ymin, xmax, ymax = expanded_bbox    
                    
                if xmax > xmin and ymax > ymin and (xmax - xmin) >= 10 and (ymax - ymin) >= 10:       
                    h_orig, w_orig = frame.shape[:2]    
                    xmin_orig = int(xmin * w_orig / 640)    
                    ymin_orig = int(ymin * h_orig / 640)    
                    xmax_orig = int(xmax * w_orig / 640)    
                    ymax_orig = int(ymax * h_orig / 640)    
                        
                    cropped_frame = frame[ymin_orig:ymax_orig, xmin_orig:xmax_orig]    
                        
                    if cropped_frame.size > 0:      
                        pil_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))    
                        visual_input = visual_transforms(pil_image).unsqueeze(0).to(device)    
                            
                        with torch.no_grad():    
                            visual_features = visual_model(visual_input).squeeze()    
                            visual_data = visual_features.cpu().numpy().flatten()    
                    
                frames_with_pose += 1    
                
        except Exception as e:    
            print(f"Warning: Error processing frame {frame_idx} from {video_path}: {e}")     
        data_row = np.concatenate([    
            [frame_idx, video_label, os.path.basename(video_path)],    
            pose_data,    
            bbox_data,    
            visual_data    
        ])    
        video_data.append(data_row)    
        frame_idx += 1    
        
    cap.release()    
    print(f"Processed {video_name}: {frame_idx} frames, {frames_with_pose} with valid poses")    
    return video_data    
    
def extract_features_and_save_multithreaded(video_list, labels_dict, metadata):    
    """Enhanced feature extraction with metadata support"""    
    pose_cols = [f'pose_{i}_{axis}' for i in range(17) for axis in ['x', 'y', 'score']]    
    bbox_cols = [f'bbox_{i}' for i in range(4)]    
    visual_cols = [f'visual_{i}' for i in range(512)]  # ResNet18: 512 features    
        
    COLUMNS = ['frame_idx', 'label', 'video'] + pose_cols + bbox_cols + visual_cols    
        
    all_data = []    

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:  
        futures = {  
            executor.submit(process_single_video, video_path, labels_dict, metadata): video_path   
            for video_path in video_list  
    }  
            
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(video_list),     
                          desc="Processing videos with enhanced pipeline"):    
            result = future.result()    
            if result:    
                all_data.extend(result)    
        
    print(f"Extracted a total of {len(all_data)} data rows.")    
        
    if not all_data:    
        print("Warning: No data was extracted from any video. Check your video files and model paths.")    
        return    
    
    try:    
        data_df = pd.DataFrame(all_data, columns=COLUMNS)    
        data_df.to_csv(output_csv, index=False)    
        print(f"Enhanced multimodal data saved to {output_csv}")    
        print("Final class distribution in CSV:")    
        print(data_df['label'].value_counts().sort_index())      
        print(f"\nEnhanced pipeline Stage 2 completed!")    
        print(f"- Used ResNet18 for 512-dim visual features")    
        print(f"- Applied metadata-driven bbox expansion (10px)")    
        print(f"- Processed {len(data_df['video'].unique())} videos")    
            
    except Exception as e:    
        print(f"Error saving data to CSV: {e}")    

if __name__ == '__main__':     
    metadata = load_video_metadata()      
    if not os.path.exists(annot_file):    
        print(f"Error: Labels file not found at {annot_file}. Please run create_dataset_1.py first.")    
        sys.exit(1)    
        
    video_labels_dict, labels_df = get_video_labels(annot_file)    
        
 
    selected_videos = labels_df['video_path'].unique().tolist()    
        
    if not selected_videos:    
        print("Error: No videos were selected. Check your video folder and labels file.")    
        sys.exit(1)    
        
    print(f"\nProcessing {len(selected_videos)} videos with enhanced multimodal pipeline...")    
    print(f"Using filtered metadata from {len(metadata)} videos for optimization")    
        
    extract_features_and_save_multithreaded(selected_videos, video_labels_dict, metadata)    
        
    print("Enhanced feature extraction finished.")