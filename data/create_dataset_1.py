import os    
import cv2    
import numpy as np    
import pandas as pd    
import sys    
import glob    
import pickle    

class_names = ['Vandalism', 'UnauthorizedFilming', 'ClimbOverFence', 'Normal']    
video_folder = '/mnt/myhdd/anhnv/intrustion_ditection/data/Train/extracted_images/intrusion'    
annot_file_2 = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/annotation_4classes.csv'    
metadata_file = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/video_metadata.pkl'    
MAX_VIDEOS_PER_CLASS = 100    
  
def get_padding_factor(category):    
    """Dynamic padding based on intrusion type for bbox optimization"""    
    padding_factors = {    
        'climb-over-fence': 0.3,         
        'facility-filming': 0.2,         
        'damage-to-facilities': 0.25,     
        'damage-to-fence': 0.25,          
        'normal': 0.15                    
    }    
    return padding_factors.get(category, 0.2)    
  
def get_bbox_expansion(category):  
    """Fixed bbox expansion - 10 pixels each direction"""  
    return 10  # Fixed 10 pixels for all categories  
  
def get_motion_complexity(category):    
    complexity_factors = {    
        'climb-over-fence': 'high',       
        'facility-filming': 'medium',       
        'damage-to-facilities': 'high',    
        'damage-to-fence': 'high',         
        'normal': 'low'                   
    }    
    return complexity_factors.get(category, 'medium')    
  
def expand_bbox_fixed(bbox, expansion_pixels=10):  
    """Expand bbox by fixed pixels in all directions"""  
    if len(bbox) == 4:  
        xmin, ymin, xmax, ymax = bbox  
        return [  
            max(0, xmin - expansion_pixels),  # Expand left  
            max(0, ymin - expansion_pixels),  # Expand top    
            xmax + expansion_pixels,          # Expand right  
            ymax + expansion_pixels           # Expand bottom  
        ]  
    return bbox  
  
def get_balanced_video_list_enhanced(video_files, max_per_class):    
    """Enhanced version with video metadata for bbox and motion optimization"""    
    selected_videos = []    
    video_metadata = {}    
  
    category_to_class_id = {    
        'climb-over-fence': 2,    
        'facility-filming': 1,    
        'normal': 3,    
        'damage-to-facilities': 0,    
        'damage-to-fence': 0,    
    }    
  
    videos_by_class = {cls: [] for cls in range(len(class_names))}    
    vandalism_sources = {'damage-to-facilities': [], 'damage-to-fence': []}    
  
    for video_file in video_files:    

        relative_path = os.path.relpath(video_file, start=video_folder)    
        category = relative_path.split(os.path.sep)[0]    
  
        if category in category_to_class_id:    
            class_id = category_to_class_id[category]     
            video_name = os.path.basename(video_file)    
            video_metadata[video_name] = {    
                'category': category,    
                'class_id': class_id,    
                'bbox_padding_factor': get_padding_factor(category),    
                'bbox_expansion_pixels': get_bbox_expansion(category),  
                'motion_complexity': get_motion_complexity(category),    
                'relative_path': relative_path    
            }    
  
            if class_id == 0:     
                vandalism_sources[category].append(video_file)    
            else:    
                videos_by_class[class_id].append(video_file)        
    print("Selected videos per class with optimization metadata:")    
    for class_id, class_name in enumerate(class_names):    
        if class_id == 0:  # Vandalism    
            max_per_vandalism_source = max_per_class // 2    
  
            selected_dam_fac = np.random.choice(vandalism_sources['damage-to-facilities'],     
                                               min(len(vandalism_sources['damage-to-facilities']), max_per_vandalism_source),     
                                               replace=False).tolist()    
            selected_dam_fence = np.random.choice(vandalism_sources['damage-to-fence'],     
                                                 min(len(vandalism_sources['damage-to-fence']), max_per_vandalism_source),     
                                                 replace=False).tolist()    
            selected_videos.extend(selected_dam_fac)    
            selected_videos.extend(selected_dam_fence)    
            print(f"  - Class {class_id} ({class_name}): {len(selected_dam_fac) + len(selected_dam_fence)} videos")    
            print(f"    + damage-to-facilities: {len(selected_dam_fac)} videos (expansion: 10px)")    
            print(f"    + damage-to-fence: {len(selected_dam_fence)} videos (expansion: 10px)")    
        else:    
            paths = videos_by_class[class_id]    
            selected_class_videos = np.random.choice(paths, min(len(paths), max_per_class), replace=False).tolist()    
            selected_videos.extend(selected_class_videos)      
            expansion = get_bbox_expansion(category)    
            print(f"  - Class {class_id} ({class_name}): {len(selected_class_videos)} videos (expansion: {expansion}px)")    
  
    return selected_videos, video_metadata    
 
video_files = []    
for category in ['climb-over-fence', 'damage-to-facilities', 'damage-to-fence', 'facility-filming', 'normal']:    
    rgb_path = os.path.join(video_folder, category, 'rgb')       
    video_files.extend(glob.glob(os.path.join(rgb_path, '**', '*.mp4'), recursive=True))    

selected_videos, video_metadata = get_balanced_video_list_enhanced(video_files, MAX_VIDEOS_PER_CLASS)    
  
if not selected_videos:    
    print("Error: No videos were selected. Check your video folder and labels file.")    
    sys.exit(1)    
   
with open(metadata_file, 'wb') as f:    
    pickle.dump(video_metadata, f)    
print(f"Video metadata saved to {metadata_file}")    
  
cols = ['video', 'frame', 'label']    
df = pd.DataFrame(columns=cols)    
  
for video_file in selected_videos:    
    relative_path = os.path.relpath(video_file, start=video_folder)    
    print(f"Processing: {relative_path}")      
    category = relative_path.split(os.path.sep)[0]    
  
    if 'damage-to-facilities' in category or 'damage-to-fence' in category:    
        default_label = 0    
    elif 'facility-filming' in category:    
        default_label = 1    
    elif 'climb-over-fence' in category:    
        default_label = 2    
    elif 'normal' in category:    
        default_label = 3    
    else:    
        continue    
  
    cap = cv2.VideoCapture(video_file)    
    if not cap.isOpened():    
        print(f"Warning: Could not open video file {video_file}")    
        continue    
  
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    
  
    video = np.array([relative_path] * frames_count)    
    frame_idx = np.arange(1, frames_count + 1)    
    label = np.array([default_label] * frames_count)    
  
    rows = np.stack([video, frame_idx, label], axis=1)    
    df = pd.concat([df, pd.DataFrame(rows, columns=cols)], ignore_index=True)    
    cap.release()    
  
df.to_csv(annot_file_2, index=False)    
print(f"Dataset created with {len(df['video'].unique())} videos and 4 classes: {class_names}")    
print("Enhanced pipeline Stage 1 completed with bbox expansion optimization!")