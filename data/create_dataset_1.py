import os  
import cv2  
import numpy as np  
import pandas as pd  
import sys  
import glob  
  
class_names = ['Vandalism', 'LookingAround', 'UnauthorizedFilming', 'ClimbOverFence']  
  
video_folder = '/mnt/myhdd/anhnv/intrustion_ditection/data/Train/extracted_images/intrusion'  
annot_file_2 = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/annotation_4classes.csv'  
  
# GIỚI HẠN VIDEOS PER CLASS  
MAX_VIDEOS_PER_CLASS = 200  
  
def get_balanced_video_list_by_source(video_files, max_per_class=200):  
    """Cân bằng video theo class và folder nguồn"""  
    class_video_counts = {0: 0, 1: 0, 2: 0, 3: 0}  
    vandalism_sources = {'damage-to-facilities': 0, 'damage-to-fence': 0}  
    selected_videos = []  
      
    for video_file in video_files:  
        path_parts = video_file.split('/')  
        category = None  
        for part in path_parts:  
            if part in ['climb-over-fence', 'damage-to-facilities', 'damage-to-fence', 'facility-filming', 'theft']:  
                category = part  
                break  
          
        if category is None:  
            continue  
              
        # Mapping category to class  
        if 'damage-to-facilities' in category or 'damage-to-fence' in category:  
            class_id = 0  # Vandalism  
            if 'damage-to-facilities' in category:  
                source = 'damage-to-facilities'  
            else:  
                source = 'damage-to-fence'  
                  
            # Lấy tối đa 100 video từ mỗi nguồn cho Vandalism  
            if vandalism_sources[source] < max_per_class // 2:  
                selected_videos.append(video_file)  
                class_video_counts[class_id] += 1  
                vandalism_sources[source] += 1  
        elif 'theft' in category:  
            class_id = 1  # LookingAround  
            if class_video_counts[class_id] < max_per_class:  
                selected_videos.append(video_file)  
                class_video_counts[class_id] += 1  
        elif 'facility-filming' in category:  
            class_id = 2  # UnauthorizedFilming  
            if class_video_counts[class_id] < max_per_class:  
                selected_videos.append(video_file)  
                class_video_counts[class_id] += 1  
        elif 'climb-over-fence' in category:  
            class_id = 3  # ClimbOverFence  
            if class_video_counts[class_id] < max_per_class:  
                selected_videos.append(video_file)  
                class_video_counts[class_id] += 1  
      
    print("Selected videos per class:")  
    for class_id, count in class_video_counts.items():  
        if class_id == 0:  
            print(f"  - Class {class_id} (Vandalism): {count} videos")  
            print(f"    + damage-to-facilities: {vandalism_sources['damage-to-facilities']} videos")  
            print(f"    + damage-to-fence: {vandalism_sources['damage-to-fence']} videos")  
        else:  
            print(f"  - Class {class_id} ({class_names[class_id]}): {count} videos")  
      
    return selected_videos  
  
video_files = glob.glob(os.path.join(video_folder, '**/*.mp4'), recursive=True)  
selected_videos = get_balanced_video_list_by_source(video_files, MAX_VIDEOS_PER_CLASS)  
  
cols = ['video', 'frame', 'label']  
df = pd.DataFrame(columns=cols)  
  
for video_file in selected_videos:  
    video_name = os.path.basename(video_file)  
    print(f"Processing: {video_name}")  
      
    # Extract category và mapping  
    path_parts = video_file.split('/')  
    category = None  
    for part in path_parts:  
        if part in ['climb-over-fence', 'damage-to-facilities', 'damage-to-fence', 'facility-filming', 'theft']:  
            category = part  
            break  
      
    # Auto-detect class từ category (gộp Vandalism)  
    if 'damage-to-facilities' in category or 'damage-to-fence' in category:  
        default_label = 0  # Vandalism  
    elif 'theft' in category:  
        default_label = 1  # LookingAround  
    elif 'facility-filming' in category:  
        default_label = 2  # UnauthorizedFilming  
    elif 'climb-over-fence' in category:  
        default_label = 3  # ClimbOverFence  
    else:  
        continue  
      
    cap = cv2.VideoCapture(video_file)  
    if not cap.isOpened():  
        continue  
          
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
      
    # AUTO ANNOTATION  
    video = np.array([video_name] * frames_count)  
    frame_idx = np.arange(1, frames_count + 1)  
    label = np.array([default_label] * frames_count)  
      
    rows = np.stack([video, frame_idx, label], axis=1)  
    df = pd.concat([df, pd.DataFrame(rows, columns=cols)], ignore_index=True)  
      
    cap.release()  
  
df.to_csv(annot_file_2, index=False)  
print(f"Dataset created with {len(df['video'].unique())} videos and 4 classes")