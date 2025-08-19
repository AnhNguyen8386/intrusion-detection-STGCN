import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import torch.nn as nn
from models.stgcn import EnhancedMultimodalGraph
from dataloader.dataset import processing_data
import os
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

class RealTimeActionRecognition:
    def __init__(self, model_path, pose_model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.class_names = ['Vandalism', 'UnauthorizedFilming', 'ClimbOverFence', 'Normal']
        graph_args = {'strategy': 'spatial'}
        self.action_model = EnhancedMultimodalGraph(graph_args, len(self.class_names)).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)

        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key[7:] 
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        self.action_model.load_state_dict(state_dict)
        self.action_model.eval() 
        self.pose_model = YOLO(pose_model_path)
        if torch.cuda.is_available():
            self.pose_model.to(self.device)

        print("Loading CNN feature extractor (ResNet18)...")
        self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor.fc = torch.nn.Identity() 
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval() 
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.sequence_length = 45
        self.pose_buffer = deque(maxlen=self.sequence_length)
        self.visual_buffer = deque(maxlen=self.sequence_length)
        self.bbox_buffer = deque(maxlen=self.sequence_length)
        self.frame_size = [640, 640] 
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]

    def normalize_points_with_size(self, points_xy, width, height):

        if points_xy.size == 0:
            return np.zeros_like(points_xy)
        points_xy[:, 0] /= width
        points_xy[:, 1] /= height
        return points_xy

    def extract_bbox_visual_features(self, frame, bbox):

        try:
            if np.all(bbox == 0): 
                return np.zeros(512, dtype=np.float32)

            xmin, ymin, xmax, ymax = bbox
            h_frame, w_frame = frame.shape[:2]
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(w_frame, int(xmax))
            ymax = min(h_frame, int(ymax))

            if xmax <= xmin or ymax <= ymin or (xmax - xmin) < 10 or (ymax - ymin) < 10:
                return np.zeros(512, dtype=np.float32)
            bbox_region = frame[ymin:ymax, xmin:xmax]
            bbox_rgb = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2RGB)
            bbox_pil = Image.fromarray(bbox_rgb)
            bbox_tensor = self.transform(bbox_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                visual_features = self.feature_extractor(bbox_tensor)
                return visual_features.cpu().squeeze().numpy().astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting visual features: {e}. Returning zeros.")
            return np.zeros(512, dtype=np.float32)

    def process_yolov8_results(self, results, thresh=0.25):

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
            if (conf < thresh) and (i in [5, 6, 11, 12, 13, 14]): 
                is_pose_valid = False

        if np.sum(pose[:, 2]) < 0.5 * 17 * thresh:
             is_pose_valid = False
        
        return pose, bbox, is_pose_valid

    def draw_skeleton_connections(self, frame, keypoints, width, height):

        for connection in self.skeleton:
            kpt_a, kpt_b = connection
            x1, y1, conf1 = keypoints[kpt_a - 1]
            x2, y2, conf2 = keypoints[kpt_b - 1]
            if conf1 > 0.3 and conf2 > 0.3:
                x1_scaled, y1_scaled = int(x1 * width / self.frame_size[0]), int(y1 * height / self.frame_size[1])
                x2_scaled, y2_scaled = int(x2 * width / self.frame_size[0]), int(y2 * height / self.frame_size[1])
                cv2.line(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 255), 2)
        
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:
                x_scaled, y_scaled = int(x * width / self.frame_size[0]), int(y * height / self.frame_size[1])
                color = (0, 255, 0) if i in [5, 6, 11, 12] else (255, 0, 0) 
                cv2.circle(frame, (x_scaled, y_scaled), 4, color, -1)

    def predict_action_multimodal(self, pose_sequence, visual_sequence, bbox_sequence):
        if len(pose_sequence) < self.sequence_length:
            return None, 0.0

        try:
            pose_seq_downsampled = np.array(pose_sequence)[::2, :, :]
            visual_seq_downsampled = np.array(visual_sequence)[::2, :]
            bbox_seq_downsampled = np.array(bbox_sequence)[::2, :]
            pose_seq_downsampled[:, :, :2] = processing_data(pose_seq_downsampled[:, :, :2])
            pose_tensor = torch.tensor(pose_seq_downsampled, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            visual_tensor = torch.tensor(visual_seq_downsampled, dtype=torch.float32).unsqueeze(0).to(self.device)
            bbox_tensor = torch.tensor(bbox_seq_downsampled, dtype=torch.float32).unsqueeze(0).to(self.device)
            mot_tensor = pose_tensor[:, :2, 1:, :] - pose_tensor[:, :2, :-1, :]
            
            min_T = mot_tensor.shape[2]
            pose_tensor_resized = pose_tensor[:, :, :min_T, :]
            visual_tensor_resized = visual_tensor[:, :min_T, :]
            bbox_tensor_resized = bbox_tensor[:, :min_T, :]

            with torch.no_grad():
                outputs = self.action_model((pose_tensor_resized, mot_tensor, visual_tensor_resized, bbox_tensor_resized))

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    return None, 0.0

                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = torch.softmax(outputs, dim=1)[0][predicted_class].item()

                return predicted_class, confidence

        except Exception as e:
            print(f"Error in multimodal prediction: {e}. Returning None, 0.0.")
            return None, 0.0

    def run_video_with_recording(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video source")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {video_path}")
        print(f"Output will be saved to: {output_path}")

        frame_count = 0
        start_time = time.time()
        
        last_prediction = "Collecting frames..."
        last_confidence = "N/A"

        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            
            frame_count += 1
            frame_resized = cv2.resize(frame, self.frame_size) 
            results = self.pose_model(frame_resized, verbose=False)
            pose_data, bbox_data, pose_valid = self.process_yolov8_results(results)
            bbox_original_scaled = bbox_data.copy()
            if pose_valid and bbox_data.sum() > 0:
                bbox_original_scaled[0] *= (width / self.frame_size[0])
                bbox_original_scaled[1] *= (height / self.frame_size[1])
                bbox_original_scaled[2] *= (width / self.frame_size[0])
                bbox_original_scaled[3] *= (height / self.frame_size[1])

            visual_features = self.extract_bbox_visual_features(frame, bbox_original_scaled)

            if pose_valid: 
                pose_normalized = self.normalize_points_with_size(pose_data.copy()[:, :2], self.frame_size[0], self.frame_size[1])
                pose_with_conf = np.hstack((pose_normalized, pose_data[:, 2:])) 
                self.pose_buffer.append(pose_with_conf)
                self.visual_buffer.append(visual_features)
                self.bbox_buffer.append(bbox_data)

                if bbox_data.sum() > 0:
                    xmin, ymin, xmax, ymax = bbox_original_scaled
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
                self.draw_skeleton_connections(frame, results[0].keypoints.data[0].cpu().numpy(), width, height)
            else:
                self.pose_buffer.append(np.zeros([17, 3]))
                self.visual_buffer.append(np.zeros(512))
                self.bbox_buffer.append(np.zeros(4))
            if len(self.pose_buffer) == self.sequence_length:
                predicted_class, confidence = self.predict_action_multimodal(
                    list(self.pose_buffer), list(self.visual_buffer), list(self.bbox_buffer)
                )
                if predicted_class is not None and not np.isnan(confidence):
                    last_prediction = f"Action: {self.class_names[predicted_class]}"
                    last_confidence = f"Confidence: {confidence:.3f}"
                else:
                    last_prediction = "No valid prediction"
                    last_confidence = "Confidence: N/A"
                self.pose_buffer.clear()
                self.visual_buffer.clear()
                self.bbox_buffer.clear()

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (650, 120), (0, 0, 0), -1) 
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0) 
            
            cv2.putText(frame, last_prediction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, last_confidence, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)            
            out.write(frame) 

            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps_processed = frame_count / elapsed_time
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                      f"Processing speed: {fps_processed:.1f} FPS")

        cap.release() 
        out.release() 
        
        total_time = time.time() - start_time
        print(f"Video processing completed! Total time: {total_time:.1f} seconds")
        print(f"Average processing speed: {frame_count / total_time:.1f} FPS")
        print(f"Output saved to: {output_path}")

def main():
    model_path = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/runs/exp1/best.pt'
    pose_model_path = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/result_model/yolov7-pose/yolo11n-pose.pt'
    if not os.path.exists(pose_model_path):
        print(f"Downloading {pose_model_path}...")
        YOLO(pose_model_path) 
    input_video = '/mnt/myhdd/anhnv/intrustion_ditection/data/Train/extracted_images/intrusion/climb-over-fence/rgb/intrusion_climb-over-fence_rgb_0001_cctv1/intrusion_climb-over-fence_rgb_0001_cctv1.mp4'
    output_video = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/output/video_out_multimodal.mp4'

    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    recognizer = RealTimeActionRecognition(model_path, pose_model_path)
    recognizer.run_video_with_recording(input_video, output_video)

if __name__ == '__main__':
    main()