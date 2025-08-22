import cv2  
import torch  
import numpy as np  
from ultralytics import YOLO  
import time  
from collections import deque, OrderedDict  
import os  
import torchvision.transforms as transforms  
import torchvision.models as models  
  
from models.stgcn import EnhancedMultimodalGraph  
from dataloader.dataset import processing_data  
  
class RealTimeActionRecognition:  
    def __init__(self, model_path, pose_model_path):  
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        self.class_names = ['Vandalism', 'UnauthorizedFilming', 'ClimbOverFence', 'Normal']  

        graph_args = {'strategy': 'spatial'}  
        self.action_model = EnhancedMultimodalGraph(  
            graph_args=graph_args,  
            num_classes=4,   
            dropout=0.6  
        ).to(self.device)  

        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)  
        if list(state_dict.keys())[0].startswith('module.'):  
            new_state_dict = OrderedDict()  
            for key, value in state_dict.items():  
                new_state_dict[key[7:]] = value  
            state_dict = new_state_dict  
          
        self.action_model.load_state_dict(state_dict)  
        self.action_model.eval()  

        self.pose_model = YOLO(pose_model_path)  
        print("Loading CNN feature extractor (ResNet18)...")  
        self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  
        self.feature_extractor.fc = torch.nn.Identity()  
        self.feature_extractor = self.feature_extractor.to(self.device)  
        self.feature_extractor.eval()  
          
        self.transform = transforms.Compose([  
            transforms.ToPILImage(),  
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])  
          
        self.sequence_length = 30  # Match training: 30 frames sampled to 15  
        self.pose_buffer = deque(maxlen=self.sequence_length)  
        self.visual_buffer = deque(maxlen=self.sequence_length)  
        self.bbox_buffer = deque(maxlen=self.sequence_length)  
        self.scene_buffer = deque(maxlen=self.sequence_length)  
        self.frame_size = [640, 640]  
  
    def expand_bbox_for_inference(self, bbox, frame_shape, expansion_pixels=20):  
        if len(bbox) == 4:  
            xmin, ymin, xmax, ymax = bbox  
            h_frame, w_frame = frame_shape[:2]  
            xmin_new = max(0, int(xmin - expansion_pixels))  
            ymin_new = max(0, int(ymin - expansion_pixels))  
            xmax_new = min(w_frame, int(xmax + expansion_pixels))  
            ymax_new = min(h_frame, int(ymax + expansion_pixels))  
              
            return [xmin_new, ymin_new, xmax_new, ymax_new]  
        return bbox  
  
    def extract_visual_features(self, frame, bbox):  
        """Extract visual features from expanded bbox region using ResNet18"""  
        try:  
            if np.all(bbox == 0):  
                return np.zeros(512, dtype=np.float32)  

            expanded_bbox = self.expand_bbox_for_inference(bbox, frame.shape)  
            xmin, ymin, xmax, ymax = expanded_bbox  

            h_orig, w_orig = frame.shape[:2]  
            xmin_orig = int(xmin * w_orig / 640)  
            ymin_orig = int(ymin * h_orig / 640)  
            xmax_orig = int(xmax * w_orig / 640)  
            ymax_orig = int(ymax * h_orig / 640)  

            if xmax_orig <= xmin_orig or ymax_orig <= ymin_orig or (xmax_orig - xmin_orig) < 10 or (ymax_orig - ymin_orig) < 10:  
                return np.zeros(512, dtype=np.float32)  

            bbox_region = frame[ymin_orig:ymax_orig, xmin_orig:xmax_orig]  
              
            if bbox_region.size > 0:  
                bbox_tensor = self.transform(bbox_region).unsqueeze(0).to(self.device)  
                  
                with torch.no_grad():  
                    visual_features = self.feature_extractor(bbox_tensor)  
                    return visual_features.cpu().squeeze().numpy().astype(np.float32)  
              
            return np.zeros(512, dtype=np.float32)  
              
        except Exception as e:  
            print(f"Error extracting visual features: {e}")  
            return np.zeros(512, dtype=np.float32)  
  
    def extract_scene_features(self, frame):  
        """Extract scene features from entire frame using ResNet18"""  
        try:  
            frame_tensor = self.transform(frame).unsqueeze(0).to(self.device)  
              
            with torch.no_grad():  
                scene_features = self.feature_extractor(frame_tensor)  
                return scene_features.cpu().squeeze().numpy().astype(np.float32)  
              
        except Exception as e:  
            print(f"Error extracting scene features: {e}")  
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
            pose[i] = [x, y, conf]  
  
        required_kpts = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  
        if any(pose[i][2] < thresh for i in required_kpts) or np.sum(pose[:, 2]) < 0.5 * 17 * thresh:  
            is_pose_valid = False  
          
        return pose, bbox, is_pose_valid  
  
    def predict_action_multimodal(self, pose_sequence, visual_sequence, bbox_sequence, scene_sequence):  
        """Predict action using EnhancedMultimodalGraph with all modalities"""  
        if len(pose_sequence) < self.sequence_length:  
            return None, 0.0  
  
        try:  
            # Ensure we have exactly 15 frames  
            if len(pose_sequence) > self.sequence_length:  
                pose_sequence = pose_sequence[-self.sequence_length:]  
                visual_sequence = visual_sequence[-self.sequence_length:]  
                bbox_sequence = bbox_sequence[-self.sequence_length:]  
                scene_sequence = scene_sequence[-self.sequence_length:]  
              
            # Process pose data - Shape: (15, 17, 3)  
            pose_seq = np.array(pose_sequence)  
            pose_seq[:, :, :2] = processing_data(pose_seq[:, :, :2])  
            pose_tensor = torch.tensor(pose_seq, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)  
              
            # Calculate motion data (2 channels: x, y motion)  
            mot_tensor = pose_tensor[:, :2, 1:, :] - pose_tensor[:, :2, :-1, :]  
            padding = torch.zeros(pose_tensor.size(0), 2, 1, pose_tensor.size(3), device=self.device)  
            mot_tensor = torch.cat([padding, mot_tensor], dim=2)  
              
            # Process visual features - Shape: (15, 512)  
            visual_seq = np.array(visual_sequence)  
            visual_tensor = torch.tensor(visual_seq, dtype=torch.float32).unsqueeze(0).to(self.device)  
              
            # Process bbox data - Shape: (15, 4) - Normalize to [0,1] range  
            bbox_seq = np.array(bbox_sequence)  
            bbox_normalized = bbox_seq.copy().astype(float)  
            bbox_normalized[:, [0, 2]] /= 640.0  # Normalize x coordinates  
            bbox_normalized[:, [1, 3]] /= 640.0  # Normalize y coordinates  
            bbox_tensor = torch.tensor(bbox_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)  
              
            # Process scene features - Shape: (15, 512)  
            scene_seq = np.array(scene_sequence)  
            scene_tensor = torch.tensor(scene_seq, dtype=torch.float32).unsqueeze(0).to(self.device)  
              
            with torch.no_grad():  
                outputs = self.action_model([pose_tensor, mot_tensor, visual_tensor, bbox_tensor, scene_tensor])  
                  
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():  
                    print("Warning: NaN or Inf detected in outputs.")  
                    return None, 0.0  
                  
                # Apply sigmoid for multi-class classification (matching training)  
                outputs = torch.sigmoid(outputs)  
                predicted_class = torch.argmax(outputs, dim=1).item()  
                confidence = outputs[0][predicted_class].item()  
                  
                return predicted_class, confidence  
                  
        except Exception as e:  
            print(f"Error in multimodal prediction: {e}")  
            return None, 0.0  
  
    def draw_skeleton_connections(self, frame, keypoints, width, height):  
        """Draw skeleton connections on frame"""  
        skeleton = [  
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],  
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],  
            [2, 4], [3, 5], [4, 6], [5, 7]  
        ]  
          
        for connection in skeleton:  
            kpt_a, kpt_b = connection  
            x1, y1, conf1 = keypoints[kpt_a - 1]  
            x2, y2, conf2 = keypoints[kpt_b - 1]  
            if conf1 > 0.3 and conf2 > 0.3:  
                x1_scaled = int(x1 * width / 640)  
                y1_scaled = int(y1 * height / 640)  
                x2_scaled = int(x2 * width / 640)  
                y2_scaled = int(y2 * height / 640)  
                cv2.line(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 255), 2)  
          
        for i, (x, y, conf) in enumerate(keypoints):  
            if conf > 0.3:  
                x_scaled = int(x * width / 640)  
                y_scaled = int(y * height / 640)  
                color = (0, 255, 0) if i in [5, 6, 11, 12] else (255, 0, 0)   
                cv2.circle(frame, (x_scaled, y_scaled), 4, color, -1)  
  
    def run_video_with_recording(self, video_path, output_path):  
        """Process video and save output with multimodal predictions"""  
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
  
            if pose_valid:  
                self.pose_buffer.append(pose_data)  
                self.visual_buffer.append(self.extract_visual_features(frame, bbox_data))  
                self.bbox_buffer.append(bbox_data)  
                self.scene_buffer.append(self.extract_scene_features(frame))  
            else:  
                self.pose_buffer.append(np.zeros([17, 3]))  
                self.visual_buffer.append(np.zeros(512, dtype=np.float32))  
                self.bbox_buffer.append(np.zeros(4))  
                self.scene_buffer.append(np.zeros(512, dtype=np.float32))  
  
            # Predict when buffer is full  
            if len(self.pose_buffer) == self.sequence_length:  
                predicted_class, confidence = self.predict_action_multimodal(  
                    list(self.pose_buffer),   
                    list(self.visual_buffer),  
                    list(self.bbox_buffer),  
                    list(self.scene_buffer)  
                )  
  
                if predicted_class is not None and not np.isnan(confidence):  
                    last_prediction = f"Action: {self.class_names[predicted_class]}"  
                    last_confidence = f"Confidence: {confidence:.3f}"  
                else:  
                    last_prediction = "No valid prediction"  
                    last_confidence = "Confidence: N/A"  
  
            # Draw visualization  
            if pose_valid and bbox_data.sum() > 0:  
                bbox_scaled = bbox_data * [width/640, height/640, width/640, height/640]  
                xmin, ymin, xmax, ymax = bbox_scaled.astype(int)  
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  
                  
                if results[0].keypoints is not None:  
                    self.draw_skeleton_connections(frame, results[0].keypoints.data[0].cpu().numpy(), width, height)  
  
            # Draw prediction text  
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

    model_path = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/runs/exp8/best.pt'  
    pose_model_path = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/result_model/yolov7-pose/yolo11n-pose.pt'  
  
    output_video = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/output/video_out_5.mp4'  
    os.makedirs(os.path.dirname(output_video), exist_ok=True)  
  
    input_video = '/mnt/myhdd/anhnv/intrustion_ditection/data/Train/extracted_images/intrusion/facility-filming/rgb/intrusion_facility-filming_rgb_0065_cctv1/intrusion_facility-filming_rgb_0065_cctv1.mp4'  
      
    recognizer = RealTimeActionRecognition(model_path, pose_model_path)  
    recognizer.run_video_with_recording(input_video, output_video)  
  
if __name__ == '__main__':  
    main()  