import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from models.stgcn import TwoStreamSpatialTemporalGraph
from dataloader.dataset import processing_data
import os


class RealTimeActionRecognition:
    def __init__(self, model_path, pose_model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load ST-GCN model  
        self.class_names = ['Vandalism', 'LookingAround', 'UnauthorizedFilming', 'ClimbOverFence']
        graph_args = {'strategy': 'spatial'}
        self.action_model = TwoStreamSpatialTemporalGraph(graph_args, len(self.class_names)).to(self.device)

        # Load state dict and handle DataParallel  
        state_dict = torch.load(model_path, map_location=self.device)

        # Remove 'module.' prefix if present (from DataParallel)  
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key[7:]  # Remove 'module.' prefix  
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        self.action_model.load_state_dict(state_dict)
        self.action_model.eval()

        # Load YOLOv8 pose model  
        self.pose_model = YOLO(pose_model_path)

        # Frame buffer for 30-frame sequences  
        self.frame_buffer = deque(maxlen=30)
        self.frame_size = [640, 640]

        # COCO skeleton connections for drawing  
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]

    def normalize_points_with_size(self, points_xy, width, height):
        """Normalize keypoints to [0,1] range"""
        points_xy[:, 0] /= width
        points_xy[:, 1] /= height
        return points_xy

    def process_yolov8_results(self, results, thresh=0.001):
        """Process YOLOv8 pose results"""
        if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints.data) == 0:
            return np.zeros([17, 3]), False

            # Get first detection  
        keypoints = results[0].keypoints.data[0].cpu().numpy()  # Shape: (17, 3)  

        # Convert to expected format  
        result = np.zeros([17, 3])
        cf = True

        for i in range(17):
            if i < len(keypoints):
                x, y, conf = keypoints[i]
                # Check if coordinates are valid  
                if not (x % 640 == 0 or y % 640 == 0):
                    result[i, 0] = x
                    result[i, 1] = y
                    result[i, 2] = conf

                    # Check confidence for main body parts  
                    if (conf < thresh) and (i in [5, 6, 11, 12, 13, 14]):
                        cf = False

        return result, cf

    def draw_skeleton_connections(self, frame, keypoints, width, height):
        """Draw skeleton connections between keypoints"""
        for connection in self.skeleton:
            kpt_a, kpt_b = connection
            if kpt_a - 1 < len(keypoints) and kpt_b - 1 < len(keypoints):
                x1, y1, conf1 = keypoints[kpt_a - 1]
                x2, y2, conf2 = keypoints[kpt_b - 1]

                if conf1 > 0.3 and conf2 > 0.3:
                    x1_scaled = int(x1 * width / 640)
                    y1_scaled = int(y1 * height / 640)
                    x2_scaled = int(x2 * width / 640)
                    y2_scaled = int(y2 * height / 640)

                    cv2.line(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled),
                             (0, 255, 255), 2)

    def predict_action(self, pose_sequence):
        """Predict action from pose sequence - FIXED normalization"""
        if len(pose_sequence) < 30:
            return None, 0.0

        try:
            # Convert to numpy array and process  
            features = np.array(pose_sequence)  # Shape: (30, 17, 3)  

            # Frame sampling: 30 -> 15 frames  
            features = features[::2, :, :]  # Shape: (15, 17, 3)  

            # FIXED: Use correct normalization logic from create_dataset_3.py  
            def scale_pose_fixed(xy):
                if xy.ndim == 2:
                    xy = np.expand_dims(xy, 0)
                xy_min = np.nanmin(xy, axis=1)  # Correct axis  
                xy_max = np.nanmax(xy, axis=1)  # Correct axis  
                for i in range(xy.shape[0]):
                    xy_range = xy_max[i] - xy_min[i]
                    # Prevent division by zero  
                    xy_range = np.where((xy_range == 0) | np.isnan(xy_range), 1, xy_range)
                    xy[i] = np.nan_to_num(((xy[i] - xy_min[i]) / xy_range) * 2 - 1, nan=0.0)
                return xy.squeeze()

                # Apply fixed normalization  

            features[:, :, :2] = scale_pose_fixed(features[:, :, :2])

            # Convert to tensor  
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(
                self.device)

            # Calculate motion  
            mot = features_tensor[:, :2, 1:, :] - features_tensor[:, :2, :-1, :]

            # Predict  
            with torch.no_grad():
                outputs = self.action_model((features_tensor, mot))

                # Check for NaN values  
                if torch.isnan(outputs).any():
                    print("Warning: NaN detected in model outputs")
                    return None, 0.0

                    # Use argmax directly on sigmoid outputs  
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = outputs[0][predicted_class].item()

                # Clamp confidence to valid range  
                confidence = max(0.0, min(1.0, confidence))

                return predicted_class, confidence

        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, 0.0

    def run_video_with_recording(self, video_path, output_path):
        """Run action recognition and save output video"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Cannot open video source")
            return

            # Get video properties  
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer  
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {video_path}")
        print(f"Output will be saved to: {output_path}")
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Resize frame for pose detection  
            frame_resized = cv2.resize(frame, self.frame_size)

            # Extract pose  
            results = self.pose_model(frame_resized, verbose=False)
            pose_data, pose_valid = self.process_yolov8_results(results)

            if pose_valid:
                # Normalize keypoints  
                pose_normalized = self.normalize_points_with_size(
                    pose_data.copy(), self.frame_size[0], self.frame_size[1]
                )
                self.frame_buffer.append(pose_normalized)

                # Draw keypoints và skeleton trên original frame  
                if len(results) > 0 and results[0].keypoints is not None:
                    keypoints = results[0].keypoints.data[0].cpu().numpy()

                    # Vẽ skeleton connections trước  
                    self.draw_skeleton_connections(frame, keypoints, width, height)

                    # Vẽ keypoints (không hiển thị số thứ tự)  
                    for i, (x, y, conf) in enumerate(keypoints):
                        if conf > 0.3:
                            # Scale keypoints back to original frame size  
                            x_scaled = int(x * width / 640)
                            y_scaled = int(y * height / 640)

                            # Vẽ keypoint với màu khác nhau cho các body parts  
                            if i in [5, 6, 11, 12, 13, 14]:  # Main body parts  
                                color = (0, 255, 0)  # Green cho main parts  
                                radius = 6
                            else:
                                color = (255, 0, 0)  # Blue cho các parts khác  
                                radius = 4

                            cv2.circle(frame, (x_scaled, y_scaled), radius, color, -1)

                            # Predict action if we have enough frames  
            action_text = "Collecting frames..."
            confidence_text = ""

            if len(self.frame_buffer) == 30:
                predicted_class, confidence = self.predict_action(list(self.frame_buffer))
                if predicted_class is not None and not np.isnan(confidence):
                    action_text = f"Action: {self.class_names[predicted_class]}"
                    confidence_text = f"Confidence: {confidence:.3f}"
                else:
                    action_text = "No valid prediction"
                    confidence_text = "Confidence: N/A"

                    # Create overlay background for better text visibility  
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (650, 120), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

            # Display results on frame  
            cv2.putText(frame, action_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, confidence_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Add frame counter  
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (width - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Write frame to output video  
            out.write(frame)

            # Progress indicator  
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps_processed = frame_count / elapsed_time
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                      f"Processing speed: {fps_processed:.1f} FPS")

                # Cleanup  
        cap.release()
        out.release()

        total_time = time.time() - start_time
        print(f"Video processing completed!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average processing speed: {frame_count / total_time:.1f} FPS")
        print(f"Output saved to: {output_path}")


def main():
    # Paths to your trained models  
    model_path = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/runs/exp10/best.pt'   
    pose_model_path = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/result_model/yolov7-pose/yolov8x-pose.pt'

    # Input and output paths  
    input_video = '/mnt/myhdd/anhnv/intrustion_ditection/data/valid/extracted_images/intrusion/damage-to-facilities/rgb/intrusion_damage-to-facilities_rgb_0014_cctv2/intrusion_damage-to-facilities_rgb_0014_cctv2.mp4'    
    output_video = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/output/video_out.mp4'    

    # Create output directory if it doesn't exist  
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    # Initialize recognizer  
    recognizer = RealTimeActionRecognition(model_path, pose_model_path)

    # Process video and save with predictions  
    recognizer.run_video_with_recording(input_video, output_video)


if __name__ == '__main__':
    main()