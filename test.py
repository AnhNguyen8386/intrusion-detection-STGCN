import os  
import pandas as pd  
import numpy as np  
import pickle  
import matplotlib.pyplot as plt  
import seaborn as sns  
from pathlib import Path  
import warnings  
warnings.filterwarnings('ignore')  
  
class DatasetPipelineValidator:  
    """  
    Comprehensive validation script cho dataset creation pipeline  
    Kiểm tra consistency qua create_dataset_1.py, create_dataset_2.py, create_dataset_3.py  
    """  
      
    def __init__(self):  
        # Paths cho các files output từ pipeline  
        self.annot_file_1 = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/annotation_4classes_multimodal.csv'  
        self.pose_file_2 = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/pose_bbox_img_features_5classes.csv'  
        self.train_pkl_3 = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/train_5classes_multimodal.pkl'  
        self.test_pkl_3 = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/test_5classes_multimodal.pkl'  
          
        # Class names cho 5 classes  
        self.class_names = ['Vandalism', 'LookingAround', 'UnauthorizedFilming', 'ClimbOverFence', 'Normal']  
          
        # Expected column structures  
        self.expected_cols_1 = ['video', 'frame', 'label']  
        self.expected_cols_2 = 572  # video(1) + frame(1) + pose(51) + bbox(6) + img(512) + label(1)  
          
        # Main body parts indices như trong create_dataset_3.py  
        self.main_idx_parts = [5, 6, 11, 12, 13, 14]  
          
        self.validation_results = {}  
          
    def validate_file_existence(self):  
        """Kiểm tra tất cả files tồn tại"""  
        print("=" * 60)  
        print("1. FILE EXISTENCE VALIDATION")  
        print("=" * 60)  
          
        files_to_check = {  
            'Script 1 Output': self.annot_file_1,  
            'Script 2 Output': self.pose_file_2,  
            'Script 3 Train': self.train_pkl_3,  
            'Script 3 Test': self.test_pkl_3  
        }  
          
        all_exist = True  
        for name, path in files_to_check.items():  
            exists = os.path.exists(path)  
            status = "✓ EXISTS" if exists else "✗ MISSING"  
            print(f"{name:<20}: {status}")  
            if not exists:  
                all_exist = False  
                  
        self.validation_results['files_exist'] = all_exist  
        return all_exist  
          
    def validate_script1_output(self):  
        """Validate output từ create_dataset_1.py"""  
        print("\n" + "=" * 60)  
        print("2. SCRIPT 1 OUTPUT VALIDATION")  
        print("=" * 60)  
          
        try:  
            df1 = pd.read_csv(self.annot_file_1)  
            print(f"✓ Loaded annotation file: {len(df1)} samples")  
              
            # Check column structure  
            cols_match = list(df1.columns) == self.expected_cols_1  
            print(f"Column structure: {'✓ CORRECT' if cols_match else '✗ INCORRECT'}")  
            if not cols_match:  
                print(f"  Expected: {self.expected_cols_1}")  
                print(f"  Actual: {list(df1.columns)}")  
                  
            # Check class distribution  
            print("\nClass distribution in Script 1:")  
            class_dist_1 = df1['label'].value_counts().sort_index()  
            for class_id, count in class_dist_1.items():  
                if class_id < len(self.class_names):  
                    print(f"  Class {class_id} ({self.class_names[class_id]}): {count}")  
                      
            # Check for missing values  
            missing_values = df1.isnull().sum().sum()  
            print(f"Missing values: {missing_values}")  
              
            # Check unique videos  
            unique_videos_1 = len(df1['video'].unique())  
            print(f"Unique videos: {unique_videos_1}")  
              
            self.validation_results['script1'] = {  
                'samples': len(df1),  
                'columns_correct': cols_match,  
                'class_distribution': class_dist_1.to_dict(),  
                'unique_videos': unique_videos_1,  
                'missing_values': missing_values  
            }  
              
            return df1  
              
        except Exception as e:  
            print(f"✗ Error loading Script 1 output: {e}")  
            self.validation_results['script1'] = {'error': str(e)}  
            return None  
              
    def validate_script2_output(self, df1):  
        """Validate output từ create_dataset_2.py"""  
        print("\n" + "=" * 60)  
        print("3. SCRIPT 2 OUTPUT VALIDATION")  
        print("=" * 60)  
          
        try:  
            df2 = pd.read_csv(self.pose_file_2)  
            print(f"✓ Loaded pose features file: {len(df2)} samples")  
              
            # Check column count  
            cols_correct = len(df2.columns) == self.expected_cols_2  
            print(f"Column count: {'✓ CORRECT' if cols_correct else '✗ INCORRECT'}")  
            print(f"  Expected: {self.expected_cols_2}, Actual: {len(df2.columns)}")  
              
            # Check video overlap với Script 1  
            if df1 is not None:  
                videos_1 = set(df1['video'].unique())  
                videos_2 = set(df2['video'].unique())  
                video_overlap = len(videos_1.intersection(videos_2))  
                total_videos_1 = len(videos_1)  
                  
                print(f"\nVideo consistency with Script 1:")  
                print(f"  Videos in Script 1: {total_videos_1}")  
                print(f"  Videos in Script 2: {len(videos_2)}")  
                print(f"  Overlap: {video_overlap}")  
                print(f"  Coverage: {video_overlap/total_videos_1*100:.1f}%")  
                  
            # Check class distribution  
            print("\nClass distribution in Script 2:")  
            class_dist_2 = df2['label'].value_counts().sort_index()  
            for class_id, count in class_dist_2.items():  
                if class_id < len(self.class_names):  
                    print(f"  Class {class_id} ({self.class_names[class_id]}): {count}")  
                      
            # Check pose data quality  
            pose_cols = df2.columns[2:53]  # Pose columns  
            pose_data = df2[pose_cols]  
              
            # Count valid poses (non-NaN main body parts)  
            main_parts_cols = ([f'left_shoulder_{coord}' for coord in ['x', 'y', 's']] +  
                             [f'right_shoulder_{coord}' for coord in ['x', 'y', 's']] +   
                             [f'left_hip_{coord}' for coord in ['x', 'y', 's']] +  
                             [f'right_hip_{coord}' for coord in ['x', 'y', 's']] +  
                             [f'left_knee_{coord}' for coord in ['x', 'y', 's']] +  
                             [f'right_knee_{coord}' for coord in ['x', 'y', 's']])  
              
            valid_poses = 0  
            for _, row in df2.iterrows():  
                main_parts_valid = all(not pd.isna(row[col]) for col in main_parts_cols if col in df2.columns)  
                if main_parts_valid:  
                    valid_poses += 1  
                      
            pose_success_rate = valid_poses / len(df2) * 100  
            print(f"\nPose detection quality:")  
            print(f"  Valid poses: {valid_poses}/{len(df2)} ({pose_success_rate:.1f}%)")  
              
            # Check image features  
            img_cols = df2.columns[59:571]  # Image feature columns  
            img_features_valid = (df2[img_cols] != 0).any(axis=1).sum()  
            img_success_rate = img_features_valid / len(df2) * 100  
            print(f"  Image features extracted: {img_features_valid}/{len(df2)} ({img_success_rate:.1f}%)")  
              
            self.validation_results['script2'] = {  
                'samples': len(df2),  
                'columns_correct': cols_correct,  
                'class_distribution': class_dist_2.to_dict(),  
                'pose_success_rate': pose_success_rate,  
                'image_success_rate': img_success_rate,  
                'video_overlap': video_overlap if df1 is not None else 0  
            }  
              
            return df2  
              
        except Exception as e:  
            print(f"✗ Error loading Script 2 output: {e}")  
            self.validation_results['script2'] = {'error': str(e)}  
            return None  
              
    def validate_script3_output(self, df2):  
        """Validate output từ create_dataset_3.py"""  
        print("\n" + "=" * 60)  
        print("4. SCRIPT 3 OUTPUT VALIDATION")  
        print("=" * 60)  
          
        try:  
            # Load training data  
            with open(self.train_pkl_3, 'rb') as f:  
                train_data = pickle.load(f)  
                  
            # Load testing data  
            with open(self.test_pkl_3, 'rb') as f:  
                test_data = pickle.load(f)  
                  
            print(f"✓ Loaded pickle files successfully")  
              
            # Check data structure  
            if len(train_data) == 4:  
                X_train, y_train, bbox_train, img_train = train_data  
                X_test, y_test, bbox_test, img_test = test_data  
                multimodal = True  
                print("✓ Multimodal data structure detected")  
            else:  
                X_train, y_train = train_data  
                X_test, y_test = test_data  
                multimodal = False  
                print("✓ Standard data structure detected")  
                  
            # Validate shapes  
            print(f"\nData shapes:")  
            print(f"  Training pose: {X_train.shape}")  
            print(f"  Training labels: {y_train.shape}")  
            if multimodal:  
                print(f"  Training bbox: {bbox_train.shape}")  
                print(f"  Training image: {img_train.shape}")  
                  
            print(f"  Testing pose: {X_test.shape}")  
            print(f"  Testing labels: {y_test.shape}")  
            if multimodal:  
                print(f"  Testing bbox: {bbox_test.shape}")  
                print(f"  Testing image: {img_test.shape}")  
                  
            # Check expected dimensions  
            expected_pose_shape = (None, 30, 17, 3)  # (samples, frames, joints, coords)  
            pose_shape_correct = (X_train.ndim == 4 and X_train.shape[1:] == (30, 17, 3))  
            print(f"\nShape validation:")  
            print(f"  Pose shape correct: {'✓' if pose_shape_correct else '✗'}")  
              
            if multimodal:  
                bbox_shape_correct = (bbox_train.ndim == 3 and bbox_train.shape[1:] == (30, 6))  
                img_shape_correct = (img_train.ndim == 3 and img_train.shape[1:] == (30, 512))  
                print(f"  Bbox shape correct: {'✓' if bbox_shape_correct else '✗'}")  
                print(f"  Image shape correct: {'✓' if img_shape_correct else '✗'}")  
                  
            # Check class distribution  
            train_labels_idx = np.argmax(y_train, axis=1)  
            test_labels_idx = np.argmax(y_test, axis=1)  
              
            print(f"\nFinal class distribution:")  
            print("Training set:")  
            for class_id in range(len(self.class_names)):  
                count = np.sum(train_labels_idx == class_id)  
                percentage = count / len(train_labels_idx) * 100  
                print(f"  Class {class_id} ({self.class_names[class_id]}): {count} ({percentage:.1f}%)")  
                  
            print("Testing set:")  
            for class_id in range(len(self.class_names)):  
                count = np.sum(test_labels_idx == class_id)  
                percentage = count / len(test_labels_idx) * 100  
                print(f"  Class {class_id} ({self.class_names[class_id]}): {count} ({percentage:.1f}%)")  
                  
            # Check data ranges  
            print(f"\nData range validation:")  
            pose_min, pose_max = X_train.min(), X_train.max()  
            print(f"  Pose data range: [{pose_min:.3f}, {pose_max:.3f}]")  
              
            if multimodal:  
                bbox_min, bbox_max = bbox_train.min(), bbox_train.max()  
                img_min, img_max = img_train.min(), img_train.max()  
                print(f"  Bbox data range: [{bbox_min:.3f}, {bbox_max:.3f}]")  
                print(f"  Image data range: [{img_min:.3f}, {img_max:.3f}]")  
                  
            # Sample consistency check với Script 2  
            total_samples_script3 = len(X_train) + len(X_test)  
            if df2 is not None:  
                samples_ratio = total_samples_script3 / len(df2)  
                print(f"\nSample consistency:")  
                print(f"  Script 2 samples: {len(df2)}")  
                print(f"  Script 3 samples: {total_samples_script3}")  
                print(f"  Ratio: {samples_ratio:.3f}")  
                  
            self.validation_results['script3'] = {  
                'train_samples': len(X_train),  
                'test_samples': len(X_test),  
                'multimodal': multimodal,  
                'pose_shape_correct': pose_shape_correct,  
                'pose_range': (float(pose_min), float(pose_max)),  
                'train_class_dist': {i: int(np.sum(train_labels_idx == i)) for i in range(len(self.class_names))},  
                'test_class_dist': {i: int(np.sum(test_labels_idx == i)) for i in range(len(self.class_names))}  
            }  
              
            return True  
              
        except Exception as e:  
            print(f"✗ Error loading Script 3 output: {e}")  
            self.validation_results['script3'] = {'error': str(e)}  
            return False  
              
    def generate_visualization_report(self):  
        """Generate visualization report cho validation results"""  
        print("\n" + "=" * 60)  
        print("5. VISUALIZATION REPORT")  
        print("=" * 60)  
          
        try:  
            # Create visualization directory  
            viz_dir = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/validation_report'  
            os.makedirs(viz_dir, exist_ok=True)  
              
            # Plot class distribution comparison  
            if 'script1' in self.validation_results and 'script2' in self.validation_results and 'script3' in self.validation_results:  
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))  
                fig.suptitle('Dataset Pipeline Class Distribution Comparison', fontsize=16)  
                  
                # Script 1 distribution  
                if 'class_distribution' in self.validation_results['script1']:  
                    dist1 = self.validation_results['script1']['class_distribution']  
                    classes = list(range(len(self.class_names)))  
                    counts1 = [dist1.get(i, 0) for i in classes]  
                      
                    axes[0,0].bar(classes, counts1, color='skyblue')  
                    axes[0,0].set_title('Script 1: Annotation Creation')  
                    axes[0,0].set_xlabel('Class ID')  
                    axes[0,0].set_ylabel('Sample Count')  
                    axes[0,0].set_xticks(classes)  
                    axes[0,0].set_xticklabels([f'{i}\n{self.class_names[i][:8]}' for i in classes], rotation=45)  
                  
                # Script 2 distribution  
                if 'class_distribution' in self.validation_results['script2']:  
                    dist2 = self.validation_results['script2']['class_distribution']  
                    counts2 = [dist2.get(i, 0) for i in classes]  
                      
                    axes[0,1].bar(classes, counts2, color='lightgreen')  
                    axes[0,1].set_title('Script 2: Feature Extraction')  
                    axes[0,1].set_xlabel('Class ID')  
                    axes[0,1].set_ylabel('Sample Count')  
                    axes[0,1].set_xticks(classes)  
                    axes[0,1].set_xticklabels([f'{i}\n{self.class_names[i][:8]}' for i in classes], rotation=45)  
                  
                # Script 3 train distribution  
                if 'train_class_dist' in self.validation_results['script3']:  
                    dist3_train = self.validation_results['script3']['train_class_dist']  
                    counts3_train = [dist3_train.get(i, 0) for i in classes]  
                      
                    axes[1,0].bar(classes, counts3_train, color='orange')  
                    axes[1,0].set_title('Script 3: Training Set')  
                    axes[1,0].set_xlabel('Class ID')  
                    axes[1,0].set_ylabel('Sample Count')  
                    axes[1,0].set_xticks(classes)  
                    axes[1,0].set_xticklabels([f'{i}\n{self.class_names[i][:8]}' for i in classes], rotation=45)  
                  
                # Script 3 test distribution  
                if 'test_class_dist' in self.validation_results['script3']:  
                    dist3_test = self.validation_results['script3']['test_class_dist']  
                    counts3_test = [dist3_test.get(i, 0) for i in classes]  
                      
                    axes[1,1].bar(classes, counts3_test, color='salmon')  
                    axes[1,1].set_title('Script 3: Testing Set')  
                    axes[1,1].set_xlabel('Class ID')  
                    axes[1,1].set_ylabel('Sample Count')  
                    axes[1,1].set_xticks(classes)  
                    axes[1,1].set_xticklabels([f'{i}\n{self.class_names[i][:8]}' for i in classes], rotation=45)  
                  
                plt.tight_layout()  
                plt.savefig(os.path.join(viz_dir, 'class_distribution_comparison.png'), dpi=300, bbox_inches='tight')  
                print(f"✓ Class distribution visualization saved to {viz_dir}/class_distribution_comparison.png")  
                  
        except Exception as e:  
            print(f"✗ Error generating visualization: {e}")  
              
    def generate_summary_report(self):  
        """Generate comprehensive summary report"""  
        print("\n" + "=" * 60)  
        print("6. COMPREHENSIVE SUMMARY REPORT")  
        print("=" * 60)  
          
        # Overall pipeline health  
        pipeline_health = "HEALTHY" if self.validation_results.get('files_exist', False) else "UNHEALTHY"  
        print(f"Pipeline Health: {pipeline_health}")  
          
        # File existence summary  
        print(f"\nFile Existence: {'✓ ALL FILES EXIST' if self.validation_results.get('files_exist', False) else '✗ MISSING FILES'}")  
          
        # Script-by-script summary  
        for script_num, script_key in enumerate(['script1', 'script2', 'script3'], 1):  
            if script_key in self.validation_results:  
                script_data = self.validation_results[script_key]  
                if 'error' in script_data:  
                    print(f"\nScript {script_num}: ✗ ERROR - {script_data['error']}")  
                else:  
                    print(f"\nScript {script_num}: ✓ SUCCESS")  
                    if 'samples' in script_data:  
                        print(f"  Samples: {script_data['samples']:,}")  
                    if 'unique_videos' in script_data:  
                        print(f"  Videos: {script_data['unique_videos']}")  
                    if 'pose_success_rate' in script_data:  
                        print(f"  Pose Success Rate: {script_data['pose_success_rate']:.1f}%")  
                    if 'image_success_rate' in script_data:  
                        print(f"  Image Success Rate: {script_data['image_success_rate']:.1f}%")  
          
        # Data consistency warnings  
        print(f"\nData Consistency Warnings:")  
        warnings_found = False  
          
        if 'script2' in self.validation_results and 'video_overlap' in self.validation_results['script2']:  
            overlap = self.validation_results['script2']['video_overlap']  
            if overlap < 100:  # Assuming we expect 100% overlap  
                print(f"  ⚠ Video overlap between Script 1 and 2: {overlap}% (expected 100%)")  
                warnings_found = True  
                  
        if 'script3' in self.validation_results:  
            script3_data = self.validation_results['script3']  
            if not script3_data.get('pose_shape_correct', True):  
                print(f"  ⚠ Pose data shape incorrect in Script 3")  
                warnings_found = True  
                  
        if not warnings_found:  
            print("  ✓ No consistency warnings detected")  
              
        # Recommendations  
        print(f"\nRecommendations:")  
        if pipeline_health == "HEALTHY":  
            print("  ✓ Pipeline appears healthy - proceed with training")  
            print("  ✓ Monitor class balance during training")  
            print("  ✓ Verify model performance on validation set")  
        else:  
            print("  ✗ Fix missing files before proceeding")  
            print("  ✗ Re-run failed scripts")  
            print("  ✗ Check data paths and permissions")  
              
    def run_full_validation(self):  
        """Run complete validation pipeline"""  
        print("DATASET PIPELINE VALIDATION")  
        print("=" * 80)  
        print("Validating consistency across create_dataset_1.py, create_dataset_2.py, create_dataset_3.py")  
        print("=" * 80)  
          
        # Step 1: Check file existence  
        if not self.validate_file_existence():  
            print("\n[CRITICAL] Missing files detected. Cannot proceed with validation.")  
            return False  
              
        # Step 2: Validate Script 1 output  
        df1 = self.validate_script1_output()  
          
        # Step 3: Validate Script 2 output  
        df2 = self.validate_script2_output(df1)  
          
        # Step 4: Validate Script 3 output  
        script3_success = self.validate_script3_output(df2)  
          
        # Step 5: Generate visualizations  
        self.generate_visualization_report()  
          
        # Step 6: Generate summary  
        self.generate_summary_report()  
          
        return script3_success  
  
  
# Main execution function  
def main():  
    """Main function để chạy validation"""  
    validator = DatasetPipelineValidator()  
      
    print("Starting comprehensive dataset pipeline validation...")  
    print("This will check consistency across all three dataset creation scripts.")  
    print()  
      
    success = validator.run_full_validation()  
      
    print("\n" + "=" * 80)  
    if success:  
        print("VALIDATION COMPLETED SUCCESSFULLY!")  
        print("Dataset pipeline appears consistent and ready for training.")  
    else:  
        print("VALIDATION COMPLETED WITH ISSUES!")  
        print("Please review the errors above and fix before proceeding.")  
    print("=" * 80)  
      
    return success  
  
  
if __name__ == '__main__':  
    main()  