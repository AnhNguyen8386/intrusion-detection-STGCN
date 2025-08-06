import torch  
import pickle  
import os  
from torch.utils.data import DataLoader, TensorDataset  
from models.stgcn import TwoStreamSpatialTemporalGraph 
import numpy as np  
from sklearn import metrics  
from util.plot import plot_cm  
from tqdm import tqdm  
from dataloader.dataset import processing_data  
  
def detect_image(path_test, path_model, batch_size=64):  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
      
    # 5 classes intrusion detection  
    class_names = ['Vandalism', 'LookingAround', 'FenceDamage', 'UnauthorizedFilming', 'ClimbOverFence']  
    num_classes = len(class_names)  
  
    print(f"Loading model from: {path_model}")  
    print(f"Loading test data from: {path_test}")  
      
    # Load model  
    graph_args = {'strategy': 'spatial'}  
    model = TwoStreamSpatialTemporalGraph(graph_args, num_classes).to(device)  
    model.load_state_dict(torch.load(path_model, map_location=device, weights_only=False))  
    model.eval()  
    print("Model loaded successfully")  
  
    # FIXED: Flexible data loading  
    with open(path_test, 'rb') as f:  
        data = pickle.load(f)  
        print(f"Loaded data type: {type(data)}")  
          
        # Handle different data formats  
        if isinstance(data, tuple) and len(data) == 2:  
            features, labels = data  
        elif isinstance(data, list) and len(data) == 2:  
            features, labels = data[0], data[1]  
        elif isinstance(data, dict):  
            # If data is dictionary format  
            if 'features' in data and 'labels' in data:  
                features, labels = data['features'], data['labels']  
            elif 'X' in data and 'y' in data:  
                features, labels = data['X'], data['y']  
            else:  
                print(f"Dictionary keys: {list(data.keys())}")  
                raise ValueError("Unknown dictionary format")  
        elif hasattr(data, '__len__') and len(data) > 2:  
            # If more than 2 elements, take first two  
            print(f"Warning: Found {len(data)} elements, using first 2")  
            features, labels = data[0], data[1]  
        else:  
            print(f"Data structure: {data}")  
            raise ValueError(f"Cannot unpack data of type {type(data)} with length {len(data) if hasattr(data, '__len__') else 'unknown'}")  
      
    print(f"Original data shape: features={features.shape}, labels={labels.shape}")  
      
    # Preprocessing - match training pipeline exactly  
    features = features[:, ::2, :, :]  # Sample to 15 frames  
    features[:, :, :, :2] = processing_data(features[:, :, :, :2])  # Normalize coordinates  
      
    # Convert one-hot to class indices if needed  
    if labels.ndim > 1 and labels.shape[1] > 1:  
        labels = labels.argmax(1)  
    elif labels.ndim > 1 and labels.shape[1] == 1:  
        labels = labels.squeeze()  
      
    print(f"Processed data shape: features={features.shape}, labels={labels.shape}")  
  
    # Class distribution  
    print("\n--------- Class Distribution ---------")  
    for i in range(num_classes):  
        count = np.sum(labels == i)  
        percentage = (count / len(labels)) * 100 if len(labels) > 0 else 0  
        print(f"{class_names[i]}: {count} samples ({percentage:.1f}%)")  
  
    # Create DataLoader  
    test_dataset = TensorDataset(  
        torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),  # (N, C, T, V)  
        torch.tensor(labels, dtype=torch.long)  
    )  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)  
  
    # Evaluation  
    y_true, y_pred = [], []  
    model.eval()  
      
    with torch.no_grad():  
        for x, y in tqdm(test_loader, desc="Evaluating"):  
            # Calculate motion stream  
            mot = x[:, :2, 1:, :] - x[:, :2, :-1, :]  
            x, mot, y = x.to(device), mot.to(device), y.to(device)  
  
            # Model inference  
            outputs = model((x, mot))  
            predictions = outputs.argmax(dim=1)  
  
            y_true.extend(y.cpu().tolist())  
            y_pred.extend(predictions.cpu().tolist())  
  
    # Compute metrics  
    cm = metrics.confusion_matrix(y_true, y_pred)  
    accuracy = metrics.accuracy_score(y_true, y_pred)  
    precision = metrics.precision_score(y_true, y_pred, average=None, zero_division=0)  
    recall = metrics.recall_score(y_true, y_pred, average=None, zero_division=0)  
    f1 = metrics.f1_score(y_true, y_pred, average=None, zero_division=0)  
  
    # Print results  
    print(f"\n{'='*50}")  
    print(f"EVALUATION RESULTS")  
    print(f"{'='*50}")  
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")  
    print(f"\nPer-class Metrics:")  
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")  
    print(f"{'-'*50}")  
      
    for i in range(num_classes):  
        print(f"{class_names[i]:<20} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f}")  
  
    # Save results  
    os.makedirs("info_stgcn", exist_ok=True)  
      
    # Plot confusion matrix  
    plot_cm(cm.T, normalize=False, save_dir='info_stgcn', names_x=class_names, names_y=class_names, show=False)  
    plot_cm(cm.T, normalize=True, save_dir='info_stgcn', names_x=class_names, names_y=class_names, show=False)  
      
    print(f"\nResults saved to info_stgcn/ directory")  
  
if __name__ == '__main__':  
    path_model = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/runs/exp10/best.pt'  
    path_test = '/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/data/test_4classes.pkl'  
      
    print("Starting ST-GCN Intrusion Detection Evaluation...")  
    detect_image(path_test, path_model, batch_size=64)  
    print("Evaluation completed!")