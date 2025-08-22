import os  
import time  
import torch  
import numpy as np  
import torch.nn.functional as F  
import matplotlib.pyplot as plt  
import matplotlib  
matplotlib.use('Agg')  
from tqdm import tqdm  
from torch.utils.data import DataLoader, TensorDataset  
from collections import OrderedDict  
import logging  
import yaml  
from dataloader.dataset import processing_data  
import datetime  
import gc  
import sys  

torch.cuda.empty_cache()  
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False   
gc.collect()  
modules_to_clear = [k for k in sys.modules.keys() if k.startswith('models')]  
for module in modules_to_clear:  
    if module in sys.modules:  
        del sys.modules[module]  
  
torch.manual_seed(42)  
np.random.seed(42)  
if torch.cuda.is_available():  
    torch.cuda.manual_seed_all(42)  
  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
print(f"Using device: {device}")  

log_file = f'training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',  
                    handlers=[  
                        logging.FileHandler(log_file),  
                        logging.StreamHandler()  
                    ])  

try:  
    with open("/mnt/myhdd/anhnv/intrustion_ditection/ST-GCN-Pytorch/config.yaml", "r") as stream:  
        config_data = yaml.safe_load(stream)  
    logging.info("Đã tải config.yaml thành công")  
except FileNotFoundError:  
    logging.error("Lỗi: Không tìm thấy file")  
    exit()  
input_dataset_train = config_data['dataset-path-train']  
input_dataset_test = config_data['dataset-path-test']  
epochs = config_data['epochs']  
batch_size = config_data['batch-size']  
input_size = config_data['img-size']  
num_frame = config_data['num-frame']  
path_save_model = config_data['project']  
  
logging.info(f"Configuration loaded from config.yaml:")  
logging.info(f"  Train dataset: {input_dataset_train}")  
logging.info(f"  Test dataset: {input_dataset_test}")  
logging.info(f"  Epochs: {epochs}")  
logging.info(f"  Batch size: {batch_size}")  
logging.info(f"  Num frames: {num_frame}")  
logging.info(f"  Project path: {path_save_model}")  
  
def load_npy_dataset(data_dir):  
    """Load dataset from .npy files in directory"""  
    try:  
        features = np.load(os.path.join(data_dir, 'X.npy'))  
        labels = np.load(os.path.join(data_dir, 'y.npy'))  
        bbox_features = np.load(os.path.join(data_dir, 'bbox.npy'))  
        visual_features = np.load(os.path.join(data_dir, 'visual.npy'))  
        scene_features = np.load(os.path.join(data_dir, 'scene.npy'))  
        flow_features = np.load(os.path.join(data_dir, 'flow.npy'))  
          
        logging.info(f"Loaded data shapes from {data_dir}:")  
        logging.info(f"  Features (X): {features.shape}")  
        logging.info(f"  Labels (y): {labels.shape}")  
        logging.info(f"  Bbox: {bbox_features.shape}")  
        logging.info(f"  Visual: {visual_features.shape}")  
        logging.info(f"  Scene: {scene_features.shape}")  
        logging.info(f"  Flow: {flow_features.shape}")  
          
        return features, labels, bbox_features, visual_features, scene_features, flow_features  
    except FileNotFoundError as e:  
        logging.error(f"Required .npy file not found in {data_dir}: {e}")  
        return None  
  
def label_smoothing(labels, smoothing=0.3):  
    """Apply label smoothing to one-hot labels"""  
    if labels.ndim == 1:  
        num_classes = 4  
        one_hot = np.zeros((len(labels), num_classes))  
        one_hot[np.arange(len(labels)), labels] = 1  
        labels = one_hot  
          
    num_classes = labels.shape[1]  
    return labels * (1 - smoothing) + smoothing / num_classes  
  
def balanced_pose_augmentation(inputs, noise_factor=0.02):  
    """Add balanced augmentation to pose data"""  
    noise = torch.randn_like(inputs) * noise_factor  
    inputs_aug = inputs + noise  
  
    if torch.rand(1) > 0.6:  
        mask_frames = torch.rand(inputs.size(2)) > 0.1  
        inputs_aug[:, :, ~mask_frames, :] = 0  
      
    if torch.rand(1) > 0.7:  
        mask_joints = torch.rand(inputs.size(3)) > 0.05  
        inputs_aug[:, :, :, ~mask_joints] = 0  
      
    return inputs_aug  

logging.info("Loading training dataset from .npy files...")  
train_data = load_npy_dataset(input_dataset_train)  
if train_data is None:  
    logging.error("Failed to load training data")  
    exit()  
  
features, labels, bbox_fts_train, visual_fts_train, scene_fts_train, flow_fts_train = train_data  

if labels.ndim == 1:  
    num_classes = 4  
    labels_one_hot = np.zeros((len(labels), num_classes))  
    labels_one_hot[np.arange(len(labels)), labels] = 1  
    labels = labels_one_hot  

if features.shape[1] == 30:  
    features = features[:, ::2, :, :]  

    if visual_fts_train.shape[1] == 30:  
        visual_fts_train = visual_fts_train[:, ::2, :]  
    if scene_fts_train.shape[1] == 30:  
        scene_fts_train = scene_fts_train[:, ::2, :]  
    if bbox_fts_train.shape[1] == 30:  
        bbox_fts_train = bbox_fts_train[:, ::2, :]  
    logging.info(f"Sampled frames from 30 to 15: {features.shape}")  
  
features[:, :, :, :2] = processing_data(features[:, :, :, :2])  
 
labels = label_smoothing(labels, smoothing=0.3)  
  
x_train = features  
y_train = labels  
bbox_train = bbox_fts_train  
visual_train = visual_fts_train  
scene_train = scene_fts_train  
  
logging.info("--------- Number class train---------")  
for i in range(4):  
    if labels.ndim > 1:  
        count = np.argmax(labels, axis=1).tolist().count(i)  
    else:  
        count = (labels == i).sum()  
    logging.info(f"class {i}: {count}")  

logging.info("Loading test dataset from .npy files...")  
test_data = load_npy_dataset(input_dataset_test)  
if test_data is None:  
    logging.error("Failed to load test data")  
    exit()  
  
features, labels, bbox_fts_test, visual_fts_test, scene_fts_test, flow_fts_test = test_data  

if labels.ndim == 1:  
    num_classes = 4  
    labels_one_hot = np.zeros((len(labels), num_classes))  
    labels_one_hot[np.arange(len(labels)), labels] = 1  
    labels = labels_one_hot  

if features.shape[1] == 30:  
    features = features[:, ::2, :, :]  
    if visual_fts_test.shape[1] == 30:  
        visual_fts_test = visual_fts_test[:, ::2, :]  
    if scene_fts_test.shape[1] == 30:  
        scene_fts_test = scene_fts_test[:, ::2, :]  
    if bbox_fts_test.shape[1] == 30:  
        bbox_fts_test = bbox_fts_test[:, ::2, :]  
 
features[:, :, :, :2] = processing_data(features[:, :, :, :2])   
labels = label_smoothing(labels, smoothing=0.15)    
x_valid = features  
y_valid = labels  
bbox_valid = bbox_fts_test  
visual_valid = visual_fts_test  
scene_valid = scene_fts_test  
  
logging.info("Number class test")  
for i in range(4):  
    if labels.ndim > 1:  
        count = np.argmax(labels, axis=1).tolist().count(i)  
    else:  
        count = (labels == i).sum()  
    logging.info(f"class {i}: {count}")  
  
del features, labels  
logging.info("Creating multimodal tensor datasets...")  
train_dataset = TensorDataset(  
    torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),  # Pose  
    torch.tensor(visual_train, dtype=torch.float32),  # Visual features  
    torch.tensor(bbox_train, dtype=torch.float32),    # Bbox features  
    torch.tensor(scene_train, dtype=torch.float32),   # Scene features  
    torch.tensor(y_train, dtype=torch.float32)        # Labels  
)  
  
val_dataset = TensorDataset(  
    torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),   
    torch.tensor(visual_valid, dtype=torch.float32),   
    torch.tensor(bbox_valid, dtype=torch.float32),      
    torch.tensor(scene_valid, dtype=torch.float32),    
    torch.tensor(y_valid, dtype=torch.float32)        
)  
  
del x_train, x_valid, y_train, y_valid  
del bbox_train, visual_train, scene_train  
del bbox_valid, visual_valid, scene_valid  
logging.info("Multimodal tensor datasets created successfully")  

if not os.path.exists(path_save_model):  
    os.makedirs(path_save_model, exist_ok=True)  
  
count = 0  
while os.path.exists(os.path.join(path_save_model, f'exp{count}')):  
    count += 1  
  
path_save_model = os.path.join(path_save_model, f'exp{count}')  
os.makedirs(path_save_model, exist_ok=True)  
logging.info(f"Experiment folder created: {path_save_model}")  

logging.info("Creating data loaders...")  
train_loader = DataLoader(  
    train_dataset,  
    batch_size=batch_size,  
    shuffle=True,  
    num_workers=0,  
    pin_memory=False  
)  
  
val_loader = DataLoader(  
    val_dataset,  
    batch_size=batch_size,  
    shuffle=False,  
    num_workers=0,  
    pin_memory=False  
)  
  
del train_dataset, val_dataset  
logging.info("Data loaders created successfully")  

from models.stgcn import EnhancedMultimodalGraph  

torch.cuda.empty_cache()  
  
logging.info("Creating EnhancedMultimodalGraph model...")  
graph_args = {'strategy': 'spatial'}  
  

model = EnhancedMultimodalGraph(graph_args, 4, dropout=0.6)  
  
logging.info("Model created on CPU successfully")  
logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")  
model = model.to(device)  
logging.info("Model moved to GPU successfully")  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5, verbose=True)  
losser = torch.nn.BCEWithLogitsLoss()    
  
def train_model(model, losser, optimizer, scheduler, num_epochs):  
    logging.info(f"Starting training for {num_epochs} epochs...")  
    logging.info("Testing forward pass with single batch...")  
    for data in train_loader:  
        pose_inputs, visual_inputs, bbox_inputs, scene_inputs, labels = data  
        pose_inputs = pose_inputs.to(device)  
        visual_inputs = visual_inputs.to(device)  
        bbox_inputs = bbox_inputs.to(device)  
        scene_inputs = scene_inputs.to(device)  
        mot = pose_inputs[:, :2, 1:, :] - pose_inputs[:, :2, :-1, :]  
        padding = torch.zeros(pose_inputs.size(0), 2, 1, pose_inputs.size(3), device=device)  
        mot = torch.cat([padding, mot], dim=2)  
          
        logging.info(f"Debug - Pose shape: {pose_inputs.shape}")  
        logging.info(f"Debug - Motion shape: {mot.shape}")  
        logging.info(f"Debug - Visual shape: {visual_inputs.shape}")  
        logging.info(f"Debug - Bbox shape: {bbox_inputs.shape}")  
        logging.info(f"Debug - Scene shape: {scene_inputs.shape}")  
          
        try:  
            with torch.no_grad():  
                outputs = model([pose_inputs, mot, visual_inputs, bbox_inputs, scene_inputs])  
                logging.info(f"Debug - Output shape: {outputs.shape}")  
                logging.info("Forward pass test successful!")  
        except Exception as e:  
            logging.error(f"Forward pass test failed: {e}", exc_info=True)  
            return None  
        break  
      
    loss_list = {'train': [], 'valid': []}  
    acc_list = {'train': [], 'valid': []}
    best_val_acc = 0.0  
    patience = 8  
    patience_counter = 0  
    best_val_loss = float('inf')  
      
    for epoch in range(num_epochs):  
        logging.info(f"Epoch {epoch+1}/{num_epochs}")  

        model.train()  
        train_loss = 0.0  
        train_corrects = 0  
        train_samples = 0  
          
        for data in tqdm(train_loader, desc='Training'):  
            pose_inputs, visual_inputs, bbox_inputs, scene_inputs, labels = data  
            pose_inputs = pose_inputs.to(device)  
            visual_inputs = visual_inputs.to(device)  
            bbox_inputs = bbox_inputs.to(device)  
            scene_inputs = scene_inputs.to(device)  
            labels = labels.to(device)  
 
            if np.random.random() > 0.5:  
                pose_inputs = balanced_pose_augmentation(pose_inputs)  
            mot = pose_inputs[:, :2, 1:, :] - pose_inputs[:, :2, :-1, :]  
            padding = torch.zeros(pose_inputs.size(0), 2, 1, pose_inputs.size(3), device=device)  
            mot = torch.cat([padding, mot], dim=2)                
            optimizer.zero_grad()  
            outputs = model([pose_inputs, mot, visual_inputs, bbox_inputs, scene_inputs])  
              
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():  
                logging.warning("NaN or Inf detected in outputs. Skipping batch.")  
                continue  
              
            loss = losser(outputs, labels)  
              
            if torch.isnan(loss) or torch.isinf(loss):  
                logging.warning("NaN or Inf detected in loss. Skipping batch.")  
                continue  
              
            loss.backward()  
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  
            optimizer.step()                
            train_loss += loss.item() * labels.size(0)  
            predicted = torch.argmax(torch.sigmoid(outputs), dim=1)  
            true_labels = torch.argmax(labels, dim=1)  
            train_corrects += (predicted == true_labels).float().sum().item()  
            train_samples += labels.size(0)  

        model.eval()  
        val_loss = 0.0  
        val_corrects = 0  
        val_samples = 0  
          
        with torch.no_grad():  
            for data in tqdm(val_loader, desc='Validation'):  
                pose_inputs, visual_inputs, bbox_inputs, scene_inputs, labels = data  
                pose_inputs = pose_inputs.to(device)  
                visual_inputs = visual_inputs.to(device)  
                bbox_inputs = bbox_inputs.to(device)  
                scene_inputs = scene_inputs.to(device)  
                labels = labels.to(device)   
                mot = pose_inputs[:, :2, 1:, :] - pose_inputs[:, :2, :-1, :]  
                padding = torch.zeros(pose_inputs.size(0), 2, 1, pose_inputs.size(3), device=device)  
                mot = torch.cat([padding, mot], dim=2)  
                  
                outputs = model([pose_inputs, mot, visual_inputs, bbox_inputs, scene_inputs])  
                  
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():  
                    continue  
                  
                loss = losser(outputs, labels)  
                  
                if torch.isnan(loss):  
                    continue  
                  
                val_loss += loss.item() * labels.size(0)  
                  
                predicted = torch.argmax(torch.sigmoid(outputs), dim=1)  
                true_labels = torch.argmax(labels, dim=1)  
                val_corrects += (predicted == true_labels).float().sum().item()  
                val_samples += labels.size(0)  

        train_loss = train_loss / len(train_loader.dataset)  
        train_acc = train_corrects / train_samples  
        val_loss = val_loss / len(val_loader.dataset)  
        val_acc = val_corrects / val_samples  
          
        loss_list['train'].append(train_loss)  
        loss_list['valid'].append(val_loss)  
        acc_list['train'].append(train_acc)  
        acc_list['valid'].append(val_acc)  
          
        logging.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")  
        logging.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")  

        if val_acc > best_val_acc:  
            best_val_acc = val_acc  
            try:  
                os.makedirs(path_save_model, exist_ok=True)  
                torch.save(model.state_dict(), os.path.join(path_save_model, 'best.pt'))  
                logging.info(f"  Saved best model (Val Acc: {best_val_acc:.4f})")  
            except Exception as e:  
                logging.error(f"Failed to save best model: {e}")  
          
        # Early stopping logic  
        if val_loss < best_val_loss:  
            best_val_loss = val_loss  
            patience_counter = 0  
        else:  
            patience_counter += 1  
            if patience_counter >= patience:  
                logging.info(f"Early stopping at epoch {epoch+1} due to no improvement")  
                break  

        scheduler.step(val_acc)  

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:  
            fig = plt.figure(figsize=(12, 4))  
              
            plt.subplot(1, 2, 1)  
            plt.plot(acc_list['train'], label="Training Accuracy", linewidth=2)  
            plt.plot(acc_list['valid'], label="Validation Accuracy", linewidth=2)  
            plt.xlabel("Epoch")  
            plt.ylabel("Accuracy")  
            plt.title("Training Progress - Accuracy")  
            plt.legend()  
            plt.grid(True, alpha=0.3)  
              
            plt.subplot(1, 2, 2)  
            plt.plot(loss_list['train'], label="Training Loss", linewidth=2)  
            plt.plot(loss_list['valid'], label="Validation Loss", linewidth=2)  
            plt.xlabel("Epoch")  
            plt.ylabel("Loss")  
            plt.title("Training Progress - Loss")  
            plt.legend()  
            plt.grid(True, alpha=0.3)  
              
            plt.tight_layout()  
            fig.savefig(os.path.join(path_save_model, 'result.png'), dpi=300, bbox_inches='tight')  
            plt.close(fig)  
      
    logging.info(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")  
    return model  
  
def main():  

    model_trained = train_model(model, losser, optimizer, scheduler, num_epochs=epochs)  
    if model_trained is not None:  
        try:  
            os.makedirs(path_save_model, exist_ok=True)  
            torch.save(model_trained.state_dict(), os.path.join(path_save_model, 'last.pt'))  
            logging.info(f'Saved last model at {os.path.join(path_save_model, "last.pt")}')  
            logging.info("Training completed successfully!")  
        except Exception as e:  
            logging.error(f"Failed to save last model: {e}")  
    else:  
        logging.error("Training failed and did not return a model.")  
  
if __name__ == '__main__':  
    main()
      
      