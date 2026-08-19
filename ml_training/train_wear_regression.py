import os
import time
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# ==========================================
# Tire Vision - Wear Regression Model
# Trains a ResNet50 model to predict tire 
# mileage/wear based on tread images.
# ==========================================

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
# Adjust path to match the actual dataset location
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Updated Tyre Dataset"))
MODEL_SAVE_PATH = "models/wear_regression_resnet50.pth"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Training on device: {device}")

# Data Augmentation & Normalization
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class TireWearDataset(Dataset):
    """
    Custom Dataset for loading tire images and their associated mileage.
    Mileage is parsed from the folder name (e.g., '10000 km' -> 10000.0).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        if os.path.exists(root_dir):
            for folder in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder)
                if os.path.isdir(folder_path):
                    try:
                        # Extract mileage from folder name, e.g., "10000 km" -> 10000.0
                        mileage_str = folder.lower().replace("km", "").strip()
                        mileage = float(mileage_str)
                        
                        for img_name in os.listdir(folder_path):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.samples.append((os.path.join(folder_path, img_name), mileage))
                    except ValueError:
                        print(f"[!] Could not parse mileage from folder: {folder}")
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, mileage = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Target should be a float tensor for MSELoss
        target = torch.tensor([mileage], dtype=torch.float32)
        return image, target

def load_data():
    print(f"[*] Loading dataset from {DATA_DIR}...")
    dataset = TireWearDataset(DATA_DIR, transform=data_transforms)
    
    if len(dataset) == 0:
        print("[!] No valid samples found. Check dataset path.")
        return None, None
        
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    
    print(f"[*] Train size: {dataset_sizes['train']} | Val size: {dataset_sizes['val']}")
    return dataloaders, dataset_sizes

def build_model():
    print("[*] Initializing ResNet50 backbone for regression...")
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the classification head with a regression head
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1) # Single output for mileage prediction
    )
    return model.to(device)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f"{phase.capitalize()} Loss (MSE): {epoch_loss:.4f} | RMSE: {epoch_loss**0.5:.2f} km")

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"[*] Best model saved with Val RMSE: {best_loss**0.5:.2f} km")

    time_elapsed = time.time() - start_time
    print(f"\n[*] Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"[*] Best Val RMSE: {best_loss**0.5:.2f} km")
    return model

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"[!] Dataset not found at {DATA_DIR}. Please check the path.")
    else:
        dataloaders, dataset_sizes = load_data()
        if dataloaders:
            model = build_model()
            
            # Using Mean Squared Error for regression
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
            trained_model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, EPOCHS)
