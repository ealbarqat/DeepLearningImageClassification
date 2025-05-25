import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

# Let's ignore that NNPACK warning - it's not important for our work
warnings.filterwarnings("ignore", message="Could not initialize NNPACK")

# This class helps us load our brain images and their labels
class BrainResponseDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        # Store the paths to our images and their labels
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Tell PyTorch how many images we have
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image and convert it to RGB (just in case it's grayscale)
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Apply any transformations we need (like resizing)
        if self.transform:
            image = self.transform(image)
            
        return image, label

# This is our brain image classifier - it's like a brain that learns to tell good from bad designs
class BrainResponseCNN(nn.Module):
    def __init__(self):
        super(BrainResponseCNN, self).__init__()
        # These layers help the model learn patterns in the images
        self.conv_layers = nn.Sequential(
            # First layer looks for simple patterns
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second layer looks for more complex patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Third layer looks for even more complex patterns
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # These layers help make the final decision
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # This helps prevent the model from memorizing too much
            nn.Linear(512, 1),
            nn.Sigmoid()  # This gives us a number between 0 and 1
        )

    def forward(self, x):
        # This is how the model processes an image
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def prepare_dataset():
    # Get all our images and their labels
    good_dir = os.path.join('topomaps', 'good')
    bad_dir = os.path.join('topomaps', 'bad')
    
    # Find all good and bad images
    good_images = [os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith('.png')]
    bad_images = [os.path.join(bad_dir, f) for f in os.listdir(bad_dir) if f.endswith('.png')]
    
    # Combine all images and their labels
    all_images = good_images + bad_images
    all_labels = [1] * len(good_images) + [0] * len(bad_images)  # 1 for good, 0 for bad
    
    # Split our data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    return X_train, X_val, y_train, y_val

def train_model():
    # Check if we can use GPU, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get our training and validation data
    X_train, X_val, y_train, y_val = prepare_dataset()
    
    # Set up how we'll transform our images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Make all images the same size
        transforms.ToTensor(),  # Convert images to numbers
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the numbers
    ])
    
    # Create our datasets
    train_dataset = BrainResponseDataset(X_train, y_train, transform=transform)
    val_dataset = BrainResponseDataset(X_val, y_val, transform=transform)
    
    # Create data loaders to feed our data to the model
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create our model
    model = BrainResponseCNN().to(device)
    
    # Set up how we'll train the model
    criterion = nn.BCELoss()  # This measures how wrong our predictions are
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # This helps improve our model
    
    # Train the model
    num_epochs = 50  # How many times we'll look at all our images
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            
            # Reset the optimizer
            optimizer.zero_grad()
            # Make predictions
            outputs = model(images)
            # Calculate how wrong we were
            loss = criterion(outputs.squeeze(), labels)
            # Learn from our mistakes
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
                # Count how many we got right
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate average losses and accuracy
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        # Print our progress
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {accuracy:.2f}%')
        
        # Save our best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model.pth')
            print(f'Model saved with validation loss: {val_loss:.4f}')

if __name__ == '__main__':
    train_model() 