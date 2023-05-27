import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import sys
from classifier.Resnet import ResNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 10
batch_size = 100
learning_rate = 0.002

# Access the command-line arguments
if len(sys.argv) < 2:
    print("Please provide the dataset directory path as an argument.")
    sys.exit(1)

# Retrieve the dataset directory path from the command-line argument
dataset_path = sys.argv[1]

# # Define the path to your dataset folder
# dataset_path = 'dataset_256X256'

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

# Load the training dataset
train_dataset = ImageFolder(root=dataset_path + "/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load the testing dataset
test_dataset = ImageFolder(root=dataset_path + "/test", transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = ResNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in tqdm(range(num_epochs)):
    train_loss = 0.0
    train_correct = 0
    total_train = 0
    
    # Set the model to train mode
    model.train()
    
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Calculate training loss and accuracy
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
    # Calculate average training loss and accuracy
    train_loss /= total_train
    train_accuracy = train_correct / total_train
    
    val_loss = 0.0
    val_correct = 0
    total_val = 0
    
    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate validation loss and accuracy
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            total_val += labels.size(0)
    
    # Calculate average validation loss and accuracy
    val_loss /= total_val
    val_accuracy = val_correct / total_val
    
    # Print or log the training and validation loss and accuracy for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}  Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}  Val Accuracy: {val_accuracy:.4f}")
    print()
    
# Save the trained model
torch.save(model.state_dict(), 'resnet_model.pth')

# torch.save(model,'resnet_model1.pth')


# run by python script.py /path/to/dataset_directory
