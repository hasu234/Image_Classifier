import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
from classifier.CustomResnet import ResNet


# Define the path to your single test image
image_path = sys.argv[1]

# Check if the saved model file exists
model_path = 'resnet_model.pth'
if os.path.isfile(model_path):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Image preprocessing modules
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    # Load the single test image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # Create an instance of the model architecture
    model = ResNet().to(device)

    # Load the saved model state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Test the loaded model on the single image
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        class_index = predicted.item()

    # Get the class label associated with the predicted class index
    class_labels = ["berry", "bird", "dog", "flower"] 
    predicted_label = class_labels[class_index]

    print("Predicted Class:", predicted_label)

else:
    print("Saved model file not found.")
