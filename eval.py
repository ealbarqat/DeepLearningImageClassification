# You can import any module, model class, etc.
# We will import the `load_and_predict()` function below to assess your assignment.

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from train import BrainResponseCNN

def load_model(model_path):
    # Check if we can use GPU, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create our model and load the saved weights
    model = BrainResponseCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Tell the model we're testing, not training
    return model, device

def predict_image(model, device, image_path):
    # Set up how we'll transform our images (same as in training)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Make the image the right size
        transforms.ToTensor(),  # Convert image to numbers
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the numbers
    ])
    
    # Load and prepare the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Make a prediction
    with torch.no_grad():  # We don't need to calculate gradients for testing
        output = model(image)
        prediction = (output.squeeze() > 0.5).float().item()  # Convert to 0 or 1
    
    return prediction

def evaluate_model(model_path, test_dir):
    # Load our trained model
    model, device = load_model(model_path)
    
    # Find all the images we want to test
    image_paths = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    
    # Make predictions for each image
    predictions = {}
    for image_path in image_paths:
        prediction = predict_image(model, device, image_path)
        predictions[image_path] = prediction
    
    return predictions

if __name__ == '__main__':
    # Set up our file paths
    model_path = 'model.pth'  # Our trained model
    test_dir = 'topomaps'  # Where our test images are
    
    # Check if our files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        exit(1)
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found!")
        exit(1)
    
    # Get predictions for all images
    predictions = evaluate_model(model_path, test_dir)
    
    # Show our results
    print("\nPredictions:")
    print("------------")
    for image_path, prediction in predictions.items():
        print(f"{os.path.basename(image_path)}: {'Good' if prediction == 1 else 'Bad'}")

