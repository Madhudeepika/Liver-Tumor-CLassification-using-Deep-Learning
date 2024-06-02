# Import necessary libraries
# pip install torch
# pip install torchvision
# pip install flask
#pip install --upgrade watchdog

# pip show watchdog

import os
print("Current Working Directory:", os.getcwd())

import os
import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torchvision.models as models  # Add this line

app = Flask(__name__)

# Define the path to the saved ResNet50 model
model_path = r'C:\Users\Lenovo\resnet\resnet50_finetuned.pth'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

num_classes = 3  # Update with the actual number of classes
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(2048, num_classes)  # 3 classes: cholangiocarcinoma, HCC, Normal Liver

# Load the trained model weights
state_dict = torch.load(model_path, map_location=torch.device('cpu'))


# Load the modified state_dict into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()


# Define a function to preprocess the image and get predictions
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = nn.functional.softmax(outputs[0], dim=0).numpy()

    # Provided class labels
    class_labels = ["cholangiocarcinoma", "HCC", "Normal Liver"]
    
    # Create a dictionary of class labels and their corresponding probabilities
    predictions = {label: prob.item() for label, prob in zip(class_labels, probabilities)}
    
    # Find the class with the highest probability
    predicted_class = max(predictions, key=predictions.get)
    highest_probability = predictions[predicted_class]

    return predicted_class, highest_probability

# Define a route for the home page
@app.route('/')
def home():
    return render_template(r'C:\Users\Lenovo\resnet\Template\liver.html')

# Define a route to handle image uploads and display predictions
@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file to a temporary location
    temp_image_path = "temp_image.jpg"
    file.save(temp_image_path)

    # Get predictions
    predicted_class, highest_probability = predict(temp_image_path)

    # Remove the temporary image file
    os.remove(temp_image_path)

    return jsonify({'predicted_class': predicted_class, 'highest_probability': highest_probability})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)



###########################



