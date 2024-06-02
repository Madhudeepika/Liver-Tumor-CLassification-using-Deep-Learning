

# Import necessary libraries
# pip install torch
# pip install torchvision
# pip install flask
# pip install --upgrade watchdog

# Import necessary libraries
import os
import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Define the path to the saved ResNet50 model
model_path = r'C:\Users\Lenovo\resnet\resnet50_finetuned.pth'

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
num_classes = 1  # Assuming you have 3 classes: cholangiocarcinoma, HCC, Normal Liver
model.fc = torch.nn.Linear(2048, num_classes)  # Adjust the number of input features



# Load your ResNet50 model
model = models.resnet50(pretrained=False)
num_classes = 2048 # Update with the actual number of classes
original_fc_size = 1 # Update with the size of the original "fc" layer

# Modify the size of the fully connected layer to match the original model
model.fc = torch.nn.Linear(original_fc_size, num_classes)

# Load the trained model weights
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Extract the weights and biases from the state dictionary
weights = state_dict['layer4.1.bn3.weight']
biases = state_dict['layer4.1.bn3.bias']

# Resize the weights to match the modified model
weights_resized = weights.view(num_classes, original_fc_size)

# Load the resized weights into the modified model
model.fc.weight.data.copy_(weights_resized)

# Resize the biases to match the modified model
biases_resized = biases.view(num_classes)

# Load the resized biases into the modified model
model.fc.bias.data.copy_(biases_resized)

# Set the model to evaluation mode
model.eval()



# Set the model to evaluation mode
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()

    # Provided class labels
    class_labels = ["cholangiocarcinoma", "HCC", "Normal Liver"]
    
    # Find the class with the highest probability
    predicted_class = class_labels[probabilities.argmax()]
    highest_probability = probabilities.max()

    return predicted_class, highest_probability

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

for key in state_dict.keys():
    print(key)
    
    
    
    
import torch
from torchvision import models

# Load your ResNet50 model
model = models.resnet50(pretrained=False)

# Print the model architecture
print(model)
