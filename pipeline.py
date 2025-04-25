import torch
import torch.nn as nn
import torch.quantization
import cv2
import numpy as np
from torchvision import transforms
import timm
from PIL import Image

# Define backend for quantized model
torch.backends.quantized.engine = 'qnnpack'

# Define EmotionModel
# We use a quantized tiny DeiT with 200k parameters
class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        # Load model and quantize
        self.model = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=7)
        self.model.load_state_dict(torch.load('/Users/h22qin/Desktop/CV_Final_Proj/model.pth', map_location=torch.device('cpu')), strict=True)
        '''
        I have to quantize the model here since my training was run on GPU and previously quantized model is not compatible
        However, if you are running on CUDA backend, can directly import 'quantized_deit_full.pth' or 'quantized_deit_0.2M.pth'
        '''
        quantized_model = torch.quantization.quantize_dynamic(
                self.model,  
                {nn.Linear, nn.Conv2d},  
                dtype=torch.qint8  
            )
        self.model = quantized_model
    
    def forward(self, x):
        return self.model(x)

emotion_model = EmotionModel().cpu() 
emotion_model.eval() 

# Same preprocessing logic as the training -- standard normalization for ImageNet
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])
    return transform(image).unsqueeze(0).cpu()  # Add batch dimension and move to CPU

# Classification from captured frame, output both prediction and probability
def classify_emotion(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Preprocess
    input_tensor = preprocess_image(pil_image)

    # Predict using EmotionModel
    with torch.no_grad():
        output = emotion_model(input_tensor)
    
    # Use softmax to find probability of labels
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get predicted label
    _, predicted = torch.max(probabilities, 1)
    predicted_emotion = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"][predicted.item()]
    emotion_prob = probabilities[0][predicted.item()].item()

    return predicted_emotion, emotion_prob

# Haar Cascade model for facial recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Main logic to capture face and run model inference
def process_frame_with_face(frame):

    # Facial detection using casecade model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If no face is detected, return the original frame
    if len(faces) == 0:
        return frame
    
    # For all faces detected, run model inference
    for (x, y, w, h) in faces:
        # Crop the fram
        face = frame[y:y+h, x:x+w]
        # Classify the emotion and get probability
        emotion, probability = classify_emotion(face)
        
        # Draw rectangleand display the emotion label and probability
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion} ({probability*100:.2f}%)", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

# Main loop to run real time emotion detection
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame_with_face(frame)

        # Show the current image with the emotion label and probability
        cv2.imshow("Video", processed_frame)

        key = cv2.waitKey(1) & 0xFF

        # Exit the loop when the 'ESC' key is pressed
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
