import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model('D:\DEEP_LEARNING_PROJECTS\Digit_Mnist\my_model.keras')

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize the image
    image = np.reshape(image, (1, 28, 28, 1))  # Reshape to match model input
    return image

# Function to predict the digit
def predict_digit(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)  # Get the digit with the highest probability
    return predicted_digit

# Streamlit app
def main():
    st.title("MNIST Digit Recognition")
    
    # Upload image
    uploaded_file = st.file_uploader("Upload an image of a handwritten digit", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        
        # Show the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Predict the digit
        predicted_digit = predict_digit(image)
        
        # Show the prediction result
        st.write(f"Predicted Digit: {predicted_digit}")

if __name__ == '__main__':
    main()
