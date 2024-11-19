from flask import Flask, render_template, request, jsonify, redirect, url_for,session
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from openai import OpenAI
import re
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)


# Define the model path and load the model
MODEL_PATH = "./models/best.keras"
model = load_model(MODEL_PATH)

# Define confidence threshold for classifying as "not a brain MRI"
CONFIDENCE_THRESHOLD = 0.6

# Define the classes based on your model training
class_labels = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']


def preprocess_image(file_path, image_size=(224, 224)):
    img = cv2.imread(file_path)  # Read the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, image_size)  # Resize to match model input size
    img = img / 255.0  # Scale image pixels if required by your model
    img = np.expand_dims(img, axis=0)  # Expand dims for batch size
    return img

# Initialize the OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Function to get classification details based on predicted class
def get_tumor_info(tumor_type):
    completion = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        messages=[
            {"role": "system", "content": "You are a healthcare assistant."},
            {"role": "user", "content": f"Provide detailed information about {tumor_type} gastrointestinal disease"}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content


def format_tumor_info(text):
    # Wrap main headings (e.g., "Types of Meningiomas") as <h3>
    text = re.sub(r"\*\*(.+?)\*\*", r"<h3 class='section-heading'>\1</h3>", text)
    
    # Wrap bolded items (e.g., subheadings or important terms) as <strong>
    text = re.sub(r"(\*\*.+?\*\*)", lambda match: f"<strong>{match.group(0)[2:-2]}</strong>", text)
    
    # Replace numbered or bullet lists with <li> items
    text = re.sub(r"(?:•\s|\d+\.\s)(.+)", r"<li>\1</li>", text)
    
    # Wrap lists with <ul> tags
    text = re.sub(r"(</h3>)\s*((?:<li>.+?</li>\s*)+)", r"\1<ul class='no-marker'>\2</ul>", text)
    
    # Replace new lines with paragraph tags
    text = re.sub(r"\n+", r"</p><p>", text)
    text = f"<p>{text}</p>"

    return text



# Route for handling chatbot messages with session history
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    # Retrieve conversation history from session or initialize with tumor info
    conversation_history = session.get('conversation_history', [])
    conversation_history.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        messages=conversation_history,
        temperature=0.7,
    )
    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})

    # Update session with the conversation history
    session['conversation_history'] = conversation_history
    return jsonify({"reply": reply})

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Route to handle prediction and store initial conversation context
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Ensure uploads folder exists
    upload_folder = './static/uploads/'
    os.makedirs(upload_folder, exist_ok=True)

    # Secure the filename and save the file
    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)

    # Print file path for debugging
    print(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_labels[predicted_class_index]
    
    # Result text
    result_text = f"Predicted gastrointestinal condition: {predicted_class_name}"
    print(result_text)
    
    # Optionally get more details (if applicable)
    raw_tumor_info = get_tumor_info(predicted_class_name)
    tumor_info = format_tumor_info(raw_tumor_info)
    
    # Initialize conversation history with information (if applicable)
    session['conversation_history'] = [
        {"role": "assistant", "content": f"{result_text}. {tumor_info}"}
    ]
    
    # Render the result template with the relative file path
    return render_template('result.html', 
                           result_text=result_text,
                           tumor_info=tumor_info,
                           file_path=f'uploads/{filename}')




# Run the app
if __name__ == '__main__':
    app.run(port=3000, debug=True)