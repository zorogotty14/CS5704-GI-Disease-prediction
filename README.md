# GI Disease Prediction Web Application
## Overview
This web application is built using Flask and is designed to predict gastrointestinal (GI) diseases based on user inputs such as symptoms, medical history, and other relevant data. The application leverages a machine learning model to provide predictions, helping users to better understand their symptoms and potential health conditions.

## Features
- User-friendly web interface to input relevant health data
- Predicts potential GI diseases using trained machine learning models
- Provides explanations and insights into the predictions
- Customizable and extendable to support additional diseases and input features

## Prerequisites
Ensure that you have the following installed:

Python 3.x
Flask
A virtual environment (recommended)
Installation
Follow these steps to set up and run the application locally:

## Clone the repository:

### Step 1: Clone the repository
```bash
Copy code
git clone https://github.com/your-repo/gi-disease-prediction-app.git
cd gi-disease-prediction-app
Create and activate a virtual environment:
``` 

### Step 2: Create virtual environment 
```bash 
# On Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```
### Step 2: Create virtual environment 
# On Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 2: Install the dependencies:

```bash
pip install -r requirements.txt

```

### Step 4: Run the Flask application:
```
python app.py
```

### Step 4: The application will be accessible at http://127.0.0.1:5000/.

Usage
- Open the application in your web browser.
- Enter the required input data, such as symptoms, age, medical history, etc., into the provided form fields.
- Click on the "Predict" button to submit the data.
- View the predicted GI disease and related information on the results page.
- for the working of LLM i am using LM studio with model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"


### Step 4: Project Structure
```bash
gi-disease-prediction-app/
│
├── app.py                 # Main application file
├── templates/
│   ├── index.html         # Home page template
│   ├── result.html        # Results page template
│   ├── style.css          # stylesheet page template   
│   └── error.html        # Error page template
├── static/
│   ├── images/
│   │   └── images for website     # images for website
│   └── uploads/
│       └── uploaded images from user to be displayed on webpage       # uploaded images from user
├── models/
│   └── best.keras  # Pre-trained machine learning model
├── uploads/
│   └── uploaded images from user  # uploaded images from user
├── requirements.txt       # List of dependencies
└── README.md              # Readme file
```


### Step 4: Customization
- Updating the Model: Replace the best.keras file in the models/ directory with your own trained model to improve predictions or add new features.
- Frontend Customization: Modify the HTML templates in the templates/ directory to change the user interface as needed.


### Step 5: Future Improvements
Expand the input data to include more comprehensive health information.
Integrate additional machine learning models for more accurate predictions.
Add authentication and user profiles for a personalized experience.


### Contributing
If you would like to contribute to this project, feel free to open a pull request or submit an issue on GitHub.

### License
This project is licensed under the MIT License.

### Contact
For any questions or feedback, please reach out to [ggali14@vt.edu].
