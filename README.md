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
LM studio

## Installation
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

### Step 3: Install LM Studio (for LLM functionality)
- Download LM Studio
- Visit the official LM Studio GitHub repository and follow the instructions for your operating system. [LM studio](https://lmstudio.ai/)

- Set up the LLM Model
- Download the Meta-Llama-3.1-8B-Instruct-GGUF model from the LM Studio website or supported sources.
- Place the model in the directory specified by LM Studio.
- Run LM Studio
- Start LM Studio and ensure it is running on the default port 1234. The application will connect to LM Studio to fetch responses for chatbot queries.

### Step 5: Run the Flask application:
```
python app.py
```

### The application will be accessible at http://127.0.0.1:3000/.

Usage
- Open the application in your web browser.
- Enter the required input data, such as symptoms, age, medical history, etc., into the provided form fields.
- Click on the "Predict" button to submit the data.
- View the predicted GI disease and related information on the results page.
- for the working of LLM i am using LM studio with model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"


### Running Test Cases
- To ensure that the application is functioning correctly, you can run the test cases provided.

- Step 1: Install pytest and related dependencies
``` 
pip install pytest pytest-flask
```
- Step 2: Run the test cases
- Navigate to the root directory of the project and execute:
```
pytest test_app.py --disable-warnings
```
- This will: Test routes like login, logout, and registration. Validate the file upload and prediction functionalities.

### Project Structure
```bash
gi-disease-prediction-app/
│
├── test_app.py                 # Main Test application file
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
│   └── users.py # Pre-trained machine learning model
├── models1/
│   └── best.keras  # Pre-trained machine learning model
├── uploads/
│   └── uploaded images from user  # uploaded images from user
├── requirements.txt       # List of dependencies
└── README.md              # Readme file
```


###  Customization
- Updating the Model: Replace the best.keras file in the models/ directory with your own trained model to improve predictions or add new features.
- Frontend Customization: Modify the HTML templates in the templates/ directory to change the user interface as needed.


### Future Improvements
Expand the input data to include more comprehensive health information.
Integrate additional machine learning models for more accurate predictions.
Add authentication and user profiles for a personalized experience.


### Contributing
If you would like to contribute to this project, feel free to open a pull request or submit an issue on GitHub.

### License
This project is licensed under the MIT License.

### Contact
For any questions or feedback, please reach out to [ggali14@vt.edu].
