# Disaster Response Pipeline Project
This project is part of a disaster response pipeline that processes real messages sent during disasters and classifies them into multiple categories. The project includes a machine learning model and an interactive web dashboard to visualize insights from the data.

### Table of Contents
- [Installation](#installation)  
- [Project Motivation](#project-motivation)  
- [File Descriptions](#file-descriptions)
- [How to Run the Project](#how-to-run-the-project)
- [Results](#results) 

### Installation
To run this project, you’ll need Python 3 and the following libraries installed:
1. pandas
2. numpy
3. sqlalchemy
4. nltk
    - Note: You may also need to download punkt from nltk
       - import nltk
       - nltk.download('punkt') 
5. scikit-learn
6. flask
7. plotly
8. joblib

### Project Motivation
This project was completed as part of the Udacity Data Scientist Nanodegree. The goal is to process real disaster response messages, classify them into multiple categories using a machine learning pipeline, and build an interactive web application that displays classification results and insightful visualizations.

The data comes from real messages sent during disaster events and can help emergency services better understand and respond to needs during crises.

### File Descriptions
- **`app/`** – Flask app that launches the web interface. Loads data from the SQLite database and model from a pickle file.
  - `run.py` – Runs the web application and renders the dashboard
  - `templates/
      - master.html` – Main page layout for index and results.
      - go.html` – Page that displays the message classification results.
  - `static/style.css` – Custom styles for layout and charts

- **`data/`** – Dataset and processing scripts
  - `disaster_messages.csv` – Dataset containing real disaster-related messages.
  - `disaster_categories.csv` – Categories associated with each message.
  - `process_data.py` – ETL pipeline script that:
      - Loads and merges both datasets;
      - Cleans and transforms the data;
      - Saves the clean data into a SQLite database.
  - `DisasterResponse.db` – SQLite database generated after running the ETL.

- **`models/`** – Machine learning pipeline
  - `train_classifier.py` – Script that:
      - Loads data from the SQLite database;
      - Builds and trains a machine learning model using a pipeline with TF-IDF and MultiOutputClassifier;
      - Uses GridSearchCV for hyperparameter tuning;
      - Outputs classification reports for all categories;
      - Saves the trained model as classifier.pkl.
  - `classifier.pkl` – Trained and serialized model file.

### How to Run the Project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to your browser and visit http://localhost:3000/

### Results
1. Classification: The model classifies a given user message into 36 possible categories.

2. Accuracy: The model achieves high accuracy across several categories. Classification reports are printed for each.

3. The app provides insightful data visualizations:
    - Genre distribution with count and percentage
    - Pareto chart of top categories up to 80%
    - Heatmap of correlation between top categories
    - Stacked bar of top categories by genre

4. You can also input a custom message and receive real-time classification across multiple disaster-related categories.
