# Disaster Response Pipeline Project
This project is part of a disaster response pipeline that processes real messages sent during disasters and classifies them into multiple categories. The project includes a machine learning model and an interactive web dashboard to visualize insights from the data.

### Table of Contents
- [Installation](#installation)  
- [Project Motivation](#project-motivation)  
- [File Descriptions](#file-descriptions)  
- [Results](#results)  
- [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)  

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Project Motivation
During natural disasters and humanitarian crises, messages from the public are sent through multiple sources. These messages often need to be classified quickly into actionable categories (e.g., requests for water, food, shelter, or medical help).

This project was built to:
    - Train a model that automatically classifies such messages into multiple categories
    - Build an interactive dashboard to analyze the data
    - Provide an easy interface for real-time message classification

### File Descriptions
├── app
│   ├── run.py - Flask app with dashboard + classifier
│   ├── templates/master.html - Main HTML layout
│   └── static/style.css - Custom styling for UI
│
├── data
│   ├── disaster_messages.csv - Raw message data
│   ├── disaster_categories.csv - Category labels
│   ├── process_data.py - Script to clean & save to SQLite
│   └── DisasterResponse.db - SQLite database
│
├── models
│   ├── train_classifier.py - Model training pipeline
│   └── classifier.pkl - Saved trained model

### Results
The dashboard includes the following visualizations:
    - Genre Distribution (Bar + % Line)
    - Pareto Chart (Cumulative % up to 80%)
    - Correlation Heatmap of Top 10 Categories
    - 100% Stacked Bar: Top Categories by Genre

You can also input a custom message and receive real-time classification across multiple disaster-related categories.
