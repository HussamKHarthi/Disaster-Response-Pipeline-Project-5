# Disaster Response Pipeline Project - Project5

This is the fifth project in Data Scientist Nano Degree from UDACITY
In this project i've gone through making ETL & ML pipelines to classify a message according its content.


this Repo. includes the following: 
- app
- - template

- - - master.html  # main page of web app

- - - go.html  # classification result page of web app

- - run.py  # Flask file that runs app

- data

- - disaster_categories.csv  # data to process 

- - disaster_messages.csv  # data to process

- - process_data.py # Perform ETL

- - DisasterResponse.db   # database to save clean data to

- models

- - train_classifier.py # Building and training of the model 

- - classifier.pkl  # saved model 

- README.md

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database 

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

To run ML pipeline that trains classifier and saves 

`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Run the following command in the app's directory to run your web app. 

`python run.py`

Here is the web app:
https://view6914b2f4-3001.udacity-student-workspaces.com
