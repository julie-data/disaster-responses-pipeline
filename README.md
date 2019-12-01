# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code is using the following libraries: numpy, pandas, sklearn, nltk, sqlalchemy and pickle. The files contains all the import and download statement required.

After downloading all the files and placing them in the same folders, follow these instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Motivation<a name="motivation"></a>

The goal of this project is to categorize real messages sent during disaster events. I have built a machine learning pipeline to effectively categorize them so that the right messages could be sent to the appropriate relief agency.

## File Descriptions <a name="files"></a>

There are three folders in this repository:

1. data: 
    - disaster_messayges.csv: Data that contains all the messages of the dataset
    - disaster_categories.csv: Data that contains the messages' categories. Both files can be merged using the ID column.
    - process_data.py: the ETL script that prepares the data for the machine learning pipeline and saves it into a database
    
2. models:
    - train_classifier.py: script that runs a ML pipeline on the data and saves the model in a pickle file.
    
3. app: all the files related to the app where the messages' categories can be accesses.


## Results<a name="results"></a>

The main findings of the code can be found at the adress mentioned in the third point of the installation (section 1 of this README).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thank you first to Figure Eight for sharing their data, you can find more information about them [here](https://www.figure-eight.com/). Second, I would like to acknowledge Udacity for their great Data Scientist nanodegree of which this project is part of. Finally, I would like to thank Nicolas Essipova, my mentor for this Data Scientist nanodegree who have helped me out throughout this project.
