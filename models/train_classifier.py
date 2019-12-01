import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    
    '''
    INPUT
    database_filepath - the path where the database has been created
    
    OUTPUT
    X - the messages
    Y - the categories (= labels)
    Y.columns - the names of the categories
    
    This function loads the data and prepare it in a X and Y function to be used by a ML model.
    '''
    
    # Get data
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterMessages',engine)
    
    # Split features from labels
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    
    return X, Y, Y.columns

def tokenize(text):
    
    '''
    INPUT
    text - text to tokenize
    
    OUTPUT
    words_lemmed - tokenized text
    
    This function transforms a text into something that can be read by a ML model:
    1. Set to lower case
    2. Remove punctuation
    3. Split the text into words
    4. Remove English stop words
    5. Lemmatize words (reduce them according to the dictionary)
    '''
    
    # Normalize
    text_normalized = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    
    # Tokenize text
    text_tokenized = word_tokenize(text_normalized)
    
    # Remove stop words
    words_no_stopwords = [w for w in text_tokenized if w not in stopwords.words("english")]
    
    # Lemmatize
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in words_no_stopwords]
    
    return words_lemmed  


def build_model():
    
    '''
    INPUT
    
    OUTPUT
    cv - final model
    
    This function creates a pipeline with a model to run on the data in order to categorize the messages automatically.
    '''
    
    # Build pipeline
    randomforest = RandomForestClassifier()

    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(randomforest))
    ])
      
    # Get best parameters with gridsearch
    parameters = {
        'clf__estimator__max_features': ['auto','log2'],
        'clf__estimator__min_samples_leaf': [1,2]
    }

    cv = GridSearchCV(pipeline,parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    INPUT
    model - model we want to evaluate (result from build_model() function)
    X_test - test split of X
    Y_test - test split of Y
    category_names - the names of the possible categories
    
    OUTPUT
    
    This function scores the model for each category separately.
    '''
    
    # Get predictions
    model_pred = model.predict(X_test)
    
    # Get scores for each category
    x = 0
    while x<36:
        print(category_names[x])
        print(classification_report(Y_test[Y_test.columns[x]],model_pred[:,x]))
        print('')
        x+=1


def save_model(model, model_filepath):
    
    '''
    INPUT
    model - model we want to evaluate (result from build_model() function)
    model_filepath - where to save the pickle file
    
    OUTPUT
    
    This function outputs the model in a pickle file.
    '''
    
    pkl_filename = model_filepath
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()