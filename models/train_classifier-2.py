import sys
import pandas as pd
import numpy as np
import re
import string
import nltk 
import pickle
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
nltk.download(['wordnet','punkt','stopwords','averaged_perceptron_tagger'])
stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
remove_punc_table = str.maketrans('', '', string.punctuation)

def load_data(database_filepath):
    '''
    This function will read our dataframe from the database we used in process_data file
    df will be split into three:
    X: text messages
    Y: messages classification
    category_names: the name of the columns
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response',engine)
    X = df.message
    Y = df.iloc[:,4:]
    category_names=Y.columns
    
    return X,Y,category_names


def tokenize(text):
    '''
    This funnction will tokenize our messages to prepare them to train our model on them.
    
    Parameter:
    text: message to be tokenized
    
    return:
    lemmatized and tokenized text
    
    '''
    text = text.translate(remove_punc_table).lower() # revmove punc. & lower case
    
    tokens = nltk.word_tokenize(text) #tokenize
    
    #lemmatize
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]


def build_model():
    '''
    This function will specify the parameters of our model
    The chosen classifier is RandomForestClassifier with gridsearch for optimal number of estimator between 10 and 50
    
    Return:
    Model with optimal nnumber of estimators
    '''
    pipeline = Pipeline([
                    ('vect',CountVectorizer(tokenizer=tokenize)), 
                    ('tfidf',TfidfTransformer()),
                    ('clf',MultiOutputClassifier(RandomForestClassifier()))
                    ])
    parameters= {'clf__estimator__n_estimators': [10,50]}
    cv = GridSearchCV(pipeline,parameters)
    return cv
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function will predict the our testing messages and print out the score for each category.
   
   Parameters:
   model: trained model
   X_test: testing messages
   Y_test: True categorization of these messages
   category_names: category names
    '''
    y_pred= model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns = Y_test.columns)
    for column in Y_test.columns:
        print(column)
        print(classification_report(Y_test[column],y_pred_df[column]))
        print('------------------------------------------------------\n')

def save_model(model, model_filepath):
    '''
    save our trained model as pickle file
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))

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