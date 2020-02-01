import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function will lead our datasets that our model will be trained on
    
    Parameters:
    messages_filepath: Messages file filepath
    categories_filepath: Categories filepath
    
    Return:
    Merged dataframe of the two files.
    '''
    messages = pd.read_csv(messages_filepath) #
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    
    return df


def clean_data(df):
    '''
    This function will clean and prepare our merged df and return it
    '''
    
    #Split each category as a seperate column
    categories = df.categories.str.split(pat=';',expand=True)
    
    # get only the name of each and discart its value
    row = categories.iloc[0,:].values
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    
    # Get the value of each entry and sets it as integer
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
    
    # Drop the column and concatenate values with messages and drop the duplicates
    df.drop(columns= 'categories',inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    
    return df



def save_data(df, database_filename):
    '''
    Here We're saving our prepared dataframe on the database
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_response', engine, index=False,if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()