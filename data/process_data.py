import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # Import messages file
    messages = pd.read_csv(messages_filepath)
    
    # Import categories file
    categories = pd.read_csv(categories_filepath)
    
    # Merge the two files together
    df = messages.merge(categories, on=('id'))
    
    return df


def clean_data(df):
    # Split the 'categories' column into separate columns
    categories = df['categories'].str.split(";",expand=True)
    
    # Get categories names and create columns
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # Convert categories values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column])
        
    # Replace 'categories' column in 'df' with new category columns
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///DisasterMessages.db')
    engine.execute('DROP TABLE IF EXISTS DisasterMessages', con = engine)
    df.to_sql('DisasterMessages', engine, index=False)

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