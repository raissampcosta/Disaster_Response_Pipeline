import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them on 'id'.

    Args:
        messages_filepath (str): File path for the messages CSV file.
        categories_filepath (str): File path for the categories CSV file.

    Returns:
        DataFrame: Merged dataset containing messages and their respective categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'left', on = 'id')
    
    return df

def clean_data(df):
    """
    Clean the merged dataset by splitting categories, converting values to binary,
    and removing duplicates.

    Args:
        df (DataFrame): Merged dataset of messages and categories.

    Returns:
        DataFrame: Cleaned dataset with binary category columns.
    """
    categories = df['categories'].str.split(';', expand = True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1:])
        categories[column] = categories[column].astype(int)

    df.drop(columns = 'categories', inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(inplace = True)

    return df

def save_data(df, database_filename):
    """
    Save the cleaned data into a SQLite database.

    Args:
        df (DataFrame): Cleaned dataset.
        database_filename (str): Path where the SQLite database will be saved.
    """
    engine = create_engine('sqlite:///{db_name}'.format(db_name = database_filename))
    df.to_sql('messages_categories', engine, index = False, if_exists='replace')  

def main():
    """
    Main function to run the ETL pipeline:
    - Load data from files
    - Clean the data
    - Save the data into a SQLite database
    """
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