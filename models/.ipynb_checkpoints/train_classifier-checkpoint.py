import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    engine = create_engine('sqlite:///{db_name}'.format(db_name = database_filepath))
    df = pd.read_sql_table('messages_categories', con = engine)
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    
    return X, Y, category_names
    
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        treat_tokens = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(treat_tokens)
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multi_clf', MultiOutputClassifier(RandomForestClassifier(random_state=42, n_jobs=-1)))
    ])
    
    parameters = {
        'multi_clf__estimator__n_estimators': [50, 100, 150],
        'multi_clf__estimator__min_samples_split': [2, 4]
        }

    cv = RandomizedSearchCV(pipeline, param_distributions=parameters, n_iter=1, cv=3, verbose=3, n_jobs=-1, random_state=42)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    model = model.best_estimator_
    y_pred = model.predict(X_test)
    
    for i, category in enumerate(category_names):
        print(f" Classification Report: {category}\n")
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))
        accuracy = (y_pred[:, i] == Y_test.iloc[:, i]).mean()
        print(f" Accuracy: {accuracy:.4f}\n")
        print("-" * 50)

def save_model(model, model_filepath):
    with open(model_filepath, "wb") as archive:
        pickle.dump(model, archive)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        
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