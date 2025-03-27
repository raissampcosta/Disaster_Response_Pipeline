import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Graph 1: Bar chart showing message count by genre with a line for percentage distribution
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    genre_total = genre_counts.sum()
    genre_percent = (genre_counts / genre_total * 100).round(1)

    graph1 = {
        'data': [
            Bar(
                x=genre_names,
                y=genre_counts,
                name='Count',
                marker=dict(color='#3B82F6')
            ),
            {
                'type': 'scatter',
                'mode': 'lines+markers+text',
                'x': genre_names,
                'y': genre_percent,
                'name': 'Percentage (%)',
                'yaxis': 'y2',
                'line': {'color': '#F79B6D', 'width': 3},
                'marker': {'size': 8, 'color': '#F79B6D'},
                'text': [f"{p}%" for p in genre_percent],
                'textposition': 'bottom center'
            }
        ],
        'layout': {
            'title': {
                'text': '<b>Distribution of Message Genres (Count + %)</b>',
                'x': 0.5,
                'xanchor': 'center'
            },
            'yaxis': {'title': "Message Count"},
            'yaxis2': {
                'title': "Percentage",
                'overlaying': 'y',
                'side': 'right',
                'showgrid': False,
                'tickformat': '.0f'
            },
            'xaxis': {'title': "Genre"},
            'margin': {'t': 80, 'l': 50, 'r': 50, 'b': 50},
            'height': 400,
            'legend': {
                'orientation': 'h',
                'x': 0.5,
                'xanchor': 'center',
                'y': 1.12
            }
        }
    }


    # Graph 2 - Pareto chart of categories showing count and cumulative percentage up to 80%
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = category_counts.index.str.replace('_', ' ').str.title()
    cumulative_percent = (category_counts.cumsum() / category_counts.sum() * 100).round(1)

    # Determine the threshold where the cumulative percentage reaches 81%
    cutoff_index = cumulative_percent[cumulative_percent <= 81].index

    # Subset the data up to the 81% cumulative threshold
    category_counts_pareto = category_counts.loc[cutoff_index]
    category_names_pareto = category_names[category_counts.index.isin(cutoff_index)]
    cumulative_percent_pareto = cumulative_percent[category_counts.index.isin(cutoff_index)]

    graph2 = {
        'data': [
            Bar(
                x=category_names_pareto,
                y=category_counts_pareto,
                name='Count',
                marker=dict(color='#3B82F6')
            ),
            {
                'type': 'scatter',
                'mode': 'lines+markers+text',
                'x': category_names_pareto,
                'y': cumulative_percent_pareto,
                'name': 'Cumulative %',
                'yaxis': 'y2',
                'line': {'color': '#F79B6D', 'width': 3},
                'marker': {'size': 6, 'color': '#F79B6D'},
                'text': [f"{p}%" for p in cumulative_percent_pareto],
                'textposition': 'top center'
            }
        ],
        'layout': {
            'title': '<b>Pareto Chart (Top Categories up to 80%)</b>',
            'yaxis': {'title': "Count"},
            'yaxis2': {
                'title': "Cumulative %",
                'overlaying': 'y',
                'side': 'right',
                'tickformat': '.0f',
                'showgrid': False
            },
            'xaxis': {
                'title': "Category",
                'tickangle': -45
            },
            'margin': {'t': 80, 'l': 50, 'r': 50, 'b': 120},
            'height': 500,
            'legend': {
                'orientation': 'h',
                'x': 0.5,
                'xanchor': 'center',
                'y': 1.12
            }
        }
    }
    
    # Graph 3: Heatmap showing the correlation between the top 10 most frequent categories
    category_data = df.iloc[:, 4:]
    correlation = category_data.corr().round(2)

    # Select the top 10 most frequent categories
    top10 = category_data.sum().sort_values(ascending=False).head(10).index
    correlation_top10 = correlation.loc[top10, top10]
    
    graph3 = {
        'data': [
            {
                'z': correlation_top10.values,
                'x': correlation_top10.columns,
                'y': correlation_top10.index,
                'type': 'heatmap',
                'colorscale': [
                    [0.0, 'F79B6D'],
                    [1.0, '3B82F6']
                ],
                'colorbar': {'title': 'Correlation'}
            }
        ],
        'layout': {
            'title': '<b>Correlation Between Top 10 Categories</b>',
            'height': 500,
            'margin': {'t': 80, 'l': 100, 'r': 30, 'b': 100},
            'xaxis': {'tickangle': -45},
        }
    }

    # Graph 4: 100% stacked bar chart showing the genre distribution across the top 5 categories

    top_categories = df.iloc[:, 4:].sum().sort_values(ascending=False).head(5).index

    # Count messages by genre within each category
    stack_data = df.groupby('genre')[top_categories].sum()

    # Normalize values to get percentage per category (column-wise)
    stack_percent = stack_data.div(stack_data.sum(axis=0), axis=1).round(4) * 100
    genre_colors = {
        'direct': '#3B82F6',
        'news': '#F79B6D',
        'social': '#C7EB63'
    }
    
    graph4 = {
        'data': [
            Bar(
                x=top_categories.str.replace('_', ' ').str.title(),
                y=stack_percent.loc[genre],
                name=genre.capitalize(),
                marker=dict(color=genre_colors[genre])
            ) for genre in stack_percent.index
        ],
        'layout': {
            'barmode': 'stack',
            'title': '<b>Top 5 Categories by Genre (100% Stacked)</b>',
            'xaxis': {'title': 'Category'},
            'yaxis': {'title': 'Percentage', 'range': [0, 100]},
            'margin': {'t': 80, 'l': 50, 'r': 50, 'b': 100},
            'legend': {
                'orientation': 'h',
                'x': 0.5,
                'xanchor': 'center',
                'y': 1.12
            },
            'height': 500
        }
    }


    # Final list containing all graphs
    graphs = [graph1, graph2, graph3, graph4]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()