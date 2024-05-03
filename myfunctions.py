import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import networkx as nx
import matplotlib.pyplot as plt
from gspan_mining import gSpan
import re
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict


nltk.download('punkt')


def combine_and_save_data(filename):
    # Read the three CSV files
    df1 = pd.read_csv("articles/disease_symtoms_articles.csv")
    df2 = pd.read_csv("articles/science_education_articles.csv")
    df3 = pd.read_csv("articles/sports_articles.csv")

    # Initialize an empty dataframe to store the combined data
    combined_df = pd.DataFrame()

    # Iterate over the rows of the dataframes and concatenate them row by row
    for i in range(15):
        combined_df = pd.concat([combined_df, df1.iloc[[i]]])
        combined_df = pd.concat([combined_df, df2.iloc[[i]]])
        combined_df = pd.concat([combined_df, df3.iloc[[i]]])

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(filename, index=False)

    # Print the length of the combined dataframe
    # print("Length of combined_df:", len(combined_df))


# Tokenization
def tokenize(text):
    return nltk.word_tokenize(text.lower())

# Stop-word removal
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

# Stemming
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Function to remove numbers, special characters, commas, full stops, and emojis
def clean_text(text):
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove special characters and emojis
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove commas and full stops
    text = text.replace(',', '').replace('.', '')
    return text

def preprocess_data(data):
    preprocessed_data = []

    for index, row in data.iterrows():
        # Clean title and content
        cleaned_title = clean_text(row['title'])
        cleaned_content = clean_text(row['content'])

        # Tokenize cleaned title and content
        title_tokens = tokenize(cleaned_title)
        content_tokens = tokenize(cleaned_content)

        # Remove stop words
        title_tokens = remove_stopwords(title_tokens)
        content_tokens = remove_stopwords(content_tokens)

        # Stem tokens
        title_tokens = stem_tokens(title_tokens)
        content_tokens = stem_tokens(content_tokens)

        # Count words in content
        words_count = len(content_tokens)

        # Append preprocessed data to the list
        preprocessed_data.append({
            'label': row['label'],
            'title_tokens': title_tokens,
            'content_tokens': content_tokens,
            'words_count': words_count
        })

    return preprocessed_data

def save_preprocessed_data(data, output_file):
    # Convert preprocessed data into a DataFrame
    preprocessed_df = pd.DataFrame(data)

    # Save the preprocessed data to a CSV file
    preprocessed_df.to_csv(output_file, index=False)

# Function to build directed graph
def construct_graph(tokens):
    graph = nx.DiGraph()
    for i in range(len(tokens) - 1):
        if not graph.has_edge(tokens[i], tokens[i+1]):
            graph.add_edge(tokens[i], tokens[i+1], weight=1)
        else:
            graph.edges[tokens[i], tokens[i+1]]['weight'] += 1
    return graph

# Function to plot the graph
def plot_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=2000, edge_color='blue', linewidths=2, font_size=12, font_weight='bold')
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_color='red', font_size=10)
    plt.show()
def knn(train_data, test_instance, k, train_labels):
    distances = []
    for i, train_instance in enumerate(train_data):
        label = train_labels[i]
        distance = compute_distance(test_instance, train_instance)
        distances.append((label, distance))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    class_counts = defaultdict(int)
    for neighbor in neighbors:
        class_counts[neighbor[0]] += 1
    predicted_class = max(class_counts, key=class_counts.get)
    return predicted_class

def compute_distance(graph1, graph2):
    mcs_size = find_mcs_size(graph1, graph2)
    max_edges = max(len(graph1.edges()), len(graph2.edges()))
    return 1 - (mcs_size / max_edges)

def find_mcs_size(graph1, graph2):
    common_edges = find_common_edges(graph1, graph2)
    return len(common_edges)

def find_common_edges(graph1, graph2):
    common_edges = set()
    for edge1 in graph1.edges():
        if edge1 in graph2.edges():
            common_edges.add(edge1)
    return common_edges