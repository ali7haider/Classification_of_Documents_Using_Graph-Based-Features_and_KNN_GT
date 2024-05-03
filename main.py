from myfunctions import *
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#                               1. Data Collection and Preparation(Pre-Processing):
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# Combine and save data from scrapped articles to csv file "uncleaned_data.csv"
filename = 'uncleaned_data.csv'
filename_preprocess = 'preprocessed_data.csv'
combine_and_save_data(filename)

# Read the uncleaned data
uncleaned_data = pd.read_csv(filename)

# Pre-Processing data on train
preprocessed_data = preprocess_data(uncleaned_data)

# Save preprocessed train dataset to CSV
save_preprocessed_data(preprocessed_data, filename_preprocess)

preprocessed_df = pd.DataFrame(preprocessed_data)

# Separating into train and test data
train_set = preprocessed_df.iloc[:36]  # Access the first 12 rows
test_set = preprocessed_df.iloc[36:]   # Access the remaining rows


#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#                               2. Graph Construction:
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# Generate the graph for the training set
train_graphs = []
for index, row in train_set.iterrows():
    # Build the directed graph
    graph = construct_graph(row['content_tokens'])
    train_graphs.append(graph)

# Generate the graph for the test set
test_graphs = []
for index, row in test_set.iterrows():
    # Build the directed graph
    graph = construct_graph(row['content_tokens'])
    test_graphs.append(graph)

# Plot a graph from the training set for visualization
plot_graph(train_graphs[2])

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#                               3. Classification with KNN:
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# Extracting labels
train_labels = train_set['label'].tolist()
test_labels = test_set['label'].tolist()

# Classification
i = 0
k = 3
predicted_labels = []
true_labels = []
for test_instance in test_graphs:
    predicted_label = knn(train_graphs, test_instance, k, train_labels)
    true_label = test_labels[i]
    i += 1
    predicted_labels.append(predicted_label)
    true_labels.append(true_label)
    print(f'Predicted class: {predicted_label} ------- Actual Class: {true_label}')

