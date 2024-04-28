from myfunctions import *
import networkx as nx
import matplotlib.pyplot as plt
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#                               1. Data Collection and Preparation:
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# Combine and save data from scrapped articles to csv file "uncleaned_data.csv"
filename = 'uncleaned_data.csv'
filename_preprocess = 'preprocessed_data.csv'
combine_and_save_data(filename)

# Read the uncleaned data
uncleaned_data = pd.read_csv(filename)

# Separating into train and test data
train_set = uncleaned_data.iloc[:12]  # Access the first 12 rows
test_set = uncleaned_data.iloc[12:]   # Access the remaining rows

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#                               2. Pre-Processing:
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# Pre-Processing data on train
preprocessed_data = preprocess_data(train_set)

# Save preprocessed train dataset to CSV
save_preprocessed_data(preprocessed_data, filename_preprocess)

preprocessed_df = pd.DataFrame(preprocessed_data)
# filename_preprocess = 'preprocessed_data.csv'

# preprocessed_df = pd.read_csv(filename_preprocess)

# Print preprocessed train dataset
# for index, article in preprocessed_df.iterrows():
#     print(f"Label: {article['label']}")
#     print(f"Title: {article['title_tokens']}")
#     print(f"Content: {article['content_tokens']}")
#     print(f"Words Counts: {article['words_count']}")
#     print()

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#                               3. Graph Construction::
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# Generate the graph
graphs_train_set = []
for index, row in preprocessed_df.iterrows():
    # Build the directed graph
    graph = construct_graph(row['content_tokens'])
    graphs_train_set.append(graph)

    # Plot the graph
    # plot_graph(graph)

# Plot the first graph in the training set
print("Graph of the first article in the training set")
plot_graph(graphs_train_set[0])