from myfunctions import *
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

# Print preprocessed train dataset

# for index, article in train_set.iterrows():
#     print(f"Label: {article['label']}")
#     print(f"Title: {article['title']}")
#     print(f"Words Counts: {article['words_count']}")
#     print()