from myfunctions import *
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# 1. Data Collection and Preparation:
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# Call the function
combine_and_save_data()

csv_file_path  = 'uncleaned_data.csv'

# Read in the data
uncleaned_data = pd.read_csv(csv_file_path )

train_set = uncleaned_data.iloc[:12]  # Access the first 12 rows
test_set = uncleaned_data.iloc[12:]   # Access the remaining rows

# Print the number of articles in each set
print("Training set size:", len(train_set))
print(train_set['label'])
print("Test set size:", len(test_set))
print(test_set['label'])
