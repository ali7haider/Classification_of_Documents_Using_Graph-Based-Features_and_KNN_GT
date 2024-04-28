import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# 1. Data Collection and Preparation:
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

def combine_and_save_data():
    # Read the three CSV files
    df1 = pd.read_csv("articles/disease_symtoms_articles.csv")
    df2 = pd.read_csv("articles/science_education_articles.csv")
    df3 = pd.read_csv("articles/sports_articles.csv")

    # Initialize an empty dataframe to store the combined data
    combined_df = pd.DataFrame()

    # Iterate over the rows of the dataframes and concatenate them row by row
    for i in range(5):
        combined_df = pd.concat([combined_df, df1.iloc[[i]]])
        combined_df = pd.concat([combined_df, df2.iloc[[i]]])
        combined_df = pd.concat([combined_df, df3.iloc[[i]]])

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv("uncleaned_data.csv", index=False)

    # Print the length of the combined dataframe
    print("Length of combined_df:", len(combined_df))

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
