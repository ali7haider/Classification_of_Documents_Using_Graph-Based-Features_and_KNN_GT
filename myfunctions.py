import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')


def combine_and_save_data(filename):
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

def preprocess_data(data):
    preprocessed_data = []

    for index, row in data.iterrows():
        # Tokenize title and content
        title_tokens = tokenize(row['title'])
        content_tokens = tokenize(row['content'])

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