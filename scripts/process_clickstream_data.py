import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
DATA_PATH = "./data/e-shop clothing 2008.csv"
OUTPUT_PATH = "./outputs/"

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_data(file_path):
    """Load the clickstream dataset."""
    try:
        data = pd.read_csv(file_path, encoding='latin1')
        print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean the dataset."""
    # Check for duplicates
    print(f"Duplicate rows before cleaning: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print(f"Duplicate rows after cleaning: {df.duplicated().sum()}")

    # Handle missing values
    print(f"Missing values before cleaning:\n{df.isnull().sum()}")
    df = df.dropna()
    print(f"Missing values after cleaning:\n{df.isnull().sum()}")

    # Rename columns for consistency
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

def perform_eda(df):
    """Perform exploratory data analysis."""
    print(f"Dataset preview:\n{df.head()}")
    print(f"Column types:\n{df.dtypes}")
    print(f"Basic statistics:\n{df.describe()}")

    # Visualize distribution of key metrics
    if 'price' in df.columns:
        sns.histplot(df['price'], kde=True, bins=20)
        plt.title('Price Distribution')
        plt.savefig(os.path.join(OUTPUT_PATH, 'price_distribution.png'))
        plt.show()

    if 'user_id' in df.columns:
        session_counts = df['user_id'].value_counts()
        sns.histplot(session_counts, bins=30, kde=False)
        plt.title('Session Interaction Distribution')
        plt.savefig(os.path.join(OUTPUT_PATH, 'session_interactions.png'))
        plt.show()

def main():
    # Load and clean data
    df = pd.read_csv(DATA_PATH, delimiter=';')
    if df is not None:
        df_clean = clean_data(df)

        # Perform EDA
        perform_eda(df_clean)

        # Save cleaned dataset
        cleaned_file_path = os.path.join(OUTPUT_PATH, "cleaned_clickstream_data.csv")
        df_clean.to_csv(cleaned_file_path, index=False)
        print(f"Cleaned data saved to {cleaned_file_path}")

if __name__ == "__main__":
    main()
