import pandas as pd
from sklearn.model_selection import train_test_split

RAW_FILE = 'data/raw/Interchange_Training_File.xlsx'
PROCESSED_TRAIN = 'data/processed/train.csv'
PROCESSED_TEST = 'data/processed/test.csv'
TEST_INPUT = 'data/test/test_input.csv'
TEST_ANSWERS = 'data/test/test_answers.csv'

def load_data(filepath):
    df = pd.read_excel(filepath)
    df.columns = ['customer_description', 'part_number']
    return df

def clean_data(df):
    df = df.dropna(subset=['part_number'])
    df = df.drop_duplicates()
    df['customer_description'] = df['customer_description'].str.strip()
    df['part_number'] = df['part_number'].str.strip()
    df = df[df['customer_description'] != '']
    df = df[df['part_number'] != '']
    return df

def split_and_save(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(PROCESSED_TRAIN, index=False)
    test_df.to_csv(PROCESSED_TEST, index=False)
    test_df[['customer_description']].to_csv(TEST_INPUT, index=False)
    test_df.to_csv(TEST_ANSWERS, index=False)
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print("Files saved successfully")



if __name__ == '__main__':
    df = load_data(RAW_FILE)
    df = clean_data(df)
    split_and_save(df)