import pandas as pd 

def read_data():
    try:
        df = pd.read_csv('bank.csv', delimiter=";")
        df.drop(columns=['day', 'month'], inplace=True)

        # Check if the DataFrame is not empty
        if not df.empty:
            print(df.head())
        else:
            print("DataFrame is empty.")
    except FileNotFoundError:
        print(f"File not found: {file}")

    return df
