import pandas as pd 
from sklearn.preprocessing import LabelEncoder , StandardScaler



def preprocessing_data(df):
    cat = ['job', 'marital'	,'education',	'default' , 'housing'	,'loan'	, 'contact', 'poutcome']
    scale = ['balance', 'duration', 'pdays']

    encoder = LabelEncoder()
    
    for column in cat:
        df[column] = encoder.fit_transform(df[column])

    scaler = StandardScaler()

    # Fit and transform the specified columns
    scaled_columns = scaler.fit_transform(df[scale])

    # Replace the original columns with scaled columns in the DataFrame
    df[scale] = scaled_columns

    return df
