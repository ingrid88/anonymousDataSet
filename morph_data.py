## pip install fancyimpute
import pandas as pd
import numpy as np
from fancyimpute import KNN

def categorical_columns(dataset):
    def is_categorical(array_like):
        return array_like.dtype.name != 'float64'

    categorical_data = []
    for k in dataset.keys():
        if is_categorical(dataset[k]):
            categorical_data.append(k)
    return categorical_data

def categorical_fill(dataframe,cat_data):
    keep = []
    df = dataframe.copy()
    for cat in cat_data:
        f = df[cat].dropna()
        empty = pd.isnull(df[cat]).nonzero()[0]
        t = pd.DataFrame(data=list(f.sample(n=len(empty))),index=empty)
        df[cat] = df[cat].fillna(t.to_dict()[0])
    return df

def dummy_columns(categorical_data,dataset):
    d = []
    for col in categorical_data:
        df = pd.get_dummies(dataset[col]) #.ix[:,1::]
        d.append(df)
    df = pd.concat(d,axis=1)
    return df


def call_morph_data(train):
    print("Find Categorical Features...")
    categorical_data = categorical_columns(train)
    print(categorical_data)
    print("Fill Categorical NaNs with Distribution...")
    fill_train = categorical_fill(train, categorical_data)
    print("Build Dummy columns for each category...")
    dummy_df = dummy_columns(categorical_data,fill_train)
    print(dummy_df.head(n=2))
    print("Fill Continuous data with k nearest neighbor values...")
    fill_train = pd.concat([fill_train.drop(categorical_data,1),dummy_df],1)
    X_filled_knn = KNN(k=3).complete(fill_train) #fill_train[keeps]
    df = pd.DataFrame(X_filled_knn,columns=list(fill_train.keys()))
    return df
