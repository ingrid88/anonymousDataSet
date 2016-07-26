
## pip install pyearth
import pandas as pd
import sys
import pickle
from pyearth import Earth
import morph_data as md
import numpy as np

if __name__ == "__main__":
# python BuildModel.py < codetest_train.txt
    train = pd.read_csv(sys.stdin, delimiter='\t')
    print("Input File Header...")
    print(train.head())
    df = md.call_morph_data(train)
    y = df['target']
    X = df.drop('target',1)

    print("Fit an Earth model...")
    model = Earth(max_degree=4)
    model.fit(np.array(X),np.array(y))
    print(model.summary())
    print(model.trace())
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved Model as model.pkl in current directory...")
    print("Next step type: python fitModel.py < codetest_test.txt")
