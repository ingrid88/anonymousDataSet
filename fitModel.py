import pandas as pd
import sys
import pickle
from pyearth import Earth
import morph_data as md
import numpy as np

if __name__ == "__main__":
# fitModel.py < codetest_test.txt
    test = pd.read_csv(sys.stdin, delimiter='\t')
    print("Input File Header...")
    print(test.head())
    df = md.call_morph_data(test)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(np.array(df))
    np.savetxt('output.txt', y_pred)
    print("Predicted results printed to output.py")
