# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:05:22 2020

@author: sukandulapati
"""
from pathlib import Path
#from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from mlxtend.data import loadlocal_mnist
from sklearn.metrics import mean_squared_error
import pandas as pd

test_images_path = Path('t10k-images.idx3-ubyte')
test_lables_path = Path('t10k-labels.idx1-ubyte')

def load_mnist_test_data(test_images_path, test_lables_path):

    test_digit, test_label = loadlocal_mnist(images_path=test_images_path,
                                             labels_path=test_lables_path)
    return test_digit, test_label

test_digit, test_label = load_mnist_test_data(test_images_path, test_lables_path)

# load the model
digit_recognition_model = joblib.load('digit_recognition_model.sav')
predictions = digit_recognition_model.predict(test_digit)

print("Accuracy of the model: ", digit_recognition_model.score(test_digit, test_label))
print("RMSE of the model: ", mean_squared_error(test_label, predictions))
result_df = pd.concat([pd.Series(predictions), pd.Series(predictions)], axis=1)
result_df.columns=['actual', 'predition']
result_df.to_csv('digit_prediction_df.csv', index=False)