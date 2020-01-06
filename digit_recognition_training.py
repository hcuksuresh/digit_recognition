# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:06:04 2020

@author: sukandulapati
"""
from pathlib import Path
from mlxtend.data import loadlocal_mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib


train_images_path = Path('train-images.idx3-ubyte')
train_labels_path = Path('train-labels.idx1-ubyte')
test_images_path = Path('t10k-images.idx3-ubyte')
test_lables_path = Path('t10k-labels.idx1-ubyte')

def load_mnist_data(train_images_path, train_labels_path, test_images_path, test_lables_path):
    digit_train, digit_label = loadlocal_mnist(images_path=train_images_path,
                                               labels_path=train_labels_path)
    test_digit, test_label = loadlocal_mnist(images_path=test_images_path,
                                             labels_path=test_lables_path)
    return digit_train, digit_label, test_digit, test_label

def train_digit_model(digit_train, digit_label):
	digit_recognition_model = RandomForestClassifier(n_estimators=100)
	digit_recognition_model.fit(digit_train, digit_label)
	return digit_recognition_model

def get_digit_prediction(test_digit):
	return digit_recognition_model.predict(test_digit)

digit_train, digit_label, test_digit, test_label = load_mnist_data(train_images_path, train_labels_path, test_images_path, test_lables_path)

digit_recognition_model = train_digit_model(digit_train, digit_label)
predictions = get_digit_prediction(test_digit)

print("Accuracy of the digit_recognition_model: ", accuracy_score(test_label, predictions))
print("RMSE of the digit_recognition_model: ", mean_squared_error(test_label, predictions))

joblib.dump(digit_recognition_model, 'digit_recognition_model.sav')