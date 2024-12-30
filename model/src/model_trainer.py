import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from model.src.model import Model

model = Model()

dataset = pd.read_csv("model/data/updated_pollution_dataset.csv")


# Good -> 0
# Moderate -> 1
# Poor -> 2

dataset["Air Quality"] = dataset["Air Quality"].replace("Good", 0)
dataset["Air Quality"] = dataset["Air Quality"].replace("Moderate", 1)
dataset["Air Quality"] = dataset["Air Quality"].replace("Poor", 2)


X = dataset.drop("Air Quality", axis = 1)
y = dataset["Air Quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
