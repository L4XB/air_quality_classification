import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.src.model import Model

model = Model()

dataset = pd.read_csv("model/data/updated_pollution_dataset.csv")

# Good -> 0
# Moderate -> 1
# Poor -> 2

dataset["Air Quality"] = dataset["Air Quality"].replace("Good", 0)
dataset["Air Quality"] = dataset["Air Quality"].replace("Moderate", 1)
dataset["Air Quality"] = dataset["Air Quality"].replace("Poor", 2)

# Überprüfen und Bereinigen der Daten
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset = dataset.dropna()

X = dataset.drop("Air Quality", axis=1).values
y = dataset["Air Quality"].values

# Normalisierung der Daten
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

epochs = 150  
losses = []
val_losses = []

for i in range(epochs):
    model.train()  
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validierung
    model.eval() 
    with torch.no_grad():  
        y_pred_val = model.forward(X_test)
        val_loss = criterion(y_pred_val, y_test)
        val_losses.append(val_loss.detach().numpy())
    
    print(f"Epoch: {i}, Training Loss: {loss}, Validation Loss: {val_loss}")

plt.plot(range(epochs), losses, label='Training Loss')
plt.plot(range(epochs), val_losses, label='Validation Loss')
plt.ylabel("Loss/Error")
plt.xlabel('Epoch')
plt.legend()
plt.show()


model.eval()  
with torch.no_grad(): 
    y_pred_test = model.forward(X_test)
    y_pred_test = torch.argmax(y_pred_test, dim=1)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")