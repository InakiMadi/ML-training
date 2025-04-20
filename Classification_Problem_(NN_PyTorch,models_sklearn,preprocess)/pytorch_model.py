# Data Analysis and Wrangling
import pandas as pd
import numpy as np
# Machine Learning
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

# Pandas print options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Auto-wrap columns
pd.set_option('display.max_colwidth', 20)  # Show full content of each column

# Better ending line for prints
separation = "\n" + "-" * 50 + "\n"

def nn_model(X_train,X_test,y_train,y_test):
    '''
        Modelling.
    '''

    # features = X_train.iloc[:,2:].columns.tolist()
    features = X_train.columns.values.tolist()
    target = 'Survived'
    # Transform from Pandas to lists
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            num_features_in = len(features)
            num_features_fc2 = num_features_in * 64 # = 512
            num_features_clasif = 2
            self.fc1 = nn.Linear(num_features_in, num_features_fc2) # Input layer: 8 features → 512 neurons
            self.fc2 = nn.Linear(num_features_fc2, num_features_fc2) # Hidden layer: 512 → 512 neurons
            self.fc3 = nn.Linear(num_features_fc2, num_features_clasif) # Output layer: 512 → 2 neurons (binary clasif.)
            self.dropout = nn.Dropout(0.2) # Regularization: randomly zero 20% of inputs (prevents overfitting)

        def forward(self, x):
            x = F.relu(self.fc1(x)) # Activation after 1st layer. 8 -> 512. ReLU (max(0,x)) introduces non-linearity.
            x = self.dropout(x)     # Apply dropout. Randomly masks 20% of the 512 outputs during training.
            x = F.relu(self.fc2(x)) # Activation after 2nd layer. 512 -> 512. ReLU introduces non-linearity.
            x = self.dropout(x)     # Apply dropout. Randomly masks 20% of the 512 outputs during training.
            x = self.fc3(x)         # Final output (no activation). 512 -> 2.
            return x

    '''
        Training.
    '''
    model = Net()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    batch_size = 64
    n_epochs = 500
    batch_no = len(X_train) // batch_size

    train_loss = 0
    train_loss_min = np.Inf
    for epoch in range(n_epochs):
        # Batch Processing
        for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size
            x_var = torch.FloatTensor(X_train[start:end]) # Features
            y_var = torch.LongTensor(y_train[start:end]) # Labels

            # Forward-Backward Pass
            optimizer.zero_grad()
            output = model(x_var)
            loss = criterion(output, y_var)
            loss.backward()
            optimizer.step()

            # Accuracy calculation
            values, labels = torch.max(output, 1) # Prediction. 0 or 1.
            num_right = np.sum(labels.data.numpy() == y_train[start:end]) # Count correct predictions.

            # Loss accumulation
            train_loss += loss.item() * batch_size # Accumulate scaled loss.

        # End of batch.
        train_loss = train_loss / len(X_train) # Normalize.
        # Model checkpoint. Saves model only when validation loss improves.
        if train_loss <= train_loss_min:
            print(
                "Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min, train_loss)
            )
            torch.save(model.state_dict(), "model.pt") # Saves best weights.
            train_loss_min = train_loss
        # Log
        if epoch % 200 == 0:
            print('')
            print("Epoch: {} \tTrain Loss: {} \tTrain Accuracy: {}".format(epoch + 1, train_loss, num_right / len(y_train[start:end])))
    print('Training Ended! ')

    '''
        Predicting.
    '''
    X_test_var = torch.FloatTensor(X_test)
    with torch.no_grad():
        test_result = model(X_test_var)
    values, labels = torch.max(test_result, dim=1)
    y_pred = labels.data.numpy()
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    print(f'Model accuracy: {accuracy}.',end=separation)

    return model

def validate(model,validation_df):
    '''
        Validation.
    '''
    X_test = validation_df.values
    X_test_var = torch.FloatTensor(X_test)
    with torch.no_grad():
        test_result = model(X_test_var)
    values, labels = torch.max(test_result, 1)
    y_pred = labels.data.numpy()

    return y_pred