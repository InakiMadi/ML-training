# Data Analysis and Wrangling
import pandas as pd
import numpy as np
# Machine Learning
from sklearn.metrics import accuracy_score, r2_score
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


def nn_model(X_train, X_test, y_train, y_test):
    '''
        Modelling.
    '''

    # features = X_train.iloc[:,2:].columns.tolist()
    features = X_train.columns.values.tolist()
    target = 'SalePrice'
    # Transform from Pandas to lists
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            num_features_in = X_train.shape[1]  # Dynamic input size
            num_features_fc2 = num_features_in * 64

            self.fc1 = nn.Linear(num_features_in, num_features_fc2)
            self.fc2 = nn.Linear(num_features_fc2, num_features_fc2)
            self.fc3 = nn.Linear(num_features_fc2, 1)
            self.dropout = nn.Dropout(0.2)

            # Proper weight initialization
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)

    # Initialize
    model = Net()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Better than SGD

    # Training Loop
    batch_size = 64
    n_epochs = 500
    train_loss_min = np.inf

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        # Shuffle data each epoch
        permutation = torch.randperm(len(X_train))

        for i in range(0, len(X_train), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X = torch.FloatTensor(X_train[indices])
            batch_y = torch.FloatTensor(y_train[indices])

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(X_train) / batch_size)

        # Validation
        if avg_loss < train_loss_min:
            print(f"Epoch {epoch + 1}: Loss improved {train_loss_min:.6f} -> {avg_loss:.6f}")
            train_loss_min = avg_loss
            torch.save(model.state_dict(), "model.pt")

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.FloatTensor(X_test)).squeeze()
        r2 = r2_score(y_test, test_pred)
        print(f"Final R² Score: {r2:.4f}")

    return model


def validate(model, validation_df):
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
