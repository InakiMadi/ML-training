from preprocess import preprocess
from pytorch_model import nn_model, validate
# Data Analysis and Wrangling
import pandas as pd
# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Pandas print options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Auto-wrap columns
pd.set_option('display.max_colwidth', 20)  # Show full content of each column

# Better ending line for prints
separation = "\n" + "-" * 50 + "\n"

''' 
    Data acquisition 
'''
dataframe = pd.read_csv("./input/titanic/train.csv")
validation_df = pd.read_csv("./input/titanic/test.csv")
validation_PassengerId = validation_df['PassengerId']
combine = [dataframe, validation_df]

'''
    Preprocess data.
'''
dataframe, validation_df, combine = preprocess(dataframe,validation_df,combine)

'''
    Train test split.
    Pclass distribution is important. (stratify)
'''
X = dataframe.drop(['Survived'], axis=1)
y = dataframe['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=dataframe['Pclass'],
    random_state=0
)

'''
###
#   Adding preprocessing with respect to sklearn: Scalar standardization.
###


scaler = StandardScaler()

train_columns = X_train.columns
test_columns = X_test.columns

X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

X_train.columns = train_columns
X_test.columns = test_columns
'''


'''
    Modelling.
'''
# We get the model that fits best the data, from a selection of models.
model = nn_model(X_train,X_test,y_train,y_test)
y_validation_pred = validate(model,validation_df)

'''
    Conclusion.
'''

submission = pd.DataFrame({
    'PassengerId': validation_PassengerId,
    'Survived': y_validation_pred
})
submission.to_csv('NN_submission.csv', index=False)
print("(NN) Submission saved!")

'''
    Possible future work for Preprocess:
    Getting the Deck from Cabin's first letter (however, many are null).
'''
