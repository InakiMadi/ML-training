# Data Analysis and Wrangling
import pandas as pd
# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor

# Pandas print options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Auto-wrap columns
pd.set_option('display.max_colwidth', 20)  # Show full content of each column

# Better ending line for prints
separation = "\n" + "-" * 50 + "\n"

def sklearn_model(X_train,X_test,y_train,y_test):
    '''
        Modelling.
    '''
    # For other examples, maybe we need to make sure there is no strings (object).
    ## print(X_train.dtypes)

    '''
        Logistic regression.
    '''
    logreg = LinearRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    mae_log = mean_absolute_error(y_test, y_pred)
    r2_log = r2_score(y_test, y_pred)
    acc_log = r2_log
    ## print(f"Logistic regression accuracy: {acc_log}.", end=separation)

    # Validate feature selection using logistic regression.
    ## coeff_df = pd.DataFrame(dataframe.columns.delete(0))
    ## coeff_df.columns = ['Feature']
    ## coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
    ## print(coeff_df.sort_values(by='Correlation', ascending=False),end=separation)

    '''
        SVM - Support Vector Machine.
    '''
    svc = SVR()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    acc_svc = r2_score(y_test, y_pred)
    ## print(f"SVM accuracy: {acc_svc}.", end=separation)

    '''
        kNN - k-Nearest Neighbor
    '''
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc_knn = r2_score(y_test, y_pred)
    ## print(f"kNN accuracy: {acc_knn}.", end=separation)

    '''
        Decision Tree
    '''
    decision_tree = DecisionTreeRegressor(random_state=1)
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    mae_decision_tree = mean_absolute_error(y_test, y_pred)
    r2_decision_tree = r2_score(y_test, y_pred)
    acc_decision_tree = r2_decision_tree
    ## print(f"Decision Tree accuracy: {acc_decision_tree}.", end=separation)

    '''
        Random Forest.
    '''
    random_forest = RandomForestRegressor(n_estimators=100)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    acc_random_forest = r2_score(y_test, y_pred)
    ## print(f"Random Forest accuracy: {acc_random_forest}.", end=separation)

    '''
        No Gaussian Naive Bayes.
    '''

    '''
        Stochastic Gradient Descent.
    '''
    sgd = SGDRegressor()
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    acc_sgd = r2_score(y_test, y_pred)

    '''
        Perceptron - PassiveAgressiveRegressor.
    '''

    '''
        Linear SVC.
    '''
    linear_svc = LinearSVR()
    linear_svc.fit(X_train, y_train)
    y_pred = linear_svc.predict(X_test)
    acc_linear_svc = r2_score(y_test, y_pred)

    # Summary.
    models = pd.DataFrame({
        'Name': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                  'Random Forest',
                  'Stochastic Gradient Decent', 'Linear SVC',
                  'Decision Tree'],
        'Score': [acc_svc, acc_knn, acc_log,
                  acc_random_forest,
                  acc_sgd, acc_linear_svc, acc_decision_tree],
        'Model': [svc,knn,logreg,random_forest,sgd,linear_svc,decision_tree]
    })
    models = models.sort_values(by='Score', ascending=False)
    print(models, end=separation)
    best_model = models.iloc[0]['Model']
    print(f"Best model that fits the data: {best_model}.", end=separation)

    '''
        Conclusion: We select the best model.
    '''
    return best_model