# Data Analysis and Wrangling
import pandas as pd
# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Pandas print options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Auto-wrap columns
pd.set_option('display.max_colwidth', 20)  # Show full content of each column

# Better ending line for prints
separation = "\n" + "-" * 50 + "\n"

def model(X_train,X_test,y_train,y_test):
    '''
        Modelling.
    '''
    # For other examples, maybe we need to make sure there is no strings (object).
    ## print(X_train.dtypes)

    '''
        Logistic regression.
    '''
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    acc_log = round(accuracy_score(y_test, y_pred) * 100, 2)
    ## print(f"Logistic regression accuracy: {acc_log}.", end=separation)

    # Validate feature selection using logistic regression.
    ## coeff_df = pd.DataFrame(dataframe.columns.delete(0))
    ## coeff_df.columns = ['Feature']
    ## coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
    ## print(coeff_df.sort_values(by='Correlation', ascending=False),end=separation)

    '''
        SVM - Support Vector Machine.
    '''
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    acc_svc = round(accuracy_score(y_test, y_pred) * 100, 2)
    ## print(f"SVM accuracy: {acc_svc}.", end=separation)

    '''
        kNN - k-Nearest Neighbor
    '''
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc_knn = round(accuracy_score(y_test, y_pred) * 100, 2)
    ## print(f"kNN accuracy: {acc_knn}.", end=separation)

    '''
        Decision Tree
    '''
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(accuracy_score(y_test, y_pred) * 100, 2)
    ## print(f"Decision Tree accuracy: {acc_decision_tree}.", end=separation)

    '''
        Random Forest.
    '''
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    acc_random_forest = round(accuracy_score(y_test, y_pred) * 100, 2)
    ## print(f"Random Forest accuracy: {acc_random_forest}.", end=separation)

    '''
        Gaussian Naive Bayes.
    '''
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    y_pred = gaussian.predict(X_test)
    acc_gaussian = round(accuracy_score(y_test, y_pred) * 100, 2)

    '''
        Stochastic Gradient Descent.
    '''
    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    acc_sgd = round(accuracy_score(y_test, y_pred) * 100, 2)

    '''
        Perceptron.
    '''
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    acc_perceptron = round(accuracy_score(y_test, y_pred) * 100, 2)

    '''
        Linear SVC.
    '''
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, y_train)
    y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(accuracy_score(y_test, y_pred) * 100, 2)

    # Summary.
    models = pd.DataFrame({
        'Name': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                  'Random Forest', 'Naive Bayes', 'Perceptron',
                  'Stochastic Gradient Decent', 'Linear SVC',
                  'Decision Tree'],
        'Score': [acc_svc, acc_knn, acc_log,
                  acc_random_forest, acc_gaussian, acc_perceptron,
                  acc_sgd, acc_linear_svc, acc_decision_tree],
        'Model': [svc,knn,logreg,random_forest,gaussian,perceptron,sgd,linear_svc,decision_tree]
    })
    models = models.sort_values(by='Score', ascending=False)
    print(models, end=separation)
    best_model = models.iloc[0]['Model']
    print(f"Best model that fits the data: {best_model}.", end=separation)

    '''
        Conclusion: We select the best model.
    '''
    return best_model