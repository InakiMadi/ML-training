# Data Analysis and Wrangling
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Pandas print options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Auto-wrap columns
pd.set_option('display.max_colwidth', 20)  # Show full content of each column

# Better ending line for prints
separation = "\n" + "-" * 50 + "\n"

# Initialize the encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

def preprocess(dataframe,validation_df,combine):
    '''
        Columns names and data info
    '''
    ## print(dataframe.columns.values,end=separation)
    ## print(dataframe.head(7), end=separation)
    ## print(dataframe['Name'][1] + "\n" + dataframe['Name'][2], end=separation)
    # Categorical: Survived, Sex, Embarked.
    # Ordinal: Pclass.
    # Continuous: Age, Fare.
    # Discrete: SibSp, Parch.
    # Alphanumeric: Cabin, Name.
    # Mixed: Ticket (alphanumeric and numeric).

    ## print(dataframe.info(), end=separation)
    # Sample size: 891 entries.
    # Embarked, Age, Cabin have null entries.
    # Seven features are int or float. Five are strings (objects).

    ## print(dataframe.describe(),end=separation)
    # (min 25% 50% 75% max - Survived) Survived is a categorical feature (binary).
    # (min 25 50 75 max - Pclass) Pclass has only values: 1, 2 or 3.
    # (mean - Survived) Around 38% survived in the sample.
    # (25 50 75% - Parch) Most passengers (>75%) didn't travel with parents or children.
    # (25 50 75% - SibSp) Less than half of passengers had siblings and/or spouse.
    # Fares varied significantly. Percentile 25, 50, 75 paid $31 or less; although the max someone paid was $512.
    # (min - Age) There were babies.
    # (max - Age) The oldest person was 80 years old.
    # (25 50 75% - Age) Most passengers (75% at least) were of age between 20-38.
    # (75% max - Age) Less than 25% could have been elder people.

    ## print(dataframe.describe(include=['object']),end=separation)
    # Only 2 Sex. The most frequent: male. Frequency: 577 out of 891.
    # Ticket has many repetitions.
    # Cabin has repetitions.
    # Embarked has 3 possible values.

    '''
        Pivoting features. Analysis.
    '''
    # Assumptions and observations:
    # Maybe gender affects Survived (female positively, male negatively?).
    # Maybe Pclass affects Survived (negatively?).
    # Maybe children affects Survived (positively?).

    ## print(dataframe[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False),end=separation)
    # Lower Pclass, bigger % Survived. Seems linear.

    ## print(dataframe[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False),end=separation)
    # Female: 0.74 - Male: 0.19. Female people has very high survival rate.

    ## print(dataframe[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False), end=separation)
    # Doesn't seem to have correlation. Maybe removing SibSp=0, but hard to imagine relationship.

    ## print(dataframe[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False), end = separation)
    # Doesn't seem to have correlation.

    '''
        Visualizing data. Analysis.
    '''
    ## grid = sns.FacetGrid(dataframe, col='Survived')
    ## grid.map(plt.hist, 'Age', bins=20)
    ## grid.set_axis_labels('Age', 'Count')
    ## plt.tight_layout()
    ## plt.show()
    # From 15-25, more people died than survived.
    # Enfants, more survived.
    # Oldest passengers (age=80) survived.

    ## grid = sns.FacetGrid(dataframe, col='Survived')
    ## grid.map(plt.hist, 'Pclass', bins=20)
    ## grid.set_axis_labels('Pclass', 'Count')
    ## plt.tight_layout()
    ## plt.show()
    # Pclass=3, 3x people died than survived.
    # Pclass=2, almost same proportion survived and died.
    # Pclass=1, more people survived.

    # Combination: Survived depending on Pclass and Age.
    ## grid = sns.FacetGrid(dataframe, col='Survived', row='Pclass')
    ## grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    ## grid.add_legend()
    ## plt.tight_layout()
    ## plt.show()
    # Enfants survived in all 3 classes. Although for almost 1 years old or more in Pclass=3, more died than survived.
    # For Pclass=1, more 35-40 years old people survived.
    # Decision: Add Pclass to model training.

    # Combination: Survived depending on Embarked and Sex.
    ## grid = sns.FacetGrid(dataframe, row='Embarked')
    ## grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    ## grid.add_legend()
    ## plt.tight_layout()
    ## plt.show()
    # Interesting. If Embarked=C, more male survived compared to female.
    # Decision: Add Sex, complete and add Embarked.

    '''
        Creating new feature from existing.
        Dropping features. Correcting.
        One-hot encoding.
    '''

    # Get Titles from Names
    for df in combine:
        df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    ## print(pd.crosstab(dataframe['Title'], dataframe['Sex']),end=separation)

    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
            'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    ## print(dataframe[['Title', 'Survived']].groupby(['Title'], as_index=False).mean(),end=separation)

    # One-hot encoding.# Initialize the encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    for index,dataset in enumerate(combine):
        # Reshape the 'Title' column for sklearn (requires 2D input)
        titles = dataset[['Title']]

        # Fit and transform (for train) or just transform (for test)
        if index == 0:
            one_hot = encoder.fit_transform(titles)
        else:
            one_hot = encoder.transform(titles)

        # Create column names (e.g., "Title_Mr", "Title_Miss")
        categories = encoder.categories_[0]
        one_hot_cols = [f'Title_{cat}' for cat in categories]

        # Add one-hot columns to DataFrame
        one_hot_df = pd.DataFrame(one_hot, columns=one_hot_cols, index=dataset.index)
        combine[index] = pd.concat([dataset, one_hot_df], axis=1)

        #Drop the original 'Title' column
        combine[index] = combine[index].drop('Title', axis=1)
    dataframe, validation_df = combine

    # Dropping features
    dataframe = dataframe.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    validation_df = validation_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    combine = [dataframe, validation_df]

    '''
        Completing features. Converting.
    '''

    # Categorical features
    for index, dataset in enumerate(combine):
        dataset['Female'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
        combine[index] = combine[index].drop('Sex', axis=1)
    dataframe, validation_df = combine

    freq_port = dataframe.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    ## print(dataframe[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',ascending=False), end=separation)

    # Completing continuous feature
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                rows_sex = dataset['Female'] == i
                rows_pclass = dataset['Pclass'] == j + 1
                rows_mask = rows_sex & rows_pclass
                guess_df = dataset[rows_mask]['Age'].dropna()

                # Fill the gaps with the median
                age_guess = guess_df.median()
                # Convert random age float to nearest .5 age
                dataset.loc[rows_mask & dataset['Age'].isnull(), 'Age'] = int(round(age_guess))

        dataset['Age'] = dataset['Age'].astype(int)

    for dataset in combine:
        dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].dropna().median())
        dataset['FareBand'] = pd.qcut(dataset['Fare'], 4)

    ## print(dataframe[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True),end=separation)
    # We see a clear correlation between FareBands and Survived. We must add it, better as an ordinal feature.

    for index,dataset in enumerate(combine):
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
        combine[index] = combine[index].drop('FareBand', axis=1)
    dataframe, validation_df = combine

    ## print(dataframe[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Fare', ascending=True),end=separation)

    '''
        Converting from numeric to One-hot encoding.
    '''
    # In 5 bands:
    for dataset in combine:
        dataset['AgeBand'] = pd.cut(dataset['Age'], 5)
    ## print(dataframe[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True),end=separation)
    # We realize that each AgeBand is between 16 years.

    # Initialize encoder
    age_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    for index, dataset in enumerate(combine):
        # Extract AgeBand as 2D array (required by sklearn)
        age_bands = dataset[['AgeBand']]

        # Fit encoder on training data (index=0), transform on both
        if index == 0:
            one_hot = age_encoder.fit_transform(age_bands)
        else:
            one_hot = age_encoder.transform(age_bands)

        # Create column names (e.g., "AgeBand_(16,32]")
        band_labels = [str(band) for band in age_encoder.categories_[0]]
        one_hot_cols = [f'AgeBand_{band}' for band in band_labels]

        # Add one-hot columns to DataFrame
        one_hot_df = pd.DataFrame(one_hot, columns=one_hot_cols, index=dataset.index)
        combine[index] = pd.concat([dataset, one_hot_df], axis=1)

        # Drop original columns
        combine[index].drop(['Age', 'AgeBand'], axis=1, inplace=True)
    dataframe, validation_df = combine

    '''
        Converting from categorical to ordinal.
    '''
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    ## print(dataframe.head(),end=separation)

    '''
        Create feature combining features.
    '''
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    ## print(dataframe[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False),end=separation)
    # We don't see a correlation. But we can at least create if someone was travelling with family or not.

    for index,dataset in enumerate(combine):
        dataset['HasFamily'] = 0
        dataset.loc[dataset['FamilySize'] > 1, 'HasFamily'] = 1
        dataset['HasFamily'] = dataset['HasFamily'].astype(int)
        combine[index] = combine[index].drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    dataframe, validation_df = combine
    ## print(dataframe[['HasFamily', 'Survived']].groupby(['HasFamily'], as_index=False).mean(),end=separation)

    '''
        Creating artificial feature combining two numeric features.

    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
    ## print(dataframe.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))
    '''

    return dataframe, validation_df, combine