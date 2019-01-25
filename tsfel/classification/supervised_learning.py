from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score


def find_best_slclassifier(features, labels, X_train, X_test, y_train, y_test):
    """ This function performs the classification of the given features using several classifiers. From the obtained results
    the classifier which best fits the data and gives the best result is chosen and the respective confusion matrix is
    showed.
    Parameters
    ----------
    X_train: array-like
      train set features
    X_test: array-like
      test set features
    y_train: array-like
      train set labels
    y_test: array-like
      test set labels
    y_test: array-like
      test set labels
    features: array-like
      set of features
    labels: array-like
      features class labels
    Returns
    -------
    c: best classifier
    """

    # Classifiers
    names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "AdaBoost", "Naive Bayes", "QDA"]
    classifiers = [
        KNeighborsClassifier(5),
        DecisionTreeClassifier(max_depth=5, min_samples_split=len(features)//10),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    best = 0
    best_classifier = None

    for n, c in zip(names, classifiers):
        print(n)
        scores = cross_val_score(c, features, labels, cv=10)
        print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

        # Train the classifier
        c.fit(X_train, y_train.ravel())

        # Predict test data
        y_test_predict = c.predict(X_test)

        # Get the classification accuracy
        accuracy = accuracy_score(y_test, y_test_predict)*100
        print("Accuracy: " + str(accuracy) + '%')
        print('-----------------------------------------')
        if np.mean([scores.mean(), accuracy]) > best:
            best_classifier = n
            best = np.mean([scores.mean(), accuracy])

    print('******** Best Classifier: ' + str(best_classifier) + ' ********')

    return c
