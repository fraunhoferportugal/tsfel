import numpy as np
import matplotlib
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, GroupKFold, StratifiedKFold
import multiprocessing
import time
# from src2.custom_svm import CustomSVC as cSVC
import itertools
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix as skcm


def cross_val_strategy(cv, groups, X, y):
    """
    Configures the cross-validation strategy.
    :param cv: Number of folds.
    :param groups: Group that each sample belongs to.
    :param X: Features of each sample.
    :param y: Targets of each sample.
    :return: Generator of training and validation sets accoding to the customized strategy.
    """

    RAND_STATE = 42
    assert((cv is not None) or (groups is not None))

    # Configure validation method according to parameters
    if groups is not None:

        # Default cv to None if cv > number of groups
        if cv is not None:
            cv = cv if cv < len(np.unique(groups)) else None

        # Leave-One-Group-Out validation
        if cv is None:
            splitter = LeaveOneGroupOut()
            cv = splitter.split(X, y, groups)
            print('Number of splits: ' + str(splitter.get_n_splits(X, y, groups)))
        # Cross-validation stratified by 'groups' instead of labels
        else:
            splitter = GroupKFold(cv)
            cv = splitter.split(X, y, groups)
            print('Number of splits: ' + str(splitter.get_n_splits(X, y, groups)))
    # Stratified cross-validation
    elif isinstance(cv, int):
        splitter = StratifiedKFold(cv, random_state=RAND_STATE)
        cv = splitter.split(X, y, groups)
        print('Number of splits: ' + str(str(splitter.get_n_splits(X, y, groups))))

    return cv


def forward_feature_selection(clf, X, y, groups=None, cv=10, n_jobs=-1, feature_names=None):
    """
    Selects the best features using the Sequential Forward Feature Selection algorithm.
    :param clf: Classifier.
    :param X: Matrix of shape (n_samples, n_features).
    :param y: Targets of each sample.
    :param groups: Group of each sample, for splitting in cross-validation.
    :param cv: Number of folds for cross validation. Defaults to None if 'groups' != None and cv > number of groups.
    :param n_jobs: Number of threads to use. Uses all available if -1.
    :return: List with the arguments of each selected feature.
    """

    print('Selecting the best features...')

    if feature_names is None:
        feature_names = np.arange(X.shape[1])
    else:
        feature_names = np.array(feature_names)

    # Configure the cross-validation strategy
    splits = cross_val_strategy(cv, groups, X, y)
    cv = [s for s in splits]

    # If the svm's probability == True, disable it during feature selection to speed it up
    probability = False
    #if isinstance(clf, svm.SVC) or isinstance(clf, cSVC):
    #    if clf.probability is True:
    #        probability = True
    #        clf.set_params(probability=False)

    # Starting time
    t_start = time.time()

    # Previous accuracy
    acc = -1

    # Current accuracy
    new_acc = 0

    # Current subset of best features
    current_features = np.empty((len(y), 0))

    # Current arguments of best features
    best_features_args = list()

    # Use all cores if n_jobs == -1
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    # Copy features array
    all_feat = X.copy()
    all_feat_names = feature_names.copy()

    # While accuracy improves
    while new_acc > acc:
        # Update previous accuracy
        acc = new_acc

        # Arguments for cross validation
        cv_args = [(clf, np.hstack((current_features, feat.T.reshape(-1, 1))), y, None, None, cv) for feat in all_feat.T]

        # Cross validation of current_features hstack with each feature (multi-core parallel)
        with multiprocessing.Pool(n_jobs) as pool:
            cv_scores = pool.starmap(cross_val_score, cv_args)

        # Accuracy computation for each cross validation result
        accs = [np.mean(cvs) for cvs in cv_scores]

        # Argument of chosen feature
        best_f = np.argmax(accs)

        # Update current accuracy
        new_acc = accs[best_f]
        new_std = np.std(cv_scores[best_f])
        if new_acc > acc:
            # Add new feature to current_features
            current_features = np.hstack((current_features, all_feat[:, best_f].reshape(-1, 1)))
            # Add argument of new feature to best_features
            best_features_args.append(np.where(feature_names == all_feat_names[best_f])[0][0])
            # Print current accuracy and best feature's arguments
            print('Accuracy: ' + str(np.round(new_acc * 100, 2)) + ' +/- ' +
                  str(np.round(new_std * 100, 2)), feature_names[best_features_args])
            # Remove best feature from all_feat
            all_feat = np.delete(all_feat, best_f, axis=1)
            all_feat_names = np.delete(all_feat_names, best_f, axis=0)

    # Return the svm's probability attribute to its previous state
    if probability is True:
        clf.set_params(probability=True)

    # Inform that feature selection is finished and display elapsed time
    print('Done!')
    print('Time elapsed: ' + str(np.round(time.time() - t_start, 1)) + 's\n')

    return best_features_args


def correlation_report(df):
    """ Performs a correlation report and removes highly correlated features.
    Parameters
    ----------
    df: dataframe
      features
    Returns
    -------
    df: feature dataframe without high correlated features
    """
    # TODO use another package
    # To correct a bug in pandas_profiling package
    BACKEND = matplotlib.get_backend()
    import pandas_profiling
    matplotlib.use(BACKEND)

    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(outputfile="CorrelationReport.html")
    inp = str(input('Do you wish to remove correlated features? Enter y/n: '))
    if inp == 'y':
        reject = profile.get_rejected_variables(threshold=0.9)
        if not list(reject):
            print('No features to remove')
        for rej in reject:
            print('Removing ' + str(rej))
            df = df.drop(rej, axis=1)
    return df