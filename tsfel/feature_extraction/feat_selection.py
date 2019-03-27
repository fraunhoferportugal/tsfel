import numpy as np
import matplotlib
from sklearn.metrics import accuracy_score


def FSE(X_train, X_test, y_train, y_test, features_descrition, classifier):
    """ Performs a forward feature selection.
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
    features_descrition: list of strings
      list with extracted features names
    classifier: object
      classifier object
    Returns
    -------
    FS_X_train: train set best set of features
    FS_X_test: test set best set of features
    FS_lab: name of the best set of features
    """
    total_acc, FS_lab, acc_list = [], [], []
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    print("*** Feature selection started ***")
    for feat_idx, feat_name in enumerate(features_descrition):
        classifier.fit(X_train[:,feat_idx].reshape(-1,1), y_train)
        y_test_predict = classifier.predict(X_test[:,feat_idx].reshape(-1,1))
        acc_list.append(accuracy_score(y_test, y_test_predict))

    curr_acc_idx = np.argmax(acc_list)
    FS_lab.append(features_descrition[curr_acc_idx])
    last_acc = acc_list[curr_acc_idx]
    FS_X_train = X_train[:,curr_acc_idx]
    FS_X_test = X_test[:,curr_acc_idx]
    total_acc.append(last_acc)

    while 1:
        acc_list = []
        for feat_idx, feat_name in enumerate(features_descrition):
            if feat_name not in FS_lab:
                curr_train = np.column_stack((FS_X_train, X_train[:, feat_idx]))
                curr_test = np.column_stack((FS_X_test, X_test[:, feat_idx]))
                classifier.fit(curr_train, y_train)
                y_test_predict = classifier.predict(curr_test)
                acc_list.append(accuracy_score(y_test, y_test_predict))
            else:
                acc_list.append(0)
        curr_acc_idx = np.argmax(acc_list)
        if last_acc < acc_list[curr_acc_idx]:
            FS_lab.append(features_descrition[curr_acc_idx])
            last_acc = acc_list[curr_acc_idx]
            total_acc.append(last_acc)

            FS_X_train = np.column_stack((FS_X_train, X_train[:, curr_acc_idx]))
            FS_X_test = np.column_stack((FS_X_test, X_test[:, curr_acc_idx]))
        else:
            print("FINAL Features: " + str(FS_lab))
            print("Number of features", len(FS_lab))
            print("Acc: ", str(total_acc))
            print("From ", str(len(X_train[0])), "to ", str(len(FS_lab)))

            break
    print("*** Feature selection finished ***")

    return np.array(FS_X_train), np.array(FS_X_test), np.array(FS_lab)


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