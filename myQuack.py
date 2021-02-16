
'''

2020

Scaffolding code for the Machine Learning assignment.

You should complete the provided functions and add more functions and classes as necessary.

You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import tree, svm, neighbors, neural_network

from sklearn.model_selection import GridSearchCV,\
    train_test_split,\
        cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, \
    confusion_matrix, \
        plot_confusion_matrix, \
            precision_score, \
                recall_score

import seaborn as sns

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def my_team():
    '''
    Returns
    -------
    list
        Returns author information; Student ID, Student Name.

    '''
    return [(9997121, 'Harry', 'Akeroyd')]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def prepare_dataset(dataset_path):
    '''
    Preprocessing of the data - medical_records.data is used. Layout of the
    data;
        Column 0 - Patient ID - Not neccessary for program.
        Column 1 - Tumor Diagnosis Classification; Malignant or Benging.
        Columns 2-32 - Ten real-valued features are computed for each cell
                       nucleus.

    Parameters
    ----------
    dataset_path : str
        Location of data for preprocessing.

    Returns
    -------
    X : float
        Response variables - refering to the medical assessment for each
        patient.
    y : int
        Feature variables - refering to the tumor classification.
        Malignant or Benign.

    '''
    # PREPROCESSING
    ft_variables = np.genfromtxt(dataset_path, delimiter=',', dtype=str, \
                                 usecols=range(2,31))
    rs_variables = np.genfromtxt(dataset_path, delimiter=',', dtype=str, \
                                 usecols=1)

    # Response variable contraints
    malignant = 1
    benign = 0

    # Processing response variables
    rs_malignant = np.argwhere(rs_variables=='M') # Searching data for M & B
    rs_benign = np.argwhere(rs_variables=='B')    # classification of data

    # Assigning indexing
    rs_variables[rs_malignant] = malignant        # Assigning standardized
    rs_variables[rs_benign] = benign              # data: M=1, B=0


    # Seperating the dataset as feature variable and response variable
    X = np.array(ft_variables, dtype=float)
    y = np.array(rs_variables, dtype=int)

    return X, y
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def build_DecisionTree_classifier(X_training, y_training):
    '''

    Parameters
    ----------
    X_training : float
        Feature variables representing the data provided, all values are float
        and are standardized. Training data must also be split.
    y_training : int
        Response variables representing the tumor diagnosis for testing.

    Returns
    -------
    DT_clf : Class
        Returns the Decision Tree Classifier thats fittted to the input data.

    '''
    # Hyperparameter assignment and Decision Tree
    depth = list(range(1,101,1))
    param = {'max_depth': depth}
    DT_clf = tree.DecisionTreeClassifier(random_state=42)

    # Optimization of Hyperparameter - MAX DEPTH
    DT_clf = GridSearchCV(DT_clf, param)

    # Fit model using training data
    DT_clf.fit(X_training, y_training)

    # Re-train using the optimized hyperparameter
    best_depth_param = DT_clf.best_params_['max_depth']
    DT_clf = tree.DecisionTreeClassifier(max_depth=best_depth_param,\
                                         random_state=42)
    DT_clf.fit(X_training, y_training)

    return DT_clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def build_NearrestNeighbours_classifier(X_training, y_training):
    '''

    Parameters
    ----------
    X_training : float
        Feature variables representing the data provided, all values are float
        and are standardized. Training data must also be split.
    y_training : int

    Returns
    -------
    KNN_clf : Class
        Returns the Nearrest Neighbour classifier thats fittted to the input
        data.

    '''
    # Hyperparameter assignment and Nearest Neighbours
    neighbours = list(range(1,11,1))
    param = {'n_neighbors': neighbours}
    KNN_clf = neighbors.KNeighborsClassifier()

    # Optimization of Hyperparameter - NEAREST NEIGHBOUR
    KNN_clf = GridSearchCV(KNN_clf, param)

    # Fit model using training data
    KNN_clf.fit(X_training, y_training)

    # Re-train using the optimized hyperparameter
    best_n_param = KNN_clf.best_params_['n_neighbors']
    KNN_clf = neighbors.KNeighborsClassifier(n_neighbors=best_n_param)
    KNN_clf.fit(X_training, y_training)

    return KNN_clf
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def build_SupportVectorMachine_classifier(X_training, y_training):
    '''

    Parameters
    ----------
    X_training : float
        Feature variables representing the data provided, all values are float
        and are standardized. Training data must also be split.
    y_training : int

    Returns
    -------
    SVM_clf : Class
        Returns the Support Vector Machine thats fittted to the input data.

    '''
    # Hyperparameter assignment and Support Vector Machine
    C_og = list(range(-51,51,1))
    param = {'C': C_og}
    SVM_clf = svm.SVC(kernel='linear', random_state=42)

    # Optimization of Hyperparameter - C
    SVM_clf = GridSearchCV(SVM_clf, param)

    # Fit model using training data
    SVM_clf.fit(X_training, y_training)

    # Re-train using the optimized hyperparameter
    best_C_param = SVM_clf.best_params_['C']
    SVM_clf = svm.SVC(C=best_C_param, random_state=42)
    SVM_clf.fit(X_training, y_training)

    return SVM_clf
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def build_NeuralNetwork_classifier(X_training, y_training):
    '''

    Parameters
    ----------
    X_training : float
        Feature variables representing the data provided, all values are float
        and are standardized. Training data must also be split.
    y_training : int

    Returns
    -------
    NN_CLF : Class
        Returns the Neural Network thats fittted to the input data.

    '''
    # Hyperparameter assignment and Neural Network
    hidden_layers = [12,12,12]
    iterations=1000 # Define the iterations for training over the dataset
    param = {'hidden_layer_sizes': hidden_layers}
    NN_clf = neural_network.MLPClassifier(random_state=42)

    # Optimization of Hyperparameter - HIDDEN LAYERS
    NN_clf = GridSearchCV(NN_clf, param)

    # Fit model using training data
    NN_clf.fit(X_training, y_training)

    # Re-train using the optimized hyperparameter
    best_HLS_param = NN_clf.best_params_['hidden_layer_sizes']
    NN_clf = neural_network.MLPClassifier(hidden_layer_sizes=best_HLS_param,\
                                          max_iter=iterations,\
                                              random_state=42)
    NN_clf.fit(X_training, y_training)

    return NN_clf
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def print_seperation():
    to_report = '--------------------------------------------------------------------------------'

    return print(to_report)

def print_author_details():
    '''
    Returns
    -------
        Student details.
    '''
    to_report = '\n\nAuthor: Harry Akeroyd; Student# N9997121; Email N9997121.qut.edu.au\n\n'

    return print(to_report)

def print_clf_accuracy(y_test, clf_predict):
    '''
    Parameters
    ----------
    y_test : int
        Testing response variables.
    clf_predict : int
        Predicted values for selected classifier, DT, k-NN, SVM, NN.

    Returns
    -------
        Prints the score related to the accuracy value for selected clf.
    '''
    to_percentage = 100
    clf_accuracy = accuracy_score(y_test, clf_predict)
    clf_accuracy = to_percentage * round(clf_accuracy, 8)
    to_report = clf_accuracy

    return print('Classifier Accuracy: ', to_report, '%')

def print_confusion_matrix(y_test, clf_predict):
    '''
    Parameters
    ----------
    y_test : int
        Testing response variables.
    clf_predict : int
        Predicted values for selected classifier, DT, k-NN, SVM, NN.

    Returns
    -------
        Prints the confusion matrix for selected clf.
    '''
    clf_cm_data = confusion_matrix(y_test, clf_predict)
    clf_cm_format_clm = ['Predicted Malignant', 'Predicted Benign']
    clf_cm_format_ind = ['True Malignant','True Benign']
    clf_confusion_matrix = pd.DataFrame(clf_cm_data,\
                                        columns=clf_cm_format_clm,\
                                        index=clf_cm_format_ind)
    to_report = clf_confusion_matrix

    return print('Classifier Confusion Matrix:\n', to_report)

def print_precision_score(y_test, clf_predict):
    '''
    Parameters
    ----------
    y_test : int
        Testing response variables.
    clf_predict : int
        Predicted values for selected classifier, DT, k-NN, SVM, NN.

    Returns
    -------
        Prints the scores related to the precision value for selected clf.
    '''
    to_percentage = 100
    clf_precision = precision_score(y_test, clf_predict)
    clf_precision = to_percentage * round(clf_precision, 8)
    to_report = clf_precision

    return print('Precision: ', to_report,'%')

def print_recall_score(y_test, clf_predict):
    '''
    Parameters
    ----------
    y_test : int
        Testing response variables.
    clf_predict : int
        Predicted values for selected classifier, DT, k-NN, SVM, NN.

    Returns
    -------
        Prints the scores related to the recall value for selected clf.
    '''
    to_percentage = 100
    clf_recall = recall_score(y_test, clf_predict)
    clf_recall = to_percentage * round(clf_recall, 8)
    to_report = clf_recall

    return print('Recall: ', to_report,'%')

def print_cross_val_score(clf, X_test, y_test):
    '''
    Parameters
    ----------
    clf : Class
        Selected classifier, DT, k-NN, SVM, NN.
    X_test : float
        Testing feature variables.
    y_test : int
        Testing response variables.

    Returns
    -------
        Prints the scores related to the cross-validation for selected clf.
    '''
    to_report = cross_val_score(clf, X_test, y_test)

    return print('Cross Validation Score: ', to_report)

def print_visual_confusion_matrix(clf, X_test, y_test):
    '''
    Parameters
    ----------
    clf : Class
        Selected classifier, DT, k-NN, SVM, NN.
    X_test : float
        Testing feature variables.
    y_test : int
        Testing response variables.

    Returns
    -------
    to_report : Plot
        Prints a plot which displays the confusion matrix for selected clf.
    '''
    plot_confusion_matrix(clf, X_test, y_test)
    to_report = plt.show()

    return to_report

def print_standardized_data(X_train):
    '''
    Parameters
    ----------
    X_train : float
        Training data.

    Returns
    -------
    to_report : plot
        Returns a heatmap to demonstrate variations in the data.

    '''
    X_heatmap = X_train[:11]
    clf_heatmap = sns.heatmap(X_heatmap, cmap="YlGnBu")
    clf_heatmap.set(title="Variation in Standardized Medical Criteria",\
                    xlabel="Medical Criteria", ylabel="Patient",)
    to_report = plt.show()

    return to_report

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == "__main__":
    pass
    # Calling Data
    dataset = ('medical_records.data')

    # Preprocessing of the dataset
    X,y = prepare_dataset(dataset)

    # Train and Test splitting of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, \
                                                        random_state=42)

    # Validation splitting of data - 16% of 80%
    # 64% Training: 16% Validation: 20% Testing
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, \
                                                      random_state=42)

    # Applying Standard scaling to get optimized results
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Uncomment as required to test different classifiers
    clf = build_DecisionTree_classifier(X_train, y_train)
    #clf = build_NearrestNeighbours_classifier(X_train, y_train)
    #clf = build_SupportVectorMachine_classifier(X_train, y_train)
    #clf = build_NeuralNetwork_classifier(X_train, y_train)

    clf_predict = clf.predict(X_test)

    print_author_details()

    # PRINTING ACCURACY SCORE TO DISPLAY FOR SELECTED CLF
    print_seperation()
    print_clf_accuracy(y_test, clf_predict)
    print_seperation()

    # PRINTING CCONFUSION MATRIX TO DISPLAY FOR SELECTED CLF
    print_seperation()
    print_confusion_matrix(y_test, clf_predict)
    print_seperation()

    # PRINTING PRECISION SCORE TO DISPLAY FOR SELECTED CLF
    print_seperation()
    print_precision_score(y_test, clf_predict)
    print_seperation()

    # PRINTING RECALL SCORE TO DISPLAY FOR SELECTED CLF
    print_seperation()
    print_recall_score(y_test, clf_predict)
    print_seperation()

    # PRINTING CROSS VALIDATION SCORE TO DISPLAY FOR SELECTED CLF
    print_seperation()
    print_cross_val_score(clf, X_test, y_test)
    print_seperation()

    # PLOTTING CONFUSION MATRIX
    print_visual_confusion_matrix(clf, X_test, y_test)

    # HEATMAP FOR STANDARDIZED TRAINING MEDICAL CRITIERIA
    print_standardized_data(X_train)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
