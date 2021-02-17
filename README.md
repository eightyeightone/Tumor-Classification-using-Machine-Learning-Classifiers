## Tumor Classification using Machine Learning Classifiers
###### Queensland University of Technology Project: CAB320 Artificial Intelligence, completed in Semester 1 of 2020 (Feb' - June).

#### Summary

The aim of this project was to build four different types of classifiers and present their performance in a report. The classification task was to predict whether a tumour would be malignant or benign, denoted by M and B, respectively. 

Types of classifiers and pre-selected hyperparameter;
1.	*Nearest Neighbour*: number of neighbours,
2.	*Decision Tree*: maximum depth of the tree,
3.	*Support Vector Machine*: parameter C,
4.	*Neural Network*: number of neurons in the hidden layer.

The maths behind each classification algorithm was not the subject of this work, the implementation of each classifier using Scikit-learn libraries was.

The works was separated into two steps;
1.	*Pre-processing the data*. Prepossessed data set to remove any anomalies and convert to NumPy arrays. Separate patient diagnosis and patient test results into response and feature variables, respectively.

2.	*Building the classifiers*. Each classifier follows the below pseudo-code with the appropriate libraries and functions used.
  *	Initialise the classifier from the sklearn library,
  *	Execute the grid search for hyperparameter optimization,
  *	Fit the data to the estimator,
  *	Re-train the estimator using the optimized parameter,
  *	Re-fit the data to the estimator.



> By Harry Akeroyd. n9997121@qut.edu.au
