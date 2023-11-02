#-------------------------------------------------------------------------
# AUTHOR: Julia Ybanez
# FILENAME: svm.py
# SPECIFICATION: description of the program
# FOR: CS 4210 - Assignment #3
# TIME SPENT: 
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highest_accuracy = 0  # Initialize the highest accuracy variable


df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
for c_value in c:
    for degree_value in degree:
        for kernel_value in kernel:
            for dfs_value in decision_function_shape:

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                clf = svm.SVC(C=c_value, degree=degree_value, kernel=kernel_value, decision_function_shape=dfs_value)

                #Fit SVM to the training data
                clf.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                correct_predictions = 0
                total_predictions = len(X_test)

                # Make predictions on the test data
                for x_testSample, y_testSample in zip(X_test, y_test):
                    prediction = clf.predict([x_testSample])
                    if prediction == y_testSample:
                        correct_predictions += 1

                # Calculate accuracy
                accuracy = correct_predictions / total_predictions

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy

                    # Print the highest accuracy and the SVM hyperparameters
                    print(f"Highest SVM accuracy so far: {highest_accuracy:.6f}, Parameters: C={c_value}, degree={degree_value}, kernel={kernel_value}, decision_function_shape='{dfs_value}'")