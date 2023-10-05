#-------------------------------------------------------------------------
# AUTHOR: Julia Ybanez
# FILENAME: decision_tree.py
# SPECIFICATION: This program uses training data and then tests against test data
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours cummulatively
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]

    featureDictionary = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3, 'Myope': 1, 'Hypermetrope': 2, 'Yes': 1,
                         'No': 2, 'Normal': 1, 'Reduced': 2}

    num_attributes = 4
    for row in range(len(dbTraining)):
        temp = []
        for feature in range(0, num_attributes):
            temp.append(int(featureDictionary[dbTraining[row][feature]]))
        X.append(temp)

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    for row in range(len(dbTraining)):
        Y.append(int(featureDictionary[dbTraining[row][4]]))

    #loop your training and test tasks 10 times here
    worstAccuracy = 1.0
    for i in range (10):

       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       dbTestPreTransform = []
       with open('contact_lens_test.csv', 'r') as csvfile:
           reader = csv.reader(csvfile)
           for index, row in enumerate(reader):
               if index > 0:  # skipping the header
                   dbTestPreTransform.append(row)

       dbTest = []
       for row in range(len(dbTestPreTransform)):
           temp = []
           for feature in range(0, num_attributes + 1):
               temp.append(int(featureDictionary[dbTestPreTransform[row][feature]]))
           dbTest.append(temp)
       resultCount = 0
       correct = 0

       for data in dbTest:
           #transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           class_predicted = clf.predict([[data[0], data[1], data[2], data[3]]])[0]


           #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           if class_predicted == data[4]:
              correct = correct + 1
           resultCount = resultCount + 1

    #find the average of this model during the 10 runs (training and test set)
       accuracy = correct / resultCount
       if (accuracy < worstAccuracy):
           worstAccuracy = accuracy

    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2

    print("final accuracy when training on " + ds + ": " + str(worstAccuracy))




