#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
db = []
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append(row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
featureDictionary = {'Sunny' : 1, 'Overcast':2, 'Rain':3, 'Hot':1, 'Mild':2, 'Cool':3,
                     'High':1, 'Normal':2, 'Weak':1, 'Strong':2, 'Yes':1, 'No':2}
X=[]
Y=[]
for row in range(len(db)):
    X.append([int(featureDictionary[db[row][1]]),
              int(featureDictionary[db[row][2]]),
              int(featureDictionary[db[row][3]]),
              int(featureDictionary[db[row][4]])])
    Y.append(int(featureDictionary[db[row][5]]))

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
testFileDB = []
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         testFileDB.append(row)

testX = []
for row in range(len(testFileDB)):
    testX.append([int(featureDictionary[testFileDB[row][1]]),
              int(featureDictionary[testFileDB[row][2]]),
              int(featureDictionary[testFileDB[row][3]]),
              int(featureDictionary[testFileDB[row][4]])])

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
testY = []
for i in range(len(testX)):
    testY.append(clf.predict_proba([testX[i]])[0])
    output = 1 if testY[i][0] > 0.75 else 0
    confidence = testY[i][0] if output == 1 else testY[i][1]
    outputStr = "Yes" if output == 1 else "No"
    if testY[i][0] >= 0.75 or testY[i][1] >= 0.75:
        print(str(testFileDB[i][0]).ljust(15)
              + str(testFileDB[i][1]).ljust(15)
              + str(testFileDB[i][2]).ljust(15)
              + str(testFileDB[i][3]).ljust(15)
              + str(testFileDB[i][4]).ljust(15)
              + outputStr.ljust(15)
              + str(round(confidence, 2)).ljust(15))


