import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection 

dataset = pd.read_csv('./sonar.csv',header=None)


# Data visualization
print(dataset.head())
print(dataset.shape)
print(dataset[60].value_counts())

# Separating the data

X = dataset.drop(columns=60)
Y = dataset[60]

print(X)
print(Y)

X_train , X_test , Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y ,random_state= 1)

model = LogisticRegression()

# Training model
model.fit(X_train,Y_train)

#Accuracy on train/test data
X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Accuracy on training data : "+ str(training_accuracy))

X_test_prediction = model.predict(X_test)
training_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Accuracy on testing data : "+ str(training_accuracy))



input_data = (0.0363,0.0478,0.0298,0.0210,0.1409,0.1916,0.1349,0.1613,0.1703,0.1444,0.1989,0.2154,0.2863,0.3570,0.3980,0.4359,0.5334,0.6304,0.6995,0.7435,0.8379,0.8641,0.9014,0.9432,0.9536,1.0000,0.9547,0.9745,0.8962,0.7196,0.5462,0.3156,0.2525,0.1969,0.2189,0.1533,0.0711,0.1498,0.1755,0.2276,0.1322,0.1056,0.1973,0.1692,0.1881,0.1177,0.0779,0.0495,0.0492,0.0194,0.0250,0.0115,0.0190,0.0055,0.0096,0.0050,0.0066,0.0114,0.0073,0.0033)

#changing input to numpy array

input_data_as_numpy = np.asarray(input_data)

#reshape the np array as we are prediciting for one instance

input_data_reshaped = input_data_as_numpy.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

print(prediction)