import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



#reading data from csv file
dataset = pd.read_csv("creditcard.csv")
#print(dataset)

#setting normal and abnormal(fraud) values
normal = dataset.loc[dataset['Class'] == 0]
abnormal = dataset.loc[dataset['Class'] == 1]
#print(abnormal)

sns.relplot(x='Amount', y='Time', hue='Class', data=dataset)

x = dataset.iloc[:, :-1]
y = dataset['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)
#setting up classifier
classifier = LogisticRegression(class_weight='balanced')
classifier.fit(x_train, y_train)

#Making predictions on the test set
y_predict = np.array(classifier.predict(x_test))

#Evaluating the model
with open('output.txt', 'wt') as f:
    print("Confusion Matrix: ", confusion_matrix(y_test, y_predict), file=f)
    print("Classification Report: ", classification_report(y_test, y_predict), file=f)
    print("Accuracy:", accuracy_score(y_test, y_predict), file=f)
#plt.show()

'''
classifier = RandomForestClassifier(class_weight='balanced')
classifier.fit(x_train, y_train)

# Make predictions on the test set
y_predict = classifier.predict(x_test)

# Evaluate the model
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
print("Accuracy:", accuracy_score(y_test, y_predict))'''