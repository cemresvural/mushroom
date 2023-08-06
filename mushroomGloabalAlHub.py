import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import classification_report


data=pd.read_csv("C:\\Users\cemre\Downloads\mushrooms.csv")
data.head()

#use value_counts method on "class" column of data object
classes =data['class'].value_counts()
classes

plt.bar('Edible',classes['e'])
plt.bar('Poisonous',classes['p'])
plt.show()


#Features and labels
X=data.loc[:,['cap-shape','cap-color','ring-number','ring-type']]
y=data.loc[:,'class']

#Converting the values
encoder =LabelEncoder()
#encode the features to integers inside a for loop
for i in X.columns:
    X[i]=encoder.fit_transform(X[i])

#encode the output labels to integers
y= encoder.fit_transform(y)

#print X and  y
X
y

#Split the data, 70-30 ratio
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.3)

#Creating and training models

logistic_classifier_model=LogisticRegression()
ridge_classifier_model=RidgeClassifier()
decision_tree_model=DecisionTreeClassifier()
naive_bayes_model=GaussianNB()
neural_network_model=MLPClassifier()


#Train all models using.fit() method of each object

logistic_classifier_model.fit(X_train,y_train)
ridge_classifier_model.fit(X_train,y_train)
decision_tree_model.fit(X_train,y_train)
naive_bayes_model.fit(X_train,y_train)
neural_network_model.fit(X_train,y_train)

#Use the.predict() method on each model

logistic_pred=logistic_classifier_model.predict(X_test)
ridge_pred=ridge_classifier_model.predict(X_test)
tree_pred=decision_tree_model.predict(X_test)
naive_pred=naive_bayes_model.predict(X_test)
neural_pred=neural_network_model.predict(X_test)

#COMPARING THE PERFORMANCES
logistic_report=classification_report(y_test,logistic_pred)
ridge_report=classification_report(y_test,ridge_pred)
tree_report=classification_report(y_test,tree_pred)
naive_bayes_report=classification_report(y_test,naive_pred)
neural_network_report=classification_report(y_test,neural_pred)



print('******  Logistic Regression *******')
print(logistic_report)
print('******  Ridge Regression *******')
print(ridge_report)
print('******  Decision Tree  *******')
print(tree_report)
print('******  Naive Bayes  *******')
print(naive_bayes_report)
print('******  Neural Network  *******')
print(neural_network_report)



#EVALUATION
random_forest_model=RandomForestClassifier()
random_forest_model.fit(X_train,y_train)
random_forest_pred=random_forest_model.predict(X_test)

random_forest_report=classification_report(y_test,random_forest_pred)
print(random_forest_report)