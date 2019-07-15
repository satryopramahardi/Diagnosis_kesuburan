import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('fertility.csv')
# print(df.describe())
# print(df.columns)
# print(df.count())

df = df.replace('yes',True)
df = df.replace('no',False)

df['High fevers in the last year'].value_counts()
df['Frequency of alcohol consumption'].value_counts()
df['Smoking habit'].value_counts()
df['Diagnosis'].value_counts()
df = df.drop(['Season'],axis=1)

print("Preprocessing data")
from sklearn import preprocessing
le_fever = preprocessing.LabelEncoder()
le_alc = preprocessing.LabelEncoder()
le_smok = preprocessing.LabelEncoder()

new_df = df
le_alc.fit(df['Frequency of alcohol consumption'])
new_df['Frequency of alcohol consumption'] = le_alc.transform(df['Frequency of alcohol consumption'])

le_smok.fit(df['Smoking habit'])
new_df['Smoking habit'] = le_smok.transform(df['Smoking habit'])

new_df['High fevers in the last year'] = new_df['High fevers in the last year'].astype(str)

le_fever.fit(df['High fevers in the last year'])
new_df['High fevers in the last year'] = le_fever.transform(df['High fevers in the last year'])

le_target = preprocessing.LabelEncoder()

le_target.fit(df['Diagnosis'])
new_df['Diagnosis'] = le_target.transform(df['Diagnosis'])


# plt.figure(figsize=(10,10))
# sns.heatmap(new_df.corr(method='kendall'),vmin=-1,cmap='coolwarm',annot=True)
# plt.show()

from sklearn.model_selection import train_test_split

X=new_df.drop(['Diagnosis'],axis=1)
y=new_df['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=99)
print("Preprocessing done")
print("Fitting model")
from sklearn.ensemble import RandomForestClassifier

rand_forest = RandomForestClassifier(n_estimators=10,random_state=2)
rand_forest.fit(X_train, y_train)  
ran_forest_pred = rand_forest.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test,ran_forest_pred))
# print(rand_forest.score(X_train,y_train))

from sklearn.linear_model import LogisticRegression
logistic_reg = LogisticRegression(random_state=0, solver='lbfgs',max_iter=1000).fit(X_train, y_train)
logistic_reg_pred = logistic_reg.predict(X_test) 

# print(confusion_matrix(y_test,logistic_reg_pred))
# print(logistic_reg.score(X_train,y_train))


from sklearn.linear_model import RidgeClassifier
ridge_clf = RidgeClassifier()
ridge_clf.fit(X_train, y_train)
ridge_clf_pred = ridge_clf.predict(X_test) 

# print(confusion_matrix(y_test,ridge_clf_pred))
# print(ridge_clf.score(X_train,y_train))

print("Fitting model done")

dict_input = [
    {'nama' : 'Arin', 'umur':29, 'child':False, 'accident':False , 'surgery': False,'Fever':'False','Alcohol': 'every day', 'smoke':'daily', 'sit': 5},
    {'nama' : 'Bebi', 'umur':31, 'child':False, 'accident':True , 'surgery': True,'Fever':'False','Alcohol': 'once a week', 'smoke':'never', 'sit': 24},
    {'nama' : 'Caca', 'umur':25, 'child':True, 'accident':False , 'surgery': False,'Fever':'False','Alcohol': 'hardly ever or never', 'smoke':'never', 'sit': 7},
    {'nama' : 'Dini', 'umur':28, 'child':False, 'accident':True , 'surgery': True,'Fever':'False','Alcohol': 'hardly ever or never', 'smoke':'daily', 'sit': 24},
    {'nama' : 'Enno', 'umur':42, 'child':True, 'accident':False , 'surgery': False,'Fever':'False','Alcohol': 'hardly ever or never', 'smoke':'never', 'sit': 8}
]

df2 = pd.DataFrame(dict_input)
df2 = df2[['nama','umur','child','accident','surgery','Fever','Alcohol','smoke','sit']]

df2['Alcohol'] = le_alc.transform(df2['Alcohol'])
df2['Fever'] = le_fever.transform(df2['Fever'])
df2['smoke'] = le_smok.transform(df2['smoke'])

df2['RandomForestClassifier'] = pd.Series()
df2['LogisticRegression'] = pd.Series()
df2['RidgeClassifier'] = pd.Series()
print("Predicting")
df2['RandomForestClassifier'] = rand_forest.predict(df2.drop(['nama','RandomForestClassifier','LogisticRegression','RidgeClassifier'],axis=1))
df2['LogisticRegression'] = logistic_reg.predict(df2.drop(['nama','RandomForestClassifier','LogisticRegression','RidgeClassifier'],axis=1))
df2['RidgeClassifier'] = ridge_clf.predict(df2.drop(['nama','RandomForestClassifier','LogisticRegression','RidgeClassifier'],axis=1))

for row in df2.iterrows():
  nama = row[1][0]
  print(f"{nama}, prediksi kesuburan: {le_target.inverse_transform([row[1][9]])[0]} (RandomForestClassifier)")
  print(f"{nama}, prediksi kesuburan: {le_target.inverse_transform([row[1][10]])[0]} (LogisticRegression)")
  print(f"{nama}, prediksi kesuburan: {le_target.inverse_transform([row[1][11]])[0]} (RidgeClassifier)")