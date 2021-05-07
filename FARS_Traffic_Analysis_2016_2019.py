#!/usr/bin/env python
# coding: utf-8

# In[123]:


#Import Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel (r'C:\Users\Life\Desktop\GMU\CS-504\dataset\AccidentData_2016_2019_cleaned.xlsx')
data


# In[124]:


# Analysis on VE_TOTAL column
print(data.groupby(['VE_TOTAL']).size().reset_index(name='counts'))
plt.hist(data['VE_TOTAL'])
plt.show()


# In[125]:


# Anlaysis on PERSONS
print(data.groupby(['PERSONS']).size().reset_index(name='counts'))
plt.hist(data['PERSONS'])
plt.show()


# In[126]:


print(data.groupby(['FATALS']).size().reset_index(name='counts'))
plt.hist(data['FATALS'])
plt.show()


# In[127]:


#Normalize Latitude/Longitude columns because they contain negative values
data['LATITUDE'] = 100 * (data['LATITUDE'] - data['LATITUDE'].min()) / (data['LATITUDE'].max() - data['LATITUDE'].min())
data['LONGITUD'] = 100 * (data['LONGITUD'] - data['LONGITUD'].min()) / (data['LONGITUD'].max() - data['LONGITUD'].min())
#Drop NaN/null values
#with pd.option_context('mode.use_inf_as_null', True):
#   data = data.dropna()
data[['LATITUDE', 'LONGITUD']]


# In[128]:


#Analysis on columns VE_TOTAL, PERSONS, FATALS used to derive Severity
data[["VE_TOTAL", "PERSONS", "FATALS"]].describe()


# In[129]:


#Normalize columns VE_TOTAL, PERSONS, FATALS based on mean & STD by calculating z-scores
# Derive column SUM_NORM by summing normalized values of VE_TOTAL, PERSONS, FATALS
VE_TOTAL_MEAN = data["VE_TOTAL"].mean()
VE_TOTAL_SD = data["VE_TOTAL"].std()
PERSONS_MEAN = data["PERSONS"].mean()
PERSONS_SD = data["PERSONS"].std()
FATALS_MEAN = data["FATALS"].mean()
FATALS_SD = data["FATALS"].std()

data["VE_TOTAL_NORM"] = round((data["VE_TOTAL"]-VE_TOTAL_MEAN)/VE_TOTAL_SD, 4)
data["PERSONS_NORM"] = round((data["PERSONS"]-PERSONS_MEAN)/PERSONS_SD, 4)
data["FATALS_NORM"] = round((data["FATALS"]-FATALS_MEAN)/FATALS_SD, 4)
data["SUM_NORM"] = data["VE_TOTAL_NORM"] + data["PERSONS_NORM"] + data["FATALS_NORM"]

df = data[["VE_TOTAL", "VE_TOTAL_NORM", "PERSONS", "PERSONS_NORM", "FATALS", "FATALS_NORM", "SUM_NORM"]]
df
df.describe()


# In[130]:


# Histogram - distribution of Normalized Sum
plt.hist(df['SUM_NORM'])
plt.show()


# In[137]:


from matplotlib.pyplot import figure
#g = sns.FacetGrid(df[['VE_TOTAL', 'PERSONS', 'FATALS', 'SUM_NORM']], col='cols', hue="target", palette="Set1")
#g = (g.map(sns.distplot, "vals", hist=False, rug=True))
#sns.plt.show()
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
#sns.set(style="darkgrid")

#fig, axs = plt.subplots(2, 2, figsize=(7, 7))

figure(figsize=(8, 6), dpi=80)

ax = plt.subplot(2, 2, 1)
# Draw the plot
ax.hist(df['VE_TOTAL'],color = 'blue', edgecolor = 'black')
ax.set_title('VE_TOTAL', size = 10)
ax.set_xlabel('VE_TOTAL', size = 10)
ax.set_ylabel('Count', size= 10)

ax = plt.subplot(2, 2, 2)
# Draw the plot
ax.hist(df['PERSONS'],color = 'blue', edgecolor = 'black')
ax.set_title('PERSONS', size = 10)
ax.set_xlabel('PERSONS', size = 10)
ax.set_ylabel('Count', size= 10)

ax = plt.subplot(2, 2, 3)
# Draw the plot
ax.hist(df['FATALS'],color = 'blue', edgecolor = 'black')
ax.set_title('FATALS', size = 10)
ax.set_xlabel('FATALS', size = 10)
ax.set_ylabel('Count', size= 10)

ax = plt.subplot(2, 2, 4)
# Draw the plot
ax.hist(df['SUM_NORM'],color = 'blue', edgecolor = 'black')
ax.set_title('SUM_NORM', size = 10)
ax.set_xlabel('SUM_NORM', size = 10)
ax.set_ylabel('Count', size= 10)

plt.tight_layout()

#sns.histplot(data=df[['VE_TOTAL', 'PERSONS', 'FATALS', 'SUM_NORM']], x="VE_TOTAL", kde=True, color="skyblue", ax=axs[0, 0])
#sns.histplot(data=df[['VE_TOTAL', 'PERSONS', 'FATALS', 'SUM_NORM']], x="PERSONS", kde=True, color="olive", ax=axs[0, 1])
#sns.histplot(data=df[['VE_TOTAL', 'PERSONS', 'FATALS', 'SUM_NORM']], x="FATALS", kde=True, color="gold", ax=axs[1, 0])
#sns.histplot(data=df[['VE_TOTAL', 'PERSONS', 'FATALS', 'SUM_NORM']], x="SUM_NORM", kde=True, color="teal", ax=axs[1, 1])

plt.show()


# In[70]:


#Define dependent variable 'SEVERITY' from derived summation value of above 3 normalized variables
conditions = [
    (data['SUM_NORM'] <= data["SUM_NORM"].describe().loc['25%']),
    (data['SUM_NORM'] > data["SUM_NORM"].describe().loc['25%']) 
        & (data['SUM_NORM'] <= data["SUM_NORM"].describe().loc['75%']),
    (data['SUM_NORM'] > data["SUM_NORM"].describe().loc['75%'])
    ]
#values = ['Low', 'Medium', 'High'] # create a list of the values we want to assign for each condition
values = [1, 2, 3] # create a list of the values we want to assign for each condition
# create a new column and use np.select to assign values to it using our lists as arguments
data['SEVERITY'] = np.select(conditions, values)

print('Dataframe size after adding SEVERITY column : '+ str(data.shape))
print('Dataframe Columns after adding SEVERITY column : '+ str(data.columns))

data.drop('VE_TOTAL', inplace=True, axis=1)
data.drop('PERSONS', inplace=True, axis=1)
data.drop('FATALS', inplace=True, axis=1)
data.drop('PERMVIT', inplace=True, axis=1)
data.drop('VE_FORMS', inplace=True, axis=1)

print('Dataframe size after dropping 5 columns : '+ str(data.shape))
print('Dataframe Columns after dropping 5 columns : '+ str(data.columns))

data.head(20)


# In[71]:


# Drop None/NAN/INF/-INF
print('Dataframe size before dropping : '+ str(len(data.index)))
with pd.option_context('mode.use_inf_as_null', True):
   data = data.dropna()
print('Dataframe size after dropping : '+ str(len(data.index)))
data


# In[118]:


# Find feature scores and pick top 20 features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.iloc[:,0:43]  #independent columns
y = data.iloc[:,-1]    #target column i.e SEVERITY
#apply SelectKBest class to extract top 20 best features
bestfeatures = SelectKBest(score_func=chi2, k=20)
fit = bestfeatures.fit(X,y)
#dfscores = pd.DataFrame(fit.scores_)
#dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
#featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#featureScores.columns = ['Attributes','feature_score']  #naming the dataframe columns
#print(featureScores.nlargest(20,'feature_score'))  #print 20 best features
#featureScores.nlargest(20,'feature_score').plot(kind='barh')
#plt.show()

feat_scores = pd.Series(fit.scores_, index=X.columns)
feat_scores.nlargest(20).plot(kind='barh')
plt.show()


# In[73]:


# Calculate feature importance - pick Top 20 features
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[74]:


#Correlation Matrix
df = pd.DataFrame(data, columns= ['MILEPT','COUNTY','STATE', 'CITY','ROUTE','LATITUDE','LONGITUD',
                  'MAN_COLL','HARM_EV','RD_OWNER','REL_ROAD','CF1','CF3','PVH_INVL','WEATHER',
                  'NOT_HOUR','ARR_HOUR','ARR_MIN','DAY_WEEK','DAY','NOT_MIN','MONTH','HOUR',
                  'PEDS','PERNOTMVIT','HOSP_HR','HOSP_MN','FUNC_SYS','YEAR','SEVERITY'])
corrmax = df.corr()

f, ax = plt.subplots(figsize=(20,16))
sns.heatmap(corrmax, annot = True)
plt.show()


# In[75]:


# Prepare Models - Train/Test data preparation
from sklearn.model_selection import train_test_split

x = data[['MILEPT','COUNTY','STATE', 'CITY','ROUTE','LATITUDE','LONGITUD',
                  'MAN_COLL','HARM_EV','RD_OWNER','REL_ROAD','CF1','CF3','PVH_INVL','WEATHER',
                  'NOT_HOUR','ARR_HOUR','ARR_MIN','DAY_WEEK','DAY','NOT_MIN','MONTH','HOUR',
                  'PEDS','PERNOTMVIT','HOSP_HR','HOSP_MN','FUNC_SYS','YEAR']]
y = data['SEVERITY']

x_train, x_test, y_train, y_test =    train_test_split(x, y, test_size=0.2, random_state=0)


# In[76]:


#Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn import metrics

lr_model = LinearRegression().fit(x_train, y_train)

# Value of R-Square
lr_r_square = lr_model.score(x_train, y_train)
print("R-square on Train data : ", lr_r_square)

# Intercept of Linear Regression line
lr_intercept = lr_model.intercept_
print("Intercept value : ", lr_intercept)

# Coefficients in Linear Regression equation
lr_coefficient = lr_model.coef_
print("Coefficients : ", lr_coefficient)

#####################################
##### Run Predictions & Evaluate
#####################################

lr_predictions_y = lr_model.predict(x_test)

# Value of R-Square on test data
lr_r_square_test = lr_model.score(x_test, y_test)
print("R-square on Test data : ", lr_r_square_test)

# Model Evaluation
print("Mean Absolute Error : ", metrics.mean_absolute_error(y_test, lr_predictions_y))
print("Mean Squared Error : ", metrics.mean_squared_error(y_test, lr_predictions_y))
print("Root Mean Squared Error : ", np.sqrt(metrics.mean_squared_error(y_test, lr_predictions_y)))


# In[78]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

logistic_model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',
                           random_state=0)
logistic_model.fit(x_train, y_train)

print("Output classes : ", logistic_model.classes_)
print("Intercept : ", logistic_model.intercept_)
print("Coefficients : ", logistic_model.coef_)

y_pred = logistic_model.predict(x_test)

print("Score on entire data : ", logistic_model.score(x, y))
print("Score on Train data : ", logistic_model.score(x_train, y_train))
print("Score on Test data : ", logistic_model.score(x_test, y_test))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.set_xlabel('Predicted outputs', color='black')
ax.set_ylabel('Actual outputs', color='black')
ax.xaxis.set(ticks=range(3))
ax.yaxis.set(ticks=range(3))
ax.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()


# In[79]:


# SVM - Support Vector machine
from sklearn.svm import LinearSVC

svc_model = LinearSVC()
svc_model.fit(x_train, y_train)
y_pred = svc_model.predict(x_test)

print("Score on entire data : ", svc_model.score(x, y))
print("Score on Train data : ", svc_model.score(x_train, y_train))
print("Score on Test data : ", svc_model.score(x_test, y_test))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.set_xlabel('Predicted outputs', color='black')
ax.set_ylabel('Actual outputs', color='black')
ax.xaxis.set(ticks=range(3))
ax.yaxis.set(ticks=range(3))
ax.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()


# In[80]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB

NB_model = GaussianNB()
NB_model.fit(x_train, y_train)
y_pred = NB_model.predict(x_test)

print("Score on entire data : ", NB_model.score(x, y))
print("Score on Train data : ", NB_model.score(x_train, y_train))
print("Score on Test data : ", NB_model.score(x_test, y_test))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.set_xlabel('Predicted outputs', color='black')
ax.set_ylabel('Actual outputs', color='black')
ax.xaxis.set(ticks=range(3))
ax.yaxis.set(ticks=range(3))
ax.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()


# In[101]:


# Neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(x_train, y_train, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(x_train, y_train)
print('Test Accuracy: %.2f' % (accuracy*100))

# make probability predictions with the model
predictions = model.predict(x_test)
# round predictions 
rounded = [round(x[0]) for x in predictions]

# make class predictions with the model
predictions = model.predict_classes(x_test)


# In[89]:


# Plot accuracy comparisons between scenario 1(assumed features) & scenario 2(features from analysis)
plot_data = pd.read_excel (r'C:\Users\Life\Desktop\GMU\CS-504\dataset\AccuracyResults.xlsx')

plot_data


# In[98]:


# Plot accuracy comparisons between scenario 1(assumed features) & scenario 2(features from analysis)
# Logistic Regression, SVM, Naive Bayes
plot_data = pd.read_excel (r'C:\Users\Life\Desktop\GMU\CS-504\dataset\AccuracyResults.xlsx')

# Initialise a figure. subplots() with no args gives one plot.
fig, ax = plt.subplots()

# data preparation
models = plot_data['Model']
x = np.arange(len(models))

# Plot 
ax.bar(x - 0.3/2, 100*plot_data['accuracy_scenario_1'], 0.3, label='Scenario 1', color='#0343df')
ax.bar(x + 0.3/2, 100*plot_data['accuracy_scenario_2'], 0.3, label='Scenario 2', color='#e50000')

# Customise some display properties
ax.set_ylabel('Model Accuracy')
ax.set_title('Other Models - Feature Selection Analysis')
ax.set_xticks(x)  
ax.set_xticklabels(models.astype(str).values, rotation='horizontal')
ax.legend()

# show the plot
plt.show()


# In[100]:


# Plot accuracy comparisons between scenario 1(assumed features) & scenario 2(features from analysis)
# Linear Regression
plot_data = pd.read_excel (r'C:\Users\Life\Desktop\GMU\CS-504\dataset\AccuracyResults.xlsx')

# Initialise a figure. subplots() with no args gives one plot.
fig, ax = plt.subplots()

# data preparation
models = plot_data['Metric']
x = np.arange(len(models))

# Plot 
ax.bar(x - 0.3/2, plot_data['scenario_1'], 0.3, label='Scenario 1', color='#0343df')
ax.bar(x + 0.3/2, plot_data['scenario_2'], 0.3, label='Scenario 2', color='#e50000')

# Customise some display properties
ax.set_ylabel('Metrics')
ax.set_title('Linear Regression - Feature Selection Analysis')
ax.set_xticks(x)  
ax.set_xticklabels(models.astype(str).values, rotation='horizontal')
ax.legend()

# show the plot
plt.show()


# In[ ]:




