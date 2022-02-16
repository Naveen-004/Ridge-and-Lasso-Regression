# Importing required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Title for the Web App
st.title('Ridge and Lasso Regression')
# Markdown for the description      
st.markdown('''
<h2>Explore Regularization</h2>
<h6>Which one is the best Regularization for a particular dataset(sklearn datasets)?\n
<ol><b><li>Ridge</li><li>Lasso</li></b></ol>''', unsafe_allow_html=True)

# Dataset selection
dataset_name = st.selectbox('Select Dataset',("Boston","Iris","Breast Cancer","Diabetes","Digits","Wine"))
st.write('You Selected' + ' ' + dataset_name + ' ' +"Dataset" )

# Function to get Features and Target --> X and Y
@st.cache # Decorator to cache the function
def get_dataset(dataset_name): # if-elif-else statement to get X and Y
    if dataset_name == 'Boston':
        df = datasets.load_boston()
        dataset = pd.DataFrame(df.data)
        dataset.columns = df.feature_names
        dataset['price'] = df.target
        X = dataset.iloc[:,:-1]
        y = dataset.iloc[:,-1]
    elif dataset_name == 'Iris':
        df = datasets.load_iris()
        dataset =pd.DataFrame(df.data)
        dataset.columns = df.feature_names
        dataset['target'] = df.target
        X = dataset.iloc[:,:-1]
        y = dataset.iloc[:,-1]
    elif dataset_name == 'Breast Cancer':
        df = datasets.load_breast_cancer()
        dataset =pd.DataFrame(df.data)
        dataset.columns = df.feature_names
        dataset['target'] = df.target
        X = dataset.iloc[:,:-1]
        y = dataset.iloc[:,-1]
    elif dataset_name == 'Diabetes':
        df = datasets.load_diabetes()
        dataset =pd.DataFrame(df.data)
        dataset.columns = df.feature_names
        dataset['target'] = df.target
        X = dataset.iloc[:,:-1]
        y = dataset.iloc[:,-1]
    elif dataset_name == 'Digits':
        df = datasets.load_digits()
        dataset =pd.DataFrame(df.data)
        dataset.columns = df.feature_names
        dataset['target'] = df.target
        X = dataset.iloc[:,:-1]
        y = dataset.iloc[:,-1]   
    else:
        df = datasets.load_wine()
        dataset =pd.DataFrame(df.data)
        dataset.columns = df.feature_names
        dataset['target'] = df.target
        X = dataset.iloc[:,:-1]
        y = dataset.iloc[:,-1]           
    return X, y

# Getting X and Y
X, y = get_dataset(dataset_name)
st.write("Shape of Features :", X.shape)
st.write('Overview of Features -', X)
st.write("Shape of Target :", y.shape)
st.write('Overview of Target data -', y)
# Description of the dataset
st.write('''As Ridge and Lasso Regression models are how of regularizing the linear models so we first got to prepare a linear model. So now, let’s code for preparing a linear regression model''')

# Heading for the Linear Regression
st.write('# Linear Model\'s MSE')
reg = LinearRegression() # Initializing the Linear Regression
# neg_mean_squared_error function to calculate the MSE
mse = cross_val_score(reg, X, y, scoring='neg_mean_squared_error',cv=5)
mean_mse = np.mean(mse)
st.write("Mean MSE of "+ dataset_name + ' ' +'Dataset is '+ ' ' +str(mean_mse))
# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Heading for the Ridge Regression
st.write('# Ridge Regression')
@st.cache # Decorator to cache the function
def ridge():
    ridge = Ridge() # Initializing the Ridge Regression
    alpha_val = np.array(range(-100,300))
    # GridSearchCV to find the best parameters
    params = {'alpha': alpha_val}
    ridge_reg = GridSearchCV(ridge, params, scoring='neg_mean_squared_error',cv=5)
    ridge_reg.fit(X,y) # Fitting the data
    return ridge_reg

st.write("Best parameters : ", str(ridge().best_params_))
st.write('Best Score : ', str(ridge().best_score_))
ridge_score = ridge().best_score_
ridge_val = str(ridge().best_params_)
pred_ridge = ridge().predict(X_test)
fig = plt.figure() # Initializing the figure
sns.distplot(y_test-pred_ridge, color='blue')
st.pyplot(fig) # plotting the figure

# Heading for the Lasso Regression
st.write('# Lasso Regression')
@st.cache # Decorator to cache the function
def lasso():
    lasso = Lasso() # initializing the Lasso Regression
    alpha_val = np.array(range(-100,300))
    # GridSearchCV to find the best parameters
    params = {'alpha': alpha_val}
    lasso_reg = GridSearchCV(lasso, params, scoring='neg_mean_squared_error',cv=5)
    lasso_reg.fit(X,y) # Fitting the data
    return lasso_reg

st.write("Best parameters : ", str(lasso().best_params_))
st.write('Best Score : ', str(lasso().best_score_))
lasso_score = lasso().best_score_
lasso_val = str(lasso().best_params_)
pred_lasso = lasso().predict(X_test)
fig = plt.figure() # initializing the figure
sns.distplot(y_test-pred_lasso, color='blue')
st.pyplot(fig) # plotting the distribution of the residuals

# Conclusion
st.write('# Conclusion')
# if-else statement to get the best model
if lasso_score < ridge_score:
    st.write('Implementation of Ridge and Lasso regularization is done. In this case, the Ridge is that the best method of adjustment, with a regularization value of ',ridge_val)
else:
    st.write('Implementation of Ridge and Lasso regularization is done. In this case, the Lasso is that the best method of adjustment, with a regularization value of ',lasso_val)