import pandas as pd
import streamlit as st
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split

st.set_page_config(
        page_title="Profit Calculator",
        
    )

st.write("""
# Profit Calculator for startups.

\n
This application calculates the expected profit of startups. The concept of Multiple Linear Regression is used.
""")

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1].values

st.write(""" ### Input Dataset : """)
st.write(dataset)


st.write(""" ### Dataset after pre-processing :

One Hot Encoding is used to change the categorical values into a form suitable for regression analysis. The resulting dataset is shown below:
""")
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))\

st.write(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

st.write(""" ### Splitting the dataset :

The dataset is split intro training(0.8) and test(0.2) sets. 
""")

regressor = LinearRegression()
regressor.fit(X_train, y_train)

st.write(""" ### Training the model :

LinearRegression is used to train a multiple regression model.
""")
#sidebar

st.sidebar.write("""
## Input Parameters

Enter the input parameters : 
""")

rnd = st.sidebar.slider('R&D Spend', 0, 200000, 100000)
admin = st.sidebar.slider('Administration', 0, 200000, 100000)
ms = st.sidebar.slider('Marketing Spend', 0, 500000, 250000)
state = st.sidebar.selectbox('State', ('New York', 'California', 'Florida'))

stateval = [0.0, 0.0, 1.0]

if "".join(str(state).split()).lower() == 'newyork':
    stateval = [0.0, 0.0, 1.0]
elif "".join(str(state).split()).lower() == 'california':
    stateval = [1.0, 0.0, 0.0]
else :
    stateval = [0.0, 1.0, 0.0]


stateval.extend([rnd, admin, ms])

input_param = pd.DataFrame([stateval], columns=list('012345'))
st.write(""" *** 
### Input Parameters :


""")
st.write(pd.DataFrame([[rnd, admin, ms, state]], columns=['R&D', 'Administration', 'Marketing Spend', 'State']))

st.write("""
***

## Expected Profit :
""")

if st.sidebar.button('Predict'):
    op = regressor.predict(input_param)

    st.write(str(round(op[0], 2)))
else:
    st.write('Please press \'Predict\' after changing input parameters')