# ===================================================================================================================================================
# ========================================================== IMPORTING LIBRARY SECTION ==============================================================
# ===================================================================================================================================================


# ===================== IMPORTNG LIBRARIES =====================
import pandas as pd
import numpy as np


# ===================== SCIKIT LIBRARIES =====================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse


# ===================== KERAS LIBRARIES =====================
from tensorflow import keras
from keras.api.optimizers import Adam


# ===================== GRAPH LIBRARIES =====================
import streamlit as stl
import altair as atr


# ===================== COMPARISON REGRESSION LIBRARIES =====================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR 
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge





# ===================================================================================================================================================
# ============================================================== KERAS MODEL CREATION ===============================================================
# ===================================================================================================================================================




# ===================== CAR PURCHASE DATASET ====================
# loading given dataset --> using csv as xlsx file not loading
df_cpd = pd.read_csv("Car_Purchasing_Data.csv")




# ========================= TESTING DATASET WORKING =========================
#testing dataset ensuring iit loaded and is readable
print(df_cpd[:5])
# or
print(df_cpd.head())


# creating space between content
print("\n\n")




# ========================= DROP UNNECESSARY COLUMNS =========================
# irrelevant columns dropped from dataset
X = df_cpd.drop(['Customer Name','Customer e-mail','Country', 'Car Purchase Amount'], axis = 1)

# output column target variable
Y = df_cpd["Car Purchase Amount"]


# creating space between content
print("\n\n")



# ======================== TRANSFORM DATASET --> PERCENTAGE BASED = BETWEEN 0 & 1 =========================
# mixmaxscaler initialisation for scaling
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

# tranforming x variable from before = input features
x_scaled = x_scaler.fit_transform(X)

# reshapring and transforming y variable
# converting to 2D array = reshape (-1.1) = compatible to regression models
y_reshaped = Y.values.reshape(-1, 1)
y_scaled = y_scaler.fit_transform(y_reshaped)


# creating space between content
print("\n\n")




# ========================= TRAIN AND TEST =========================
# Splitting the dataset --> 20% to testing and 80% for training
X_train, X_test, Y_train, Y_test = train_test_split(x_scaled, y_scaled, test_size = 0.2, train_size = 0.8, random_state = 42)


# creating space from content to content
print("\n\n")




# ========================= MODEL DEFINING | TRAINING | COMPILING =========================
# creating and defining keras sequential model
# numbers = nerons --> type of connection that connection neurons from previous layer
# input dimension = feature we training 
# RELU = Rectified Linear Unit --> outputs the input directly if it is positive; otherwise, it will output zero
# dense --> layer with 64 neurons and ReLU activation takes the input data and processes it.
# dropout --> dropout rate of 0.2 randomly drops 20% of the neurons to prevent overfitting.

model = keras.models.Sequential(
    [
        keras.layers.Dense(64, activation = 'relu', input_dim = X_train.shape[1]),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation = 'relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1), # output layer
    ]
)

# compiling and training the model
# learn rate default = 0.01
# adam = adaptive moment estimation
model.compile(optimizer = Adam(learning_rate = 0.001), loss = "mean_squared_error", metrics = ['mean_absolute_error'])

# epoch = how many times model goes through data
# x & y train are hte targets
# batch size = update weights after the model had processed ix x times
# validation data model eveluates  performance after each epoch
history = model.fit(X_train, Y_train, epochs = 69, batch_size = 10, validation_data = (X_test, Y_test), validation_split = 0.2 )


# creating space from content to content
print("\n\n")



# ======================== PREDICTION OF MODEL =========================
# predicting the car purchase amount using training model
new_model_pred = model.predict(X_test)
model_pred = y_scaler.inverse_transform(new_model_pred)
orig_y_test = y_scaler.inverse_transform(Y_test)

# calculating  root mean sqaure for given mdoel
rmse_keras = mse(orig_y_test, model_pred)
print(f"Keras Model mean squared error is: {rmse_keras}")


# creating space from content to content
print("\n\n")
print("\n\n")



# ===================================================================================================================================================
# ============================================================== REGRESSION MODEL COMPARISON ========================================================
# ===================================================================================================================================================


# ========================= IMPORT & INITIALISE AI MODELS  =========================
# already existing regression models
# assign variables and initialise models
line_reg = LinearRegression()
lasso_reg = Lasso()
ridge_reg = Ridge()
elastic_reg = ElasticNet(alpha = 1.0, l1_ratio = 0.5)
desTree_reg = DecisionTreeRegressor()
svm_reg = SVR()
xgr_reg = XGBRegressor()
randForest_reg = RandomForestRegressor()
bays_reg = BayesianRidge()





# ========================= TRAINING MODELS  =========================
# use .fit -->  method is how a machine learning model learns from the training data. It adjusts the model's parameters so that it can make accurate predictions.
# training each model on the dataset training
line_reg.fit(X_train, Y_train)
lasso_reg.fit(X_train, Y_train)
ridge_reg.fit(X_train, Y_train)
elastic_reg.fit(X_train, Y_train)
desTree_reg.fit(X_train, Y_train)
svm_reg.fit(X_train, Y_train)
xgr_reg.fit(X_train, Y_train)
randForest_reg.fit(X_train, Y_train)
bays_reg.fit(X_train, Y_train)



# ========================= PREDICT TEST DATA =========================
# predicting the x_test  to the models variables from earlier
# predicting car purchase amount using traingg models
# assigned variables
line_pred =  y_scaler.inverse_transform(line_reg.predict(X_test).reshape(-1,1))
lasso_pred = y_scaler.inverse_transform(lasso_reg.predict(X_test).reshape(-1,1))
ridge_pred = y_scaler.inverse_transform(ridge_reg.predict(X_test).reshape(-1,1))
elastic_pred = y_scaler.inverse_transform(elastic_reg.predict(X_test).reshape(-1,1))
desTree_pred = y_scaler.inverse_transform(desTree_reg.predict(X_test).reshape(-1,1))
svm_pred = y_scaler.inverse_transform(svm_reg.predict(X_test).reshape(-1,1))
xgr_pred = y_scaler.inverse_transform(xgr_reg.predict(X_test).reshape(-1,1))
randForest_pred = y_scaler.inverse_transform(randForest_reg.predict(X_test).reshape(-1,1))
bays_pred = y_scaler.inverse_transform(bays_reg.predict(X_test).reshape(-1,1))





# ============ MODEL PERFORMANCE =============
# use RMSE --> Root Mean Squared Error
# mean_squared_error alias = mse
# calculating root mean square for given model
rmse_linear = mse(orig_y_test, line_pred)
rmse_lasso = mse(orig_y_test, lasso_pred)
rmse_ridge = mse(orig_y_test, ridge_pred)
rmse_elastic = mse(orig_y_test, elastic_pred)
rmse_desTree = mse(orig_y_test, desTree_pred)
rmse_svm = mse(orig_y_test, svm_pred)
rmse_xgr = mse(orig_y_test, xgr_pred)
rmse_randForest = mse(orig_y_test, randForest_pred)
rmse_bays = mse(orig_y_test, bays_pred)

# printing root mean square for given model
print(f"Linear Model root mean squared is: {rmse_linear}")
print(f"Lasso Model root mean squared is: {rmse_lasso}")
print(f"Ridge Model root mean squared is: {rmse_ridge}")
print(f"Elastic Model root mean squared is: {rmse_elastic}")
print(f"Descion Tree Model root mean squared is: {rmse_desTree}")
print(f"Support Vector Model root mean squared is: {rmse_svm}")
print(f"Extreme Gradient Model root mean squared is: {rmse_xgr}")
print(f"Random Forest Model root mean squared is: {rmse_randForest}")
print(f"Bayesian Ridge Model root mean squared is: {rmse_bays}")


# creating space from content to content
print("\n\n")




# ======================== GRAPHING =========================
# graph title for streamlit app
stl.title("New Model Car Purchase Amount Prediction")


# dataframe comparison prediction for each model
df_cpd_results = pd.DataFrame(
    {
        "Models":["Keras", "Linear Regression", "Lasso Regression" , "Ridge Regression" , "Elastic Regression", "Desicion Tree Regression", "SVR", "XGBoost Regression", "Random Forest Regression", "Baysian Ridge Regression"],
        "RMSE": [rmse_keras, rmse_linear, rmse_lasso, rmse_ridge, rmse_elastic, rmse_desTree, rmse_svm, rmse_xgr, rmse_randForest, rmse_bays]
    }
)

#displaying original vs predicted
stl.write("Regression Model Comparision")


bar_chart = atr.Chart(df_cpd_results).mark_bar().encode(
    x = atr.X("Models", sort = None),
    y = atr.Y("RMSE"),
    tooltip = ["Models", "RMSE"]
).properties(
    title = "New CPD Neural Network Model",
    width = 600,
    height = 400
)

# display the chart --> will occur in new window on browser
stl.altair_chart(bar_chart, use_container_width = True)

# streamlit run "file Name"
# streamlit run creatingAIModel.py
