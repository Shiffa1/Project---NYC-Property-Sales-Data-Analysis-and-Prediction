import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Loading the Property Sales Dataset
processed_df = pd.read_csv('/home/laptop-obs-303/Downloads/DOWNLOADS FOLDER/Main Project/deployment/price predictor/data/processed_data.csv')


# Train_Test Splitting
from sklearn.model_selection import train_test_split

y = processed_df['sale_price']
X = processed_df.drop('sale_price', axis=1)

X_train ,X_test, y_train , y_test = train_test_split(X , y , test_size = 0.3, random_state = 42)


#from detailed modelling, RF Regressor gave best performance

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)

mean_squared_error(y_test,y_pred)

#RMSE for linear regression
np.sqrt(mean_squared_error(y_test,y_pred))

print("R^2: {}".format(rf_reg.score(X_test, y_test)))

#pickling the model (random forest without tuning as there is not much difference)
import pickle 

with open('rf_model.pkl', 'wb') as f: #'wb' instead 'w' for binary file
    pickle.dump(rf_reg, f) #dumping data to f
