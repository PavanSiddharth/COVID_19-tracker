
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('Train_dataset.xlsx')
dataset = dataset.drop(['Designation','Name'],axis=1)
dataset[['Children']] = dataset[['Children']].fillna(0)
dataset[['Occupation']] = dataset[['Occupation']].fillna('None')
dataset[['comorbidity']] = dataset[['comorbidity']].fillna('None')
dataset[['cardiological pressure']] = dataset[['cardiological pressure']].fillna('Normal')

# Handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dataset[['Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose']])
dataset[['Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose']] = imputer.transform(dataset[['Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose']])
dataset[['Insurance']] = dataset[['Insurance']].fillna(0)
dataset[['salary']] = dataset[['salary']].fillna(0)
dataset[['FT/month']] = dataset[['FT/month']].fillna(0)


# Test dataset
dataset1 = pd.read_excel('Test_dataset.xlsx')
dataset1 = dataset1.drop(['Designation','Name'],axis=1)
dataset1[['Children']] = dataset1[['Children']].fillna(0)
dataset1[['Occupation']] = dataset1[['Occupation']].fillna('None')
dataset1[['comorbidity']] = dataset1[['comorbidity']].fillna('None')
dataset1[['cardiological pressure']] = dataset1[['cardiological pressure']].fillna('Normal')

# Handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dataset[['Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose']])
dataset1[['Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose']] = imputer.transform(dataset1[['Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose']])
dataset1[['Insurance']] = dataset1[['Insurance']].fillna(0)
dataset1[['salary']] = dataset1[['salary']].fillna(0)
dataset1[['FT/month']] = dataset1[['FT/month']].fillna(0)


X_train = dataset.iloc[:, 1:25].values
X_train = pd.DataFrame(X_train)


X_test = dataset1.iloc[:, 1:25].values
X_test = pd.DataFrame(X_test)

#  Handle non numeric data
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

X_train[[0,1,2,4,5,8,11,12]] = handle_non_numerical_data(X_train[[0,1,2,4,5,8,11,12]])

X_test[[0,1,2,4,5,8,11,12]] = handle_non_numerical_data(X_test[[0,1,2,4,5,8,11,12]])





y_train = dataset.iloc[:, 25].values




# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting
y_pred = regressor.predict(X_test)

people_id = dataset1.iloc[:,0].values

result = np.column_stack((people_id,y_pred))

result = pd.DataFrame(result)

result.to_csv('Result1.csv', index=False)
