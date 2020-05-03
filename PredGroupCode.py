"""
Created on Wed Nov 27 16:13:52 2019

"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

# applying target encoder
def targetEncode(training_set, target_variable, cat_cols, alpha):
    mean = training_set[target_variable].mean()
    map_categories = dict()
    map_defaultVal = dict()

    for col in cat_cols:
        cat_count = training_set.groupby(col).size()
        target_cat_mean = training_set.groupby(col)[target_variable].mean()
        reg_smooth_val = ((target_cat_mean * cat_count) + (mean * alpha))/(cat_count + alpha)

        training_set.loc[:, col] = training_set[col].map(reg_smooth_val)
        training_set[col].fillna(mean, inplace =True)

        map_categories[col] = reg_smooth_val
        map_defaultVal[col] = mean
    return training_set, map_categories, map_defaultVal

# filling the missing values for numerical columns
def fill_missing_NaN_values(dataframe, num_cols):
    data = {}
    for column in num_cols:
        data[column] = dataframe[column].mean()
    return dataframe.fillna(value = data)

#null values
nullVals={
	'Year of Record': ["#N/A"],
    'Housing Situation': ["0", "nA"],
    'Work Experience in Current Job [years]': ["#NUM!"],
    'Satisfation with employer': ["#N/A"],
    'Gender': ["#N/A", "0", "unknown"],
    'Country': ["0"],
    'Profession': ["#N/A"],
    'University Degree': ["0", "#N/A"],
    'Hair Color': ["#N/A", "0", "Unknown"]
}

#reading the train data
trainData= pd.read_csv('tcd-ml-1920-group-income-train.csv', na_values=nullVals, low_memory=False)
trainData = trainData.drop(['Instance', 'Wears Glasses'], axis = 1)

#renaming the "Yearly Income in addition to Salary (e.g. Rental Income)" to "AdditionalIncome"
renameCol = {
    "Yearly Income in addition to Salary (e.g. Rental Income)": "AdditionalIncome"
}

trainData = trainData.rename(columns=renameCol)

trainData['AdditionalIncome'] = trainData.AdditionalIncome.str.split(' ').str[0].str.strip()
trainData['AdditionalIncome'] = trainData['AdditionalIncome'].astype('float64')

#removing duplicates
trainData = trainData.drop_duplicates()

#target encode on the categorical columns 
categoricalCols = ['Housing Situation', 'Satisfation with employer', 'Gender', 'Country', 'Profession', 'University Degree', 'Hair Color']
targetVar = 'Total Yearly Income [EUR]'
train_data, map_categories, map_defaultVal = targetEncode(trainData, targetVar, categoricalCols, 10)

#imputing missing values for numerical columns 
numericalCols = ['Year of Record', 'Age', 'Work Experience in Current Job [years]']
train_data = fill_missing_NaN_values(train_data, numericalCols)

train_dataX = train_data.iloc[:, :-1]
train_dataY = train_data.iloc[:, -1]

# splitting the data into training set & testing set
XTrain, XTest, yTrain, yTest = train_test_split(train_dataX, train_dataY, test_size = 0.2, random_state = 7)

parameters = {
          'max_depth': 20,
          'learning_rate': 0.001,
          "boosting": "gbdt",
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
         }

train_Data = lgb.Dataset(XTrain, label=yTrain)
test_Data = lgb.Dataset(XTest, label=yTest)

predictor = lgb.train(parameters,
                train_Data,
                150000,
                valid_sets = [train_Data, test_Data],
                verbose_eval=1000,
                early_stopping_rounds=500)

yPred = predictor.predict(XTest)

print('Mean Absolute Error: ', metrics.mean_absolute_error(yTest, yPred))

#load the test dataset
testData = pd.read_csv('tcd-ml-1920-group-income-test.csv', na_values=nullVals, low_memory=False)
testData = testData.drop(['Instance', 'Wears Glasses'], axis = 1)
#ts_dataset.info()

testData = testData.rename(columns=renameCol)

testData['AdditionalIncome'] = testData.AdditionalIncome.str.split(' ').str[0].str.strip()
testData['AdditionalIncome'] = testData['AdditionalIncome'].astype('float64')

#mapping the test dataset with target encoding values
for col in categoricalCols:
    testData.loc[:, col] = testData[col].map(map_categories[col])
    testData[col].fillna(map_defaultVal[col], inplace =True)
    

#filling the missing numerical values in the test dataset
testData = fill_missing_NaN_values(testData, numericalCols)

test_dataX = testData.iloc[:, :-1]
test_dataY = testData.iloc[:, -1]

#predicting the dependant variable for the test dataset
test_yPred = predictor.predict(test_dataX)
