from pandas import read_csv
from pandas import get_dummies
from pandas import set_option
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pandas import DataFrame
from pickle import load

set_option('max_columns',50)
set_option('display.width',1000)
set_option('display.max_rows',1000)

# load data
trainData = read_csv('train.csv')
testData = read_csv('test.csv')

# load model
loaded_model = load(open('finalized_model.sav','rb'))

# train data feature selection
selFeature = list(trainData.columns.values)
selFeature.remove('PassengerId')
selFeature.remove('Name')
selFeature.remove('Cabin')
selFeature.remove('Ticket')
targetCol = 'Survived'
selFeature.remove(targetCol)

# clean train data
trainData.fillna(value = trainData.Age.mean(), inplace = True)
trainData.fillna(value = 'X', inplace = True)
cleanedTrainData = trainData[selFeature]
cleanedTrainData = get_dummies(cleanedTrainData,columns=['Embarked','Sex'],prefix=['Embarked','Sex'])
selClean = list(cleanedTrainData.columns.values)
selClean.remove('Embarked_29.6991176471')
cleanedTrainData = cleanedTrainData[selClean]
X = cleanedTrainData.values
Y = trainData[targetCol].values

# scaler
scaler = StandardScaler().fit(X,Y)
rescaledX = scaler.transform(X)

# test data feature selection
selFeature = list(testData.columns.values)
selFeature.remove('PassengerId')
selFeature.remove('Name')
selFeature.remove('Cabin')
selFeature.remove('Ticket')

# clean test data
testData.fillna(value = trainData.Age.mean(), inplace = True)
testData.fillna(value = 'X', inplace = True)
cleanedTestData = testData[selFeature]
cleanedTestData = get_dummies(cleanedTestData,columns=['Embarked','Sex'],prefix=['Embarked','Sex'])
cleanedTestData = cleanedTestData[selClean]
X_test = cleanedTestData.values
rescaledX_test = scaler.transform(X_test)

# save submission file
predictions = loaded_model.predict(rescaledX_test)
submission = DataFrame({'PassengerId': testData.PassengerId, 'Survived': [x for x in predictions]})
submission.to_csv('submission.csv', index=False)
