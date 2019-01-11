# Tune SVC
from pandas import read_csv
from pandas import DataFrame
from pandas import get_dummies
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pickle import dump

# load file
testData = read_csv('test.csv')
trainData = read_csv('train.csv')

# Cleaning Data
print(trainData.apply(lambda x : x.isnull().any()))
print(DataFrame({'percent missing': trainData.isnull().sum() * 100 / len(trainData)}))
print(DataFrame({'percent unique': trainData.apply(lambda x : x.unique().size / float(x.size) * 100)}))
# ignore 100 % unique data like passengerID ,Name and A lot of missing like Cabin

# Feature Selection
selFeature = list(trainData.columns.values)
selFeature.remove('PassengerId')
selFeature.remove('Name')
selFeature.remove('Cabin')
selFeature.remove('Ticket')
targetCol = 'Survived'
selFeature.remove(targetCol)

# fill missing data
trainData.fillna(value = trainData.Age.mean(), inplace = True)
trainData.fillna(value = 'X', inplace = True)
cleanedTrainData = trainData[selFeature]
cleanedTrainData = get_dummies(cleanedTrainData,columns=['Embarked','Sex'],prefix=['Embarked','Sex'])
selClean = list(cleanedTrainData.columns.values)
selClean.remove('Embarked_29.6991176471')
cleanedTrainData = cleanedTrainData[selClean]
X = cleanedTrainData.values
Y = trainData[targetCol].values

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 7)

# Hyperparameter tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# C = 0.7 kernel = 'rbf'

# finalized_model
model = SVC(C = 0.7,kernel = 'rbf')
model.fit(X_train, Y_train)

# Save model
filename = 'finalized_model.sav'
dump(model, open(filename, 'wb'))
