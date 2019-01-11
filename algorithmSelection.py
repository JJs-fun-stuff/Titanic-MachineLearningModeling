import numpy
from pandas import get_dummies
from matplotlib import pyplot
from pandas import DataFrame
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from pickle import dump

# load file
testData = read_csv('test.csv')
trainData = read_csv('train.csv')

# Data Summarize
set_option('max_columns', 50)
set_option('display.width',1000)
print(trainData.describe())
print(trainData.head(20))
print(trainData.dtypes)

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
X = cleanedTrainData.values
Y = trainData[targetCol].values


# Data visualizations
# scatter_matrix(trainData)
# pyplot.show()

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 7)

# Standardize Data and modeling
pipelines = []
pipelines.append(('ScaledLR',Pipeline([('Scaler',StandardScaler()),('LR',LogisticRegression())])))
pipelines.append(('ScaledCART',Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeClassifier())])))
pipelines.append(('ScaledKNN',Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsClassifier())])))
pipelines.append(('ScaledLDA',Pipeline([('Scaler',StandardScaler()),('LDA',LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledGNB',Pipeline([('Scaler',StandardScaler()),('GNB',GaussianNB())])))
pipelines.append(('ScaledSVC',Pipeline([('Scaler',StandardScaler()),('SVC',SVC())])))
pipelines.append(('ScaledAB',Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostClassifier())])))
pipelines.append(('ScaledGBM',Pipeline([('Scaler',StandardScaler()),('GBM',GradientBoostingClassifier())])))
pipelines.append(('ScaledRF',Pipeline([('Scaler',StandardScaler()),('RF',RandomForestClassifier())])))
pipelines.append(('ScaledET',Pipeline([('Scaler',StandardScaler()),('ET',ExtraTreesClassifier())])))

# Algorithm Comparison
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits = 10, random_state = 7)
    result = cross_val_score(model,X,Y,cv = kfold,scoring = 'accuracy')
    results.append(result)
    names.append(name)
    print(name,result.mean())


fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# The best is SVC
