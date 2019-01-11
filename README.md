# Titanic-MachineLearningModeling

**algorithmSelection.py**
* load Data from test.py and train.py
* Cleaning Data : 
  * delete 100 % unique : ``PassengerId, Name, Ticket``
  * delete missing value : ``Cabin``
* Scaler : ``sklearn.preprocessing.StandardScaler()``
* CrossValidation : ``sklearn.model_selection.KFold(n_splits = 10, random_state = 7)``
* Scoring Metric : ``sklearn.metrics.accuracy_score()``
* Best Model : ``sklearn.svm.SVC()``

**algorithmTuning.py**
* Use ``sklearn.model_selection.GridSearchCV()``
* Best hyperparameter for SVC : ``C = 0.7 , kernel = 'rbf'``

**predictingDataFromModel**
* Result : ``accuracy : 0.7846``
