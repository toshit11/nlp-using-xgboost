# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from vecstack import stacking
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
# Importing the dataset
#dataset = pd.read_excel('train1.xlsx', delimiter = '\t', quoting = 3)
dataset1=pd.read_csv('train.csv')
dataset2=pd.read_csv('test.csv')
dataset=dataset1.append(dataset2)
# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 7613):
    text = re.sub('[^a-zA-Z]', ' ',dataset1['text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)

for i in range(0, 3263):
    text = re.sub('[^a-zA-Z]', ' ',dataset2['text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(corpus).toarray()
z=dataset.iloc[:,1].values

y = dataset1.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
X_train=X[:7613,:]
y_train=y
X_test=X[7613:,:]

models = [
    XGBClassifier(random_state=1, n_jobs=0, learning_rate=0.7, 
                  n_estimators=300, max_depth=7),
        
    XGBClassifier(random_state=100, n_jobs=-1, learning_rate=0.3, 
                  n_estimators=500, max_depth=3),
    XGBClassifier(random_state=43, n_jobs=1, learning_rate=0.5, 
                  n_estimators=500, max_depth=5),
]


S_train, S_test = stacking(models,                   
                           X_train, y_train, X_test,   
                           regression=False, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=accuracy_score, 
    
                           n_folds=4, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=100,    
         
                           verbose=2)

"""model=LogisticRegression(random_state = 43,tol=1e-5)
"""
model = XGBClassifier(random_state=20, n_jobs=1, learning_rate=0.3, 
                      n_estimators=500, max_depth=5)
 
model = model.fit(S_train, y_train)
y_pred = model.predict(S_test)
"""from sklearn.model_selection import GridSearchCV
parameters = [{'random': [1, 10,], 'learning_rate': [0.1,0.3]},
              {'max_depth': [1,3,5]}]
grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
"""
prediction = pd.DataFrame(y_pred, columns=['target']).to_csv('prediction1.csv')
