#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pylab as plt
import scipy as sc
from mlxtend.classifier import StackingCVClassifier
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def load_data():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    return df_train,df_test

def clean_data(df_train, df_test):
    train_len = len(df_train.index)
    df = pd.concat([df_train, df_test], ignore_index=True)

    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).map({
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don":        "Royalty",
        "Sir" :       "Royalty",
        "Dr":         "Officer",
        "Rev":        "Officer",
        "the Countess":"Royalty",
        "Dona":       "Royalty",
        "Mme":        "Mrs",
        "Mlle":       "Miss",
        "Ms":         "Mrs",
        "Mr" :        "Mr",
        "Mrs" :       "Mrs",
        "Miss" :      "Miss",
        "Master" :    "Master",
        "Lady" :      "Royalty"}).fillna('Other')
    df = pd.concat([df, pd.get_dummies(df['Title'], prefix='Title')], axis=1)

    df['Sex_Val'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    
    df['Embarked'].fillna('S', inplace=True)
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
    df.drop(['Embarked'], axis=1, inplace=True)
    
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    
    df['AgeFill'] = df['Age']
    df['AgeFill'] = df['AgeFill'].groupby([df['Sex'], df['Pclass'], df['Title']]).apply(
        lambda x: x.fillna(x.median()))

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Singleton'] = df['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    df['SmallFamily'] = df['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    df['LargeFamily'] = df['FamilySize'].map(lambda s : 1 if 5<=s else 0)

    df['Cabin'].fillna('U', inplace=True)
    df['Cabin'] = df['Cabin'].map(lambda c: c[0])
    df = pd.concat([df, pd.get_dummies(df['Cabin'], prefix='Cabin')], axis=1)
    df.drop(['Cabin'], axis=1, inplace=True)
    
    df = pd.concat([df, pd.get_dummies(df['Pclass'], prefix='Pclass')], axis=1)
    df.drop(['Pclass'], axis=1, inplace=True)
    
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    
    df['Ticket'] = df['Ticket'].map(cleanTicket)
    df = pd.concat([df, pd.get_dummies(df['Ticket'], prefix='Ticket')], axis=1)
    df.drop(['Ticket'], axis=1, inplace=True)
    
    # Drop the columns we won't use:
    df = df.drop(['Name', 'Sex', 'Age', 'Title'], axis=1)
    
    features = list(df.columns)
    features.remove('PassengerId')
    features.remove('Survived')
    df[features] = StandardScaler().fit_transform(df[features])
    
    return df[0:train_len].drop(['PassengerId'], axis=1), df.iloc[train_len:].drop(['Survived'], axis=1)

def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for column in range(X.shape[1]):
        cur_score, cur_p = pearsonr(X[:,column], y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores), np.array(pvalues))

def multivariate_spearmanr(X, y):
    scores, pvalues = [], []
    for column in range(X.shape[1]):
        cur_score, cur_p = sc.stats.spearmanr(X[:,column], y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores), np.array(pvalues))

def feature_selector_pearsonr(X_train, y_train):
    # 相关系数法选择特征
    selector = SelectKBest(score_func=multivariate_pearsonr, k=26)
    selector.fit(X_train, y_train)
    return selector

def feature_selector_spearmanr(X_train, y_train):
    # 相关系数法选择特征
    selector = SelectKBest(score_func=multivariate_spearmanr, k=25)
    selector.fit(X_train, y_train)
    return selector

def feature_selector_embedded(X_train, y_train):
    # Embedded方法选择特征
    model = SelectFromModel(LogisticRegression())
    model.fit(X_train, y_train)
    return model

def feature_selector_rfe(X_train, y_train):
    # 递归消除法选择特征
    selector = RFE(estimator=LogisticRegression(), n_features_to_select=10)
    selector.fit(X_train, y_train)
    return selector

def feature_selector_chi2(X_train, y_train):
    # 卡方检验法选择特征
    selector = SelectKBest(score_func=chi2, k=20)
    selector.fit(X_train, y_train)
    return selector

def feature_selector_random_forest(X_train, y_train):
    clf = ExtraTreesClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    selector = SelectFromModel(clf, prefit=True)
    return selector

def select_features(df_train, df_test):
    X_train = df_train.drop(['Survived'], axis=1).values
    y_train = df_train['Survived'].values
    X_test = df_test.drop(['PassengerId'], axis=1).values
    selector = feature_selector_random_forest(X_train, y_train)
    X_train_reduced = selector.transform(X_train)
    X_test_reduced = selector.transform(X_test)
    return X_train_reduced, y_train, X_test_reduced

def run_data(X, y):
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, train_size=0.5)
    
    clfs = {
        'GBDT': {
            'estimator': GradientBoostingClassifier(),
            'parameter_grid' : {
                'n_estimators': [ 10, 100 ],
                'max_features': [ 'auto', 'sqrt', 'log2' ],
                'max_depth': [ 3, 5, 8, 15, 25, 30 ],
                'max_leaf_nodes': [ 3, 4, 6, 8, 10 ]
                #'min_samples_leaf': [ 2, 5 ],
                #'min_samples_split': [ 2, 5, 10 ]
            }
        },
        'Random Forest': {
            'estimator' : RandomForestClassifier(random_state=1),
            'parameter_grid': {
                'n_estimators': [ 200,210,240,250 ],
                'criterion': [ 'gini', 'entropy' ],
                'max_depth': [ 4, 5, 6, 7, 8, 9, 10, 11 ],
                'max_features': [ 'sqrt' ]
            }
        },
        'SVM': {
            'estimator': SVC(),
            'parameter_grid': {
                'C': [ 0.1, 1, 10 ],
                'gamma': [ 'auto', 0.1, 1, 10 ],
                'class_weight': [ 'balanced', None ],
                'kernel': [ 'rbf' ]
            }
        },
        'Logistic Regression' : {
            'estimator': LogisticRegression(),
            'parameter_grid': {
                'C': [ 0.001, 0.01, 0.1, 1, 10, 100, 1000 ],
                'penalty': [ 'l1', 'l2' ]
            }
        },
        'AdaBoost' : {
            'estimator': AdaBoostClassifier(),
            'parameter_grid': {
                'n_estimators': [ 50, 100 ]
            }
        }
    }
    cv = StratifiedKFold(n_splits=5)
    base_models = []
    for name, obj in clfs.iteritems():
        grid_search = GridSearchCV(obj['estimator'], 
                                   scoring='accuracy', param_grid=obj['parameter_grid'], 
                                   cv=cv,
                                   n_jobs=-1)
        grid_search.fit(X_dev, y_dev)
        print('[%s] Mean validation score: %0.2f' % (name, grid_search.best_score_))
        base_models.append(grid_search.best_estimator_)

    sclf = StackingCVClassifier(classifiers=base_models,
                                meta_classifier=LogisticRegression(),
                                cv=5)
    parameter_grid = {
        'meta-logisticregression__C': [ 0.001, 0.01, 0.1, 1, 10, 100, 1000 ],
        'meta-logisticregression__penalty': [ 'l1', 'l2' ]
    }
    grid_search = GridSearchCV(sclf, scoring='accuracy', param_grid=parameter_grid,
                               cv=cv, n_jobs=-1)
    grid_search.fit(X_test, y_test)
    print('[%s] Mean validation score: %0.2f' % ('StackingClassifier', grid_search.best_score_))
    
    return grid_search.best_estimator_

if __name__ == '__main__':
    df_train,df_test = load_data()
    df_train,df_test = clean_data(df_train, df_test)
    X_train_reduced,y_train,X_test_reduced = select_features(df_train, df_test)
    print(X_train_reduced.shape)
    clf = run_data(X_train_reduced, y_train)
    df_test['Survived'] = clf.predict(X_test_reduced).astype(int)
    df_test[['PassengerId', 'Survived']].to_csv('stacking-results.csv', index=False)
