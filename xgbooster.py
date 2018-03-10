# smafjal
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
import data_preprocessing as data_loader

# Scored 0.77033 #features = 6

def train_xgbooster(train_X,train_Y,test_X,test_Y):
    print train_X.shape,train_Y.shape
    gbm = xgb.XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.05).fit(train_X, train_Y)
    pred_y = gbm.predict(test_X)
    print (accuracy_score(test_Y,pred_y))
    return gbm

def generate_submission(clf,data_test):
    passenger_id = data_test['PassengerId']
    # predictions = clf.predict(data_test)
    predictions = clf.predict(data_test.drop('PassengerId', axis=1))
    output = pd.DataFrame({ 'PassengerId' : passenger_id, 'Survived': predictions })
    output.to_csv('model/xgb-predicted-value-titanic.csv', index = False)
    print output.head()

def data_train_test_split(data_X,data_Y):
    num_test = 0.20
    train_X,test_X,train_Y,test_Y = train_test_split(data_X,data_Y, test_size=num_test, random_state=23)
    return train_X,train_Y,test_X,test_Y

def save_model(model,path):
    with open(path,'wb') as file:
        pickle.dump(model,file)

def load_model(path):
    with open(path,'rb') as file:
        model=pickle.load(file)
        return model

def main():
    train_path='data/train.csv'
    test_path='data/test.csv'

    train_X,train_Y = data_loader.preprocess_data(train_path,data_mode='train')
    data_test, _ = data_loader.preprocess_data(test_path,data_mode='test')

    train_X,train_Y,test_X,test_Y = data_train_test_split(train_X,train_Y)
    model=train_xgbooster(train_X=train_X,train_Y=train_Y,test_X=test_X,test_Y=test_Y)

    save_model(model,'model/xgb-model.pkl')
    generate_submission(model,data_test)


if __name__ == "__main__":
    main()
