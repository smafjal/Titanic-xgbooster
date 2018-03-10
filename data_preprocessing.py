# smafjal
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin

# http://stackoverflow.com/a/25562948
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

def normalize_name(data):
    data['Class'] = data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    dis_dictionary = {
            "Capt":       "Officer",
            "Col":        "Officer",
            "Major":      "Officer",
            "Jonkheer":   "Royalty",
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
            "Lady" :      "Royalty"}

    data['Class'] = data.Class.map(dis_dictionary)
    return data

def encode_features(data):
    features = ['Sex','Class']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(data[feature])
        data[feature] = le.transform(data[feature])
    return data

def preprocess_data(path,data_mode='train'):
    data_df = pd.read_csv(path, header=0)
    data_df=normalize_name(data_df)
    features = ['Pclass','Sex','Age','Fare','Parch','Class']

    data_X = data_df[features]
    data_X = DataFrameImputer().fit_transform(data_X)

    data_X=encode_features(data_X)
    data_Y=None
    if data_mode == 'train':
        data_Y=data_df['Survived']
    else:
        data_X['PassengerId']=data_df['PassengerId']

    return data_X,data_Y


def main():
    train_path='data/train.csv'
    test_path='data/test.csv'
    train_X,train_Y=preprocess_data(train_path,data_mode='train')

    print train_X.shape,train_Y.shape
    print train_X.columns.values

    data_test,_=preprocess_data(test_path,data_mode='test')

    print data_test.shape
    print data_test.columns.values


if __name__ == "__main__":
    main()
