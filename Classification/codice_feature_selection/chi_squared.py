import csv
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

def chi_squared(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_feature


case=['5','10','20','30','40','50','60','70','80','90','100','110','120','130','140','150']

num_feature = 100 # numero di feature da scegliere

for c in case:
    df = pd.read_csv('./full_features/feature_vector'+c+'.csv') 

    numcols = list(df.columns)

    df = df[numcols]
    traindf = df[numcols]
    features = traindf.columns

    traindf = traindf.dropna()
    traindf = pd.DataFrame(traindf,columns=features)
    y = traindf['avg Consumption']
    X = traindf.copy()

    del X['avg Consumption']

    cor_feature = chi_squared(X, y, num_feature)
    print(c,str(len(cor_feature)))
    #print(casa,cor_feature)

    df1 = X[cor_feature]
    df1['avg Consumption'] = traindf['avg Consumption']
    df1.to_csv(r'./selected_vectors'+c+'/selected_chi'+str(num_feature)+'.csv', index = False)
