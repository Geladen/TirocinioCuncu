import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def select_from_model(X,X_norm, y,num_feats):
    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
    embeded_lr_selector.fit(X_norm, y)

    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
    return embeded_lr_feature

case=['5','10','20','30','40','50','60','70','80','90','100','110','120','130','140','150']

num_feature = 100 # numero feature da scegliere

for casa in case:
    df = pd.read_csv('./full_features/feature_vector'+casa+'.csv') 

    numcols = list(df.columns)

    df = df[numcols]
    traindf = df[numcols]
    features = traindf.columns

    traindf = traindf.dropna()
    traindf = pd.DataFrame(traindf,columns=features)
    y = traindf['avg Consumption']
    X = traindf.copy()
    del X['avg Consumption']
    X_norm = MinMaxScaler().fit_transform(X)
    

    cor_feature = select_from_model(X, X_norm,y, num_feature)
    print(casa,str(len(cor_feature)))
    #print(casa,cor_feature)
    
    df1 = X[cor_feature]
    df1['avg Consumption'] = traindf['avg Consumption']
    df1.to_csv(r'./selected_vectors'+casa+'/selected_sfm'+str(num_feature)+'.csv', index = False)
    