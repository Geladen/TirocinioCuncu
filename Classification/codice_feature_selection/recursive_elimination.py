import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def rec_feature_elimination(X,X_norm, y,num_feat):
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feat, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return rfe_feature


case=['5','10','20','30','40','50','60','70','80','90','100','110','120','130','140','150']
num_feature = 350 # numero features da scegliere

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
    X_norm = MinMaxScaler().fit_transform(X)
    

    cor_feature = rec_feature_elimination(X, X_norm,y, num_feature)
    print(c,str(len(cor_feature)))
    #print(casa,cor_feature)

    df1 = X[cor_feature]
    df1['avg Consumption'] = traindf['avg Consumption']
    df1.to_csv(r'./selected_vectors'+c+'/selected_fe'+str(num_feature)+'.csv', index = False)
