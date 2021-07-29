import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def select_from_model_tree(X, y,num_feats):
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embeded_rf_selector.fit(X, y)

    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    return embeded_rf_feature


case=['5','10','20','30','40','50','60','70','80','90','100','110','120','130','140','150']
num_feature = 350# numero di features da scegliere

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

    cor_feature = select_from_model_tree(X,y, num_feature)
    print(c,str(len(cor_feature)))
    #print(casa,cor_feature)
    
    df1 = X[cor_feature]
    df1['avg Consumption'] = traindf['avg Consumption']
    df1.to_csv(r'./selected_vectors'+c+'/selected_sfmt'+str(num_feature)+'.csv', index = False)
    