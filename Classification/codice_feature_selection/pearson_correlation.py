import csv
import pandas as pd
import numpy as np

def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        #print("-------------------------------------------------",X[i])
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


case=['5','10','20','30','40','50','60','70','80','90','100','110','120','130','140','150']

num_feature = 350 # numero feature da scegliere

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

    cor_support, cor_feature = cor_selector(X, y, num_feature)
    print(str(len(cor_feature)), 'selected features')
    print(cor_feature)

    df1 = X[cor_feature]
    df1['avg Consumption'] = traindf['avg Consumption']
    df1.to_csv(r'./selected_vectors'+c+'/selected_cs'+str(num_feature)+'.csv', index = False)
