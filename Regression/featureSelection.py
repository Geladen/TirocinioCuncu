import csv
import pandas as pd
import numpy as np
import arff
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_digits

def pandas2arff(df,filename,wekaname = "pandasdata",cleanstringdata=True,cleannan=True):
    """
    converts the pandas dataframe to a weka compatible file
    df: dataframe in pandas format
    filename: the filename you want the weka compatible file to be in
    wekaname: the name you want to give to the weka dataset (this will be visible to you when you open it in Weka)
    cleanstringdata: clean up data which may have spaces and replace with "_", special characters etc which seem to annoy Weka. 
                     To suppress this, set this to False
    cleannan: replaces all nan values with "?" which is Weka's standard for missing values. 
              To suppress this, set this to False
    """
    import re
    
    def cleanstring(s):
        if s!="?":
            return re.sub('[^A-Za-z0-9]+', "_", str(s))
        else:
            return "?"
            
    dfcopy = df #all cleaning operations get done on this copy

    
    if cleannan!=False:
        dfcopy = dfcopy.fillna(-999999999) #this is so that we can swap this out for "?"
        #this makes sure that certain numerical columns with missing values don't get stuck with "object" type
 
    f = open(filename,"w")
    arffList = []
    arffList.append("@relation " + wekaname + "\n")
    #look at each column's dtype. If it's an "object", make it "nominal" under Weka for now (can be changed in source for dates.. etc)
    for i in range(df.shape[1]):
        if dfcopy.dtypes[i]=='O' or (df.columns[i] in ["Class","CLASS","class"]):
            if cleannan!=False:
                dfcopy.iloc[:,i] = dfcopy.iloc[:,i].replace(to_replace=-999999999, value="?")
            if cleanstringdata!=False:
                dfcopy.iloc[:,i] = dfcopy.iloc[:,i].apply(cleanstring)
            _uniqueNominalVals = [str(_i) for _i in np.unique(dfcopy.iloc[:,i])]
            _uniqueNominalVals = ",".join(_uniqueNominalVals)
            _uniqueNominalVals = _uniqueNominalVals.replace("[","")
            _uniqueNominalVals = _uniqueNominalVals.replace("]","")
            _uniqueValuesString = "{" + _uniqueNominalVals +"}" 
            arffList.append("@attribute " + df.columns[i] + _uniqueValuesString + "\n")
        else:
            arffList.append("@attribute " + df.columns[i] + " real\n") 
            #even if it is an integer, let's just deal with it as a real number for now
    arffList.append("@data\n")           
    for i in range(dfcopy.shape[0]):#instances
        _instanceString = ""
        for j in range(df.shape[1]):#features
                if dfcopy.dtypes[j]=='O':
                    _instanceString+="\"" + str(dfcopy.iloc[i,j]) + "\""
                else:
                    _instanceString+=str(dfcopy.iloc[i,j])
                if j!=dfcopy.shape[1]-1:#if it's not the last feature, add a comma
                    _instanceString+=","
        _instanceString+="\n"
        if cleannan!=False:
            _instanceString = _instanceString.replace("-999999999.0","?") #for numeric missing values
            _instanceString = _instanceString.replace("\"?\"","?") #for categorical missing values
        arffList.append(_instanceString)
    f.writelines(arffList)
    f.close()
    del dfcopy
    return True

def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


def chi_squared(X, y,num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_feature

def rec_feature_elimination(X,X_norm, y,num_feat):
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feat, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return rfe_feature

def select_from_model(X,X_norm, y,num_feats):
    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
    embeded_lr_selector.fit(X_norm, y)

    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
    return embeded_lr_feature

def select_from_model_tree(X, y,num_feats):
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embeded_rf_selector.fit(X, y)

    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    return embeded_rf_feature

def make_df(cor_feature,df):
    df1 = df[cor_feature]
    return df1

def make_csv_from_index(cor_feature,df, numcols, attr):
    for i in attr:
        cor_feature.append(numcols[i-1])
    print(cor_feature)
    df1 = df[cor_feature]
    return df1

#esegue ogni metodo con lunghezza di feature : 350,300,250,200,250,200,150,100 e salva su diversi file
def generate_all_combo(df,numcols): 

    opt = [350,300,250,200,250,200,150,100]
    df = df[numcols]
    traindf = df[numcols]
    features = traindf.columns

    traindf = traindf.dropna()
    traindf = pd.DataFrame(traindf, columns = features)
    y = traindf['avg Consumption']
    X = traindf.copy()
    #del X['avg Consumption']
    y=y.astype('int')
    X_norm = MinMaxScaler().fit_transform(X)

    for i in opt:
        
        cor_support, cor_feature = cor_selector(X, y, i)
        df1 = make_df(cor_feature,df)
        df1.to_csv(r'./selected_vectors5/selected'+'_cs'+str(i)+'.csv', index = False)
        #print(cor_feature)
    
        if i < 400:
            cor_feature = chi_squared(X, y, i)
            if 'avg Consumption' not in cor_feature:
                cor_feature.append('avg Consumption')
            df1 = make_df(cor_feature,df)
            df1.to_csv(r'./selected_vectors5/selected'+'_chi'+str(i)+'.csv', index = False)
            #print(cor_feature)

        cor_feature = rec_feature_elimination(X,X_norm,y,i)
        df1 = make_df(cor_feature,df)
        df1.to_csv(r'./selected_vectors5/selected'+'_fe'+str(i)+'.csv', index = False)
        #print(cor_feature)

        if i < 400:
            cor_feature = select_from_model(X,X_norm,y,i)
            df1 = make_df(cor_feature,df)
            df1.to_csv(r'./selected_vectors5/selected'+'_sfm'+str(i)+'.csv', index = False)
            #print(cor_feature)

        if i < 400:
            cor_feature = select_from_model_tree(X,y,i)
            df1 = make_df(cor_feature,df)
            df1.to_csv(r'./selected_vectors5/selected'+'_sfmt'+str(i)+'.csv', index = False)
            #print(cor_feature)


#attr=[393,39,59,60,61,19,81,115,20,62,18,36,38,37,16,47,94,40,48,278,17,32,110,29,35,156,53,103,57,72,21,66,232,96,45,30,290,54,104,28,65,64,23,97,31,71,249,15,86,55,22,112,338,93,46,321,44,332,105,63,51,73,50,56,43,266,49,313,279,70,79,342,148,95,74,69,369,373,392,179,212,297,67,233,193,113,181,124,229,100,58,203,365,161,127,207,68,41,311,386,131,192,140,168,235,252,155,351,204,27,185,322,152,239,200,319,42,276,13,345,344,366,157,385,14,190,146,253,102,224,210,248,282,135,330,262,99,187,269,191,309,375,177,183,76,292,257,296,243,89,80,237,306,84,263,254,268,288,244,350,52,123,129,380,175,88,101,343,25,300,153,223,284,166,314,267,324,241,250,240,176,331,144,362,294,132,352,390,133,24,286,121,301,379,281,274,170,163,145,238,85,134,91,92,90,356,259,357,355,341,109,384,108,260,107,347,318,246,291,304,310,299,201,265,182,394]

df = pd.read_csv('./full_features/feature_vector5.csv') # legge csv con tutte le feature
numcols = list(df.columns)

generate_all_combo(df,numcols)

exit()

cor_feature=['time (s) Activity7', 'n paz260', 'n paz159', 'n paz166', 'n m020', 'n d013', 'n d004', 'n paz206', 'n paz195', 'n paz48', 'n paz155', 'n paz209', 'n paz30', 'n paz202', 'n paz83', 'n paz122', 'n paz97', 'n paz35', 'n paz77', 'n paz205', 'n paz68', 'n paz67', 'n m005', 'n paz223', 'n paz80', 'n paz43', 'n paz61', 'n paz139', 'n paz87', 'n paz242', 'time (s) Activity6', 'n paz42', 'n paz105', 'n paz265', 'n paz229', 'n paz241', 'n paz234', 'n paz38', 'n paz264', 'n paz118', 'n paz135', 'n paz237', 'n paz163', 'n paz130', 'time (s) Activity5', 'n paz5', 'n paz51', 'n paz101', 'n paz246', 'n paz108', 'n m011', 'n paz170', 'n paz113', 'n paz9', 'n paz23', 'n paz100', 'n paz191', 'n paz81', 'n paz75', 'n paz138', 'n paz147', 'n paz117', 'n Action12', 'n paz98', 'n paz156', 'n Action11', 'n paz47', 'n paz14', 'n paz269', 'n paz194', 'n paz243', 'n paz274', 'n m019', 'n m046', 'n paz125', 'n Task6', 'n paz90', 'n paz10', 'n paz44', 'n paz248', 'n paz86', 'n paz111', 'n m036', 'n Task8', 'n paz112', 'n paz7', 'n paz64', 'n Action8', 'n paz21', 'n paz76', 'n m012', 'n paz96', 'n paz72', 'n paz116', 'n d008', 'n paz203', 'n paz82', 'n m045', 'n m004', 'n paz218', 'n paz180', 'n paz95', 'n paz62', 'n paz32', 'n paz33', 'n paz85', 'trend', 'n paz256', 'n paz252', 'n d003', 'n m047', 'n m052', 'n Task3', 'n paz31', 'n paz225', 'n Action6', 'n d005', 'n d009', 'n m048', 'time (s) Activity4', 'n paz162', 'n paz196', 'n m027', 'n Action9', 'n paz149', 'n m021', 'time (s) Activity1', 'time (s) Activity3', 'n m034', 'avg t003', 'n m028', 'n m051', 'n m029', 'n m041', 'n Task13', 'n paz215', 'n m022', 'avg t002', 'n paz204', 'n m024', 'n Task1', 'time (s) Activity2', 'n paz221', 'n Action7', 'time (s) Activity16', 'n m033', 'n d012', 'time (s) Activity9', 'n paz132', 'n m049', 'n m009', 'n Task5', 'n m001', 'n m042', 'n m043', 'n m006', 'n Task12', 'n m032', 'n paz173', 'Time', 'n m008', 'n m023', 'n Task4', 'n d001', 'n paz115', 'n m044', 'time (s) Activity15', 'n m050', 'n m035', 'n Task11', 'n m031', 'n paz39', 'avg t004', 'n m013', 'n m007', 'n Action5', 'n m010', 'time (s) Activity11', 'n paz161', 'n m026', 'n m018', 'n Task2', 'n m025', 'time (s) Activity10', 'n m015', 'n m016', 'n m014', 'time (s) Activity12', 'avg t005', 'n m040', 'time (s) Activity14', 'n Action10', 'n d007', 'time (s) Activity13', 'n m039', 'n m038', 'n m037', 'n m017', 'prev Consumption', 'avg Consumption']

df1=make_df(cor_feature,df) # crea csv dato il nome delle colonne
df1.to_csv(r'./selected_vectors/feature_vector90_cs200.csv', index = False)
#arff.dump('./nomefile.arff', df1.values, names=df.columns)
#'''
