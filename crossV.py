import os
import pickle
import tempfile
import traceback
import javabridge
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.classifiers import Classifier
from weka.core.classes import serialization_read, serialization_write, serialization_read_all, serialization_write_all

from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.classifiers import FilteredClassifier
from weka.filters import Filter

import xlsxwriter

def main():
    """
    Just runs some example code.
    """
    #case=["selected_chi100","selected_chi150","selected_chi200","selected_chi250","selected_chi300","selected_chi350","selected_fe100","selected_fe150","selected_fe200","selected_fe250","selected_fe300","selected_fe350","selected_sfm100","selected_sfm150","selected_sfm200","selected_sfm250","selected_sfm300","selected_sfm350","selected_sfmt100","selected_sfmt150","selected_sfmt200","selected_sfmt250","selected_sfmt300","selected_sfmt350","selected_cs100","selected_cs150","selected_cs200","selected_cs250","selected_cs300","selected_cs350"]
    #case=["feature_vector100","feature_vector110","feature_vector120","feature_vector130","feature_vector140","feature_vector150","feature_vector90","feature_vector80","feature_vector70","feature_vector60","feature_vector50","feature_vector40","feature_vector30","feature_vector20","feature_vector10","feature_vector5"]
    case=["selected_chi100","selected_chi150","selected_chi200","selected_chi250","selected_chi300","selected_chi350","selected_fe100","selected_fe150","selected_fe200","selected_fe250","selected_fe300","selected_fe350","selected_sfm100","selected_sfm150","selected_sfm200","selected_sfm250","selected_sfm300","selected_sfm350","selected_sfmt100","selected_sfmt150","selected_sfmt200","selected_sfmt250","selected_sfmt300","selected_sfmt350"]
    #case=["selected_chi100","selected_chi150","selected_chi200","selected_fe100","selected_fe150","selected_fe200","selected_sfm100","selected_sfm150","selected_sfm200","selected_sfmt100","selected_sfmt150","selected_sfmt200","selected_cs100","selected_cs150","selected_cs200",]

    for nomefile in case:

        print(nomefile)
        feature_vector = "./selected_vectors30/"+nomefile+".arff" # legge il file in formato arff

        loader = Loader("weka.core.converters.ArffLoader")
        data = loader.load_file(feature_vector)
        data.class_is_last()

        #print(data)

        f=open("./selected_vectors30/risultati/"+nomefile+".txt","w+") # file risultati in txt

        #intestazione excel
        intest=["Correlation coefficient","Mean absolute error","Root mean squared error","Relative absolute error","Root relative squared error","Total Number of Instances"]
        workbook = xlsxwriter.Workbook("./selected_vectors30/risultati/"+nomefile+".xlsx") # file excel
        worksheet = workbook.add_worksheet()

        for col_num, dati in enumerate(intest):
                worksheet.write(0, col_num+1, dati)
        riga=1

        #lista degli algoritmi da eseguire
        #alg=["meta.Bagging","meta.RandomSubSpace","rules.M5Rules","trees.M5P","trees.RandomForest"]
        alg=["bayes.NaiveBayes","bayes.NaiveBayesUpdateable","functions.Logistic","functions.SGD","functions.SimpleLogistic","functions.SMO","functions.VotedPerceptron","meta.AdaBoostM1","meta.AttributeSelectedClassifier","meta.Bagging","meta.ClassificationViaRegression","meta.IterativeClassifierOptimizer","meta.LogitBoost","meta.RandomCommittee","meta.RandomSubSpace","rules.DecisionTable","rules.JRip","rules.OneR","trees.DecisionStump","trees.J48","trees.RandomForest","trees.REPTree"]
        
        for row_num, dati in enumerate(alg):
            worksheet.write(row_num+1, 0, dati)
        
        for i in alg:
            remove = Filter(classname="weka.filters.unsupervised.attribute.Remove")
            cls = Classifier(classname="weka.classifiers."+i) 
            fc = FilteredClassifier()
            fc.filter = remove
            fc.classifier = cls

            evl = Evaluation(data)
            evl.crossvalidate_model(fc, data, 10, Random(1)) # 10 fold cross validation
            #evl.evaluate_train_test_split(fc,data,50,None,None) # 50% split cross validation

            k=evl.summary()

            #scrittura sui file
            f.write(i+"\n")
            f.write(k+"\n")
            my_list = k.split('\n')
            for col_num, dati in enumerate(my_list):
                worksheet.write(riga, col_num, dati[-10:])
            print(i)
            riga+=1
        f.close()
        workbook.close()

if __name__ == "__main__":
    try:
        jvm.start(max_heap_size="14336m") # memoria ram da dare alla jvm
        main()
    except Exception as e:
        print(traceback.format_exc())
        print("Errore")
    finally:
        jvm.stop()