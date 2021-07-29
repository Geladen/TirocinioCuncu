import csv
from scipy.stats import pearsonr

name=["feature_vector5","feature_vector10","feature_vector20","feature_vector30","feature_vector40","feature_vector50","feature_vector60","feature_vector70","feature_vector80","feature_vector90","feature_vector100","feature_vector110","feature_vector120","feature_vector130","feature_vector140","feature_vector150"]
#name=["feature_vector5"]
#size=[6691]
size=[6691,3870,1902,1166,892,665,521,478,386,283,266,244,230,226,224,148] # numero di righe dei file ordinati da 5 a 150 minuti
j=0

for i in name:
    orig=[0]*size[j]
    prev=[0]*size[j]
    with open('./full_features/'+i+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:

            if line_count == 0: # intenstazioni
                line_count += 1

            else: # crea 2 liste una con i valori attuali (prev) e una con i valori da prevedere (orig)
                #orig[line_count] = row[-1]
                #print(line_count)
                orig[line_count-1] = float(row[-1])
                prev[line_count-1] = float(row[-2])
                line_count += 1

    j+=1

    # calculate Pearson's correlation
    corr, _ = pearsonr(prev, orig)
    print("file: "+i)
    print('Pearsons correlation: %.3f' % corr)
    print("-----------------------\n")
