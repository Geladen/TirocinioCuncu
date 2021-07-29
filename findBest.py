import pandas as pd
case=["selected_chi100","selected_chi150","selected_chi200","selected_chi250","selected_chi300","selected_chi350","selected_fe100","selected_fe150","selected_fe200","selected_fe250","selected_fe300","selected_fe350","selected_sfm100","selected_sfm150","selected_sfm200","selected_sfm250","selected_sfm300","selected_sfm350","selected_sfmt100","selected_sfmt150","selected_sfmt200","selected_sfmt250","selected_sfmt300","selected_sfmt350"]
#case=["selected_chi100","selected_chi150","selected_chi200","selected_chi250","selected_chi300","selected_chi350","selected_fe100","selected_fe150","selected_fe200","selected_fe250","selected_fe300","selected_fe350","selected_sfm100","selected_sfm150","selected_sfm200","selected_sfm250","selected_sfm300","selected_sfm350","selected_sfmt100","selected_sfmt150","selected_sfmt200","selected_sfmt250","selected_sfmt300","selected_sfmt350","selected_cs100","selected_cs150","selected_cs200","selected_cs250","selected_cs300","selected_cs350"]

#case=["feature_vector5"]
max = 0.12
for c in case:
    data=pd.read_excel('./Update_Label/selected_vectors10/risultati/'+c+'.xlsx', index_col=0,engine='openpyxl')
    colonna=data.iloc[:, [0]]
    cont=0

    # trova il risultato migliore nella lista dei file passati e lo stampa
    for index, row in colonna.iterrows():
        new=float(row.to_string().split()[2])
        if new > max:
            max = new

print(max)
