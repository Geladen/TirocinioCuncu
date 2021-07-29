import csv
import psycopg2
import datetime
import time
import re
import numpy as np
import hashlib

from scipy.stats import pearsonr

# prende data e restituisce i minuti dalla mezzanotte
def getMinutes(t):
    return int(datetime.timedelta(hours=t.hour, minutes=t.minute,seconds=t.second).total_seconds()//60)

def getSeconds(t):
    return int(datetime.timedelta(hours=t.hour, minutes=t.minute,seconds=t.second).total_seconds())

# prende minuti e restituisce la data
def getTime(t):
    return str(datetime.timedelta(minutes=t))    

#calcola trend
def trendline(index,data, order=1):
    coeffs = np.polyfit(index, list(data), order)
    slope = coeffs[-2]
    return float(slope) , float(coeffs[-1])

# crea il file csv
def easyPrev(time,size,isNominal):

    N_PATIENTS = 400 # EVENTUALMENTE TOGLIERE I PAZIENTI CON PROBLEMI DI SALUTE
    TIME = time
    ip=0
    SIZE_VECTOR = 5

    orig=[0]*size
    linePrev=[0]*size

    conn = psycopg2.connect("dbname = 'CASAS400-power' user = 'postgres' host = 'localhost' password = 'password'")
    cur = conn.cursor()

    cont=0
    for i in range(1,N_PATIENTS+1):

        try:
            # query senza pazienti con problemi di salute
            cur.execute("SELECT time FROM events JOIN patients ON (events.patient = patients.patient_id) WHERE patient = {} AND (patients.diagnosis <> 1 AND patients.diagnosis <> 2) order by time LIMIT 1".format(i))
            #cur.execute("SELECT time FROM events WHERE patient = {} LIMIT 1".format(i)) # query tutti i pazienti

            start = getMinutes(cur.fetchone()[0])
            end = start+TIME
            ip+=1

        except:
            continue

        while True:

            # azzera strutture dati per il prossimo ciclo
            f_vector =[0] * SIZE_VECTOR
            avg_p = 0
            f_vector[0] = end

            #print(prev_t)

            # gestione consumo elettrico
            cur.execute("SELECT value, events.time FROM events WHERE patient = {} AND sensor = '{}' AND time BETWEEN '{}' AND '{}'".format(i,'p001',getTime(start),getTime(end)))
            if cur.rowcount == 0: #il paziente ha finito le misurazioni
                break

            index = [0]*cur.rowcount
            cons = [0]*cur.rowcount
            for j in range(cur.rowcount):                      
                row = cur.fetchone()
                cons[j] = float(row[0])

                index[j] = float(getSeconds(row[1])/60)
                f_vector[-2] += float(row[0])
                avg_p += 1

            
            if len(index) == 1: #se c'è solo una lettura del consumo il trend è a 0
                f_vector[-4] = 0    
                f_vector[-3] = 0             
            else:
                f_vector[-4], f_vector[-3] = trendline(index,cons)

            f_vector[-2] = f_vector[-2]/avg_p

            # aggiorna l'orario
            start = end
            end = start+TIME

            app=0
            avg_p=0

            # gestione consumo elettrico prossimo campo
            cur.execute("SELECT value FROM events WHERE patient = {} AND sensor = '{}' AND time BETWEEN '{}' AND '{}'".format(i,'p001',getTime(start),getTime(end)))
            if cur.rowcount == 0: #il paziente ha finito le misurazioni
                break

            for j in range(cur.rowcount):                      
                row = cur.fetchone()
                
                app += float(row[0])
                avg_p += 1

            f_vector[-1] = app/avg_p

            if isNominal: #nel caso della classifiazione
                if f_vector[-1] > f_vector[-2]:
                    f_vector[-1] = "UP"
                else:
                    f_vector[-1] = "DOWN"

            orig[cont]=f_vector[-1]

            linePrev[cont] = f_vector[-4]*((start+end)/2) + f_vector[-3] #proiezione del trend nella prossima finestra temporale

            if linePrev[cont] == 0:
                linePrev[cont] = f_vector[-2]
                
            #print(f_vector)

            if isNominal:
                if f_vector[-4] > 0:
                    linePrev[cont] = "UP"
                else:
                    linePrev[cont] = "DOWN"

            cont+=1

    #print(orig)
    return orig, linePrev


if __name__ == "__main__":

    right=0
    time=[5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150] 

    size=[6691,3870,1902,1166,892,665,521,478,386,283,266,244,230,226,224,148] # numero righe csv ordinati da 5 a 150 minuti


    for i in range(len(time)):
        isNominal = False
        right=0
        orig , linePrev = easyPrev(time[i],size[i],isNominal)

        if isNominal: # calcolo percentuale

            for j in range(size[i]):
                #print(j)
                if orig[j] == linePrev[j]:
                    right+=1

            print("time: ",time[i])
            print("Correctly Classified Instances",right *100 / len(orig))
            print("-----------------------\n")

        else: #calcolo pearson coefficient
            for l in range(len(linePrev)):
                if linePrev[l] < 0:
                    linePrev[l]*=(-1)

            corr, _ = pearsonr(linePrev, orig)
            print("time: ",time[i])
            print('Pearsons correlation: %.3f' % corr)
            print("-----------------------\n")
