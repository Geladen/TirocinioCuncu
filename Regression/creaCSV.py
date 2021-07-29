import csv
import psycopg2
import datetime
import time
import re
import numpy as np
import hashlib


# prende data e restituisce i minuti dalla mezzanotte
def getMinutes(t):
    return int(datetime.timedelta(hours=t.hour, minutes=t.minute,seconds=t.second).total_seconds()//60)

def getSeconds(t):
    return int(datetime.timedelta(hours=t.hour, minutes=t.minute,seconds=t.second).total_seconds())

# prende minuti e restituisce la data
def getTime(t):
    return str(datetime.timedelta(minutes=t))    


def trendline(index,data, order=1):
    coeffs = np.polyfit(index, list(data), order)
    slope = coeffs[-2]
    return float(slope)

# crea il file csv
def createCSV(temp,activity,sMov,sDoor,task,action,time):

    totalTime=0
    PATH='./feature_vector'+str(time)+'.csv' #path file da scrivere
    N_PATIENTS = 400 # EVENTUALMENTE TOGLIERE I PAZIENTI CON PROBLEMI DI SALUTE

    N_SENS_TEMP = 5
    N_ACTIVITY = 16 # vero numero altrimenti 24
    N_SENS_MOV = 52
    N_SENS_DOOR = 18
    N_TASK = 13
    N_ACTION = 12
    CONS = 3 # numero di campi del consumo
    NP = 274 # numero pazienti
    TIME = time
    ip=0

    # toglie i campi che non si voglio inserire nel file
    if not temp:
        N_SENS_TEMP = 0
    if not activity:
        N_ACTIVITY = 0
    if not sMov:
        N_SENS_MOV = 0
    if not sDoor:
        N_SENS_DOOR = 0
    if not task:
        N_TASK = 0
    if not action:
        N_ACTION = 0

    SIZE_VECTOR = 1+N_SENS_TEMP+N_SENS_MOV+N_SENS_DOOR+N_ACTIVITY+N_TASK+N_ACTION+NP+CONS

    conn = psycopg2.connect("dbname = 'CASAS400-power' user = 'postgres' host = 'localhost' password = 'password'")
    cur = conn.cursor()

    with open(PATH, mode='w',newline='') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        head=["Time"]
        head_temp = ["avg t00"+str(i) for i in range(1,N_SENS_TEMP+1)]
        head_activity = ["time (s) Activity"+str(i) for i in range(1,N_ACTIVITY+1)]
        head_mov = ["n m"+'0'*(3-len(str(i)))+str(i) for i in range(1,N_SENS_MOV+1)]
        head_door = ["n d"+'0'*(3-len(str(i)))+str(i) for i in range(1,N_SENS_DOOR+1)]
        head_task = ["n Task"+str(i) for i in range(1,N_TASK+1)]
        head_action = ["n Action"+str(i) for i in range(1,N_ACTION+1)]
        head_p = ["n paz"+str(i) for i in range(1,NP+1)]
        head+=head_temp+head_activity+head_mov+head_door+head_task+head_action+head_p+["trend"]+["act Consumption"]+["avg Consumption"]
        #print(head)

        #intestazione csv
        employee_writer.writerow(head)

        prev_t = [0] * N_SENS_TEMP

        for i in range(1,N_PATIENTS+1):
            
            #controllo esistenza dati dei pazienti
            try:
                # query senza pazienti con malattie
                cur.execute("SELECT time FROM events JOIN patients ON (events.patient = patients.patient_id) WHERE patient = {} AND (patients.diagnosis <> 1 AND patients.diagnosis <> 2) order by time LIMIT 1".format(i))
                #cur.execute("SELECT time FROM events WHERE patient = {} LIMIT 1".format(i)) # query tutti i pazienti

                start = getMinutes(cur.fetchone()[0])
                end = start+TIME
                ip+=1

            except:
                continue

            print("Paziente:",i)


            while True:

                # azzera strutture dati per il prossimo ciclo
                f_vector =[0] * SIZE_VECTOR
                avg_t = [0] * N_SENS_TEMP
                avg_p = 0
                f_vector[0] = end

                #print(prev_t)

                # gestione sensori temperatura
                if temp:
                    for j in range(1,N_SENS_TEMP+1):                    
                        cur.execute("SELECT value FROM events WHERE patient = {} AND sensor = '{}'".format(i,'t00'+str(j)))
                        for k in range(cur.rowcount):  
                            row = cur.fetchone()
                            f_vector[j] += float(row[0])
                            avg_t[j-1] += 1
                            
                        if avg_t[j-1] == 0:
                            f_vector[j] = prev_t[j-1] # mette 0 dove non è registrato
                            #f_vector[j] = -1 # mette -1 dove non è registrato
                        else:
                            f_vector[j] = f_vector[j]/avg_t[j-1]
                            prev_t[j-1] = f_vector[j]

                # gestione attività svolte
                if activity:
                    for j in range(1,N_ACTIVITY+1):
                        # attvità che iniziano e/o finiscono nel range
                        cur.execute("SELECT activities.start, activities.end  FROM activities WHERE patient = {} AND activity_type = '{}' AND (activities.start BETWEEN '{}' AND '{}' OR  activities.end BETWEEN '{}' AND '{}')".format(i,j,getTime(start),getTime(end),getTime(start),getTime(end)))

                        for k in range(cur.rowcount):                      
                            row = cur.fetchone()
                            start_act = getSeconds(row[0])
                            end_act = getSeconds(row[1])
                            if getMinutes(row[0]) < start:
                                start_act = start*60
                            elif getMinutes(row[1]) > end:
                                end_act = end*60

                            f_vector[N_SENS_TEMP+j] += end_act-start_act
                          
                # gestione sensori di movimento
                if sMov:
                    for j in range(1,N_SENS_MOV+1):    
                        cur.execute("SELECT value FROM events WHERE patient = {} AND sensor = '{}' AND time BETWEEN '{}' AND '{}'".format(i,'m'+'0'*(3-len(str(j)))+str(j),getTime(start),getTime(end)))
                        
                        for k in range(cur.rowcount):  
                            row = cur.fetchone()
                            f_vector[N_SENS_TEMP+N_ACTIVITY+j] += 1
                           
                # gestione sensori porte
                if sDoor:
                    for j in range(1,N_SENS_DOOR+1):
                        cur.execute("SELECT * FROM events WHERE patient = {} AND sensor = '{}' AND time BETWEEN '{}' AND '{}'".format(i,'d'+'0'*(3-len(str(j)))+str(j),getTime(start),getTime(end)))
                        
                        for k in range(cur.rowcount):  
                            row = cur.fetchone()
                            f_vector[N_SENS_TEMP+N_ACTIVITY+N_SENS_MOV+j] += 1
                            
                # gestione task eseguiti
                if task:
                    for j in range(1,N_TASK+1):
                        cur.execute("SELECT task FROM tasks WHERE patient = {} AND task ={} AND time BETWEEN '{}' AND  '{}'".format(i,j,getTime(start),getTime(end)))
                        
                        for k in range(cur.rowcount):  
                            row = cur.fetchone()
                            f_vector[N_SENS_TEMP+N_ACTIVITY+N_SENS_MOV+N_SENS_DOOR+j] += 1
                         
                # gestione azioni eseguite
                if action:
                    for j in range(1,N_ACTION+1):
                        cur.execute("SELECT action_type FROM tasks JOIN task_types ON (tasks.task = task_types.task_id AND tasks.activity = task_types.activity_id) WHERE patient ={} AND action_type={} AND time BETWEEN '{}' AND  '{}'".format(i,j,getTime(start),getTime(end)))
                        
                        for k in range(cur.rowcount):  
                            row = cur.fetchone()
                            f_vector[N_SENS_TEMP+N_ACTIVITY+N_SENS_MOV+N_SENS_DOOR+N_TASK+j] += 1


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
                    #print(index[j],cons[j])
                    f_vector[-2] += float(row[0])
                    avg_p += 1

                if len(index) == 1:
                    f_vector[-3] = 0                
                else:
                    f_vector[-3] = trendline(index,cons)

                f_vector[-2] = f_vector[-2]/avg_p

                f_vector[N_SENS_TEMP+N_ACTIVITY+N_SENS_MOV+N_SENS_DOOR+N_TASK+N_ACTION+ip] = 1


                # aggiorna l'orario
                start = end
                totalTime+=TIME
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

                
                employee_writer.writerow(f_vector) # scrive nel file

        cur.close()
        print("time", TIME)
        print("total tim: ",totalTime)
        

if __name__ == "__main__":


    time=[5,10,20,40,30,150,140,130,120,110,100,90,80,70,60,50]


    for t in time:
        createCSV(True,True,True,True,True,True,t)     # temperatura,attività,movimento,porte,task,azioni,tempo