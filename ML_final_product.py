import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
import random
import streamlit as st
import warnings
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

def categorise(row):
    if (row["pH"]>=6 and row["pH"]<=7) and (row["EC"]>=1260.0 and row["EC"]<=1610.0)and(row["hum"]>=50.0 and row["hum"]<=63.0):
        return 1 #1
    #High-1-3
    elif(row["pH"]>=7 ) and (row["EC"]>=1260.0 and row["EC"]<=1610.0)and(row["hum"]>=50.0 and row["hum"]<=63.0):
        return 2 #2
    elif(row["pH"]>=6 and row["pH"]<=7 ) and (row["EC"]>=1610.0)and(row["hum"]>=50.0 and row["hum"]<=63.0):
        return 3 #3
    elif(row["pH"]>=6 and row["pH"]<=7) and (row["EC"]>=1260.0 and row["EC"]<=1610.0)and(row["hum"]>=63.0):
        return 4 #4
    #low-1-3
    elif(row["pH"]<=6) and (row["EC"]>=1260.0 and row["EC"]<=1610.0)and(row["hum"]>=50.0 and row["hum"]<=63.0):
        return 5 #5
    elif(row["pH"]>=6 and row["pH"]<=7) and (row["EC"]<=1260.0)and(row["hum"]>=50.0 and row["hum"]<=63.0):
        return 6 #6
    elif(row["pH"]>=6 and row["pH"]<=7) and (row["EC"]>=1260.0 and row["EC"]<=1610.0)and(row["hum"]<=50.0):
        return 7 #7
    #High-2-3
    elif(row["pH"]>=7 ) and (row["EC"]>=1610.0)and(row["hum"]>=50.0 and row["hum"]<=63.0):
        return 8 #8
    elif(row["pH"]>=7 ) and (row["EC"]>=1260.0 and row["EC"]<=1610.0)and(row["hum"]>=63.0):
        return 9 #9
    elif(row["pH"]>=6 and row["pH"]<=7 ) and (row["EC"]>=1610.0)and(row["hum"]>=63.0):
        return 10 #10
    #low 2-3
    elif(row["pH"]<=6) and (row["EC"]<=1260.0 )and(row["hum"]>=50.0 and row["hum"]<=63.0):
        return 11 #11
    elif(row["pH"]<=6) and (row["EC"]>=1260.0 and row["EC"]<=1610.0)and(row["hum"]<=50.0):
        return 12 #12
    elif(row["pH"]>=6 and row["pH"]<=7) and (row["EC"]<=1260.0)and(row["hum"]<=50.0):
        return 13 #13
    #High-3-1
    elif(row["pH"]>=7) and (row["EC"]>=1610.0)and(row["hum"]>=63.0):
        return 14 #14
    #low-3-1
    elif (row["pH"]<=6 ) and (row["EC"]<=1260.0 )and(row["hum"]<=50.0):
        return 15 #15
    #high-2low-1-3
    elif(row["pH"]>=7) and (row["EC"]>=1610.0)and(row["hum"]<=50.0):
        return 16 #16
    elif(row["pH"]>=7 ) and (row["EC"]<=1260.0)and(row["hum"]>=63.0):
        return 17 #17
    elif(row["pH"]<=6) and (row["EC"]>=1610.0)and(row["hum"]>=63.0):
        return 18 #18
    #low-2high-1-3
    elif(row["pH"]<=6) and (row["EC"]<=1260.0 )and(row["hum"]>=63.0):
        return 19 #19
    elif(row["pH"]<=6) and (row["EC"]>=1610.0)and(row["hum"]<=50.0):
        return 20 #20
    elif(row["pH"]>=7) and (row["EC"]<=1260.0)and(row["hum"]<=50.0):
        return 21 #21
    #low high normal-6
    elif (row["pH"]<=6) and (row["EC"]>=1610.0)and(row["hum"]>=50.0 and row["hum"]<=63.0):
        return 22 #22
    elif (row["pH"]>=6 and row["pH"]<=7) and (row["EC"]<=1260.0)and(row["hum"]>=63.0):
        return 23 #23
    elif (row["pH"]>=7) and (row["EC"]<=1260.0)and(row["hum"]>=50.0 and row["hum"]<=63.0):
        return 24 #24
    elif (row["pH"]>=6 and row["pH"]<=7) and (row["EC"]>=1610.0)and(row["hum"]<=50.0):
        return 25 #25
    elif (row["pH"]>=7) and (row["EC"]>=1260.0 and row["EC"]<=1610.0)and(row["hum"]<=50.0):
        return 26 #26
    elif (row["pH"]<=6) and (row["EC"]>=1260.0 and row["EC"]<=1610.0)and(row["hum"]>=63.0):
        return 27

def action(label):
    if (label==1):
        return "Chiller off, TDS up & down pump off, pH up & down pump off" #1
    #High-1-3
    elif(label==2):
        return "pH down pump on" #2
    elif(label==3):
        return "TDS down pump on" #3
    elif(label==4):
        return "Chiller on" #4
    #low-1-3
    elif(label==5):
        return 'pH up pump on' #5
    elif (label==6):
        return 'Nutrition ab pump on' #6
    elif(label==7):
        return 'Humidifiers on' #7
    #High-2-3
    elif(label==8):
        return 'pH down pump on, TDS down pump on' #8
    elif(label==9):
        return 'pH down pump on, chiller on' #9
    elif(label==10):
        return 'TDS down pump on, chiller on' #10
    #low 2-3
    elif(label==11):
        return 'pH up pump on , nutrition ab pump on' #11
    elif(label==12):
        return 'pH up pump on, Humidifiers on' #12
    elif(label==13):
        return 'Nutrition ab pump on, Humidifiers on' #13
    #High-3-1
    elif(label==14):
        return 'pH down pump on, TDS down pump on, Dehumidifiers on' #14
    #low-3-1
    elif (label==15):
        return 'pH up pump on, nutrition ab pump on, Humidifiers on' #15
    #high-2low-1-3
    elif(label==16):
        return 'pH down pump on, TDS down pump on, Humidifiers on' #16
    elif(label==17):
        return 'pH down pump on, Dehumidifiers on' #17
    elif(label==18):
        return 'pH up pump on, TDS down pump on, Dehumidifiers on' #18
    #low-2high-1-3
    elif(label==19):
        return 'pH up pump on, nutrition ab pump on, Dehumidifiers on' #19
    elif(label==20):
        return 'pH up pump on, TDS down pump on, Humidifiers on' #20
    elif(label==21):
        return 'pH down pump on, nutrition ab on, Humidifiers on' #21
    #low high normal-6
    elif (label==22):
        return 'pH up pump on, nutrition ab pump on' #22
    elif (label==23):
        return 'Nutrition ab pump on, Dehumidifiers on' #23
    elif (label==24):
        return 'pH down pump on, nutrition ab pump on' #24
    elif (label==25):
        return 'TDS down pump on, Humidifiers on' #25
    elif (label==26):
        return 'pH down pump on, Humidifiers on' #26
    elif (label==27):
        return 'pH up pump on, Humidifiers on' #27
def user_report():
    ph= st.slider(' Select pH ',0.0,14.0,key='phval')  
    st.write("pH",ph)
    ec= st.slider(' Select ppm ',0.0,2000.0,key='ecval')    
    st.write("ppm",ec)
    temp_air=st.slider(' Select temp air  ',0.0,30.0,key='tempairval')  
    st.write("temp air",temp_air)
    temp_water=st.slider(' Select temp water  ',0.0,30.0,key='tempwaterval')  
    st.write("temp water ",temp_water)
    hum=st.slider( 'Select Humanity ' , 0 ,  100,key='humval')
    st.write("Humidity",hum)
  
    user_report_data = {
        'EC': ec,
        'PH': ph,
        'HUM': hum,
        'TEMP_AIR': temp_air,
        'TEMP_WATER': temp_water 

  }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data
def stack():
    clf = [('dtc',dtc),('svc',svc),('knn',knn),('mlp',mlp)] #list of (str, estimator) ('mlp',mlp),('nb',nb)
    #st.write("Odituu iruku...")
    stack_model = StackingClassifier( estimators = clf,final_estimator = rfc)
    #st.write("nadandutu iruku")
    stack_model.fit(X,y)
    #st.write("parandutuu iruku")
    user_result =stack_model.predict(user_data)
    st.subheader('Your Report: ')
    st.write(user_result,action(user_result))

    score = cross_val_score(stack_model,X,y,cv = 5,scoring = 'accuracy')
    st.write("The accuracy score of is:",score.mean())
    return score.mean()
data=pd.read_csv("D:\Collage\Project reports\AI  and ML\dataset.csv")
data["EC"]=data["EC"]*700 #converting EC to ppm700
data["label"]=data.apply(lambda row : categorise(row),axis=1)
X=data.drop(["label","created_at","entry_id"],axis=1)

y=data["label"]
pH_list = []
ppm_list=[]
hum_list=[]
airt_list=[]
watert_list=[]
for i in range(0, 1000):
    pH_list.append(round(random.uniform(0.0,14),2))
    ppm_list.append(random.randint(1000,1800))
    hum_list.append(round(random.uniform(30,100),2))
    airt_list.append(round(random.uniform(17,27),2))
    watert_list.append(round(random.uniform(15,25),2))
df = pd.DataFrame(list(zip(pH_list, ppm_list,airt_list,hum_list,watert_list)), columns =['pH', 'EC','temp_air','hum','temp_water']) 

added_df=pd.concat([data,df],ignore_index=True)

added_df["label"]=added_df.apply(lambda row : categorise(row),axis=1)

df1 = added_df[added_df.isna().any(axis=1)]
df1=df1.drop(['created_at','entry_id'],axis=1)
df1 = df1[df1.isna().any(axis=1)]
added_df = added_df.reset_index( drop=True)

X=added_df.drop(["label","created_at","entry_id"],axis=1)
# st.write(X.columns)
y=added_df['label']
# st.write(y.columns)
# X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=.3,random_state=22)
user_data = user_report()
st.subheader('User Data')
st.write(user_data)

dtc =  DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn =  KNeighborsClassifier()
nb = GaussianNB()
mlp = MLPClassifier(hidden_layer_sizes=(6,5),
                    random_state=5,
                    learning_rate_init=0.01)
svc = SVC(kernel = 'linear', random_state = 0)
clf = [dtc,rfc,knn,mlp,svc]

if(st.button("Check")):
    # for algo in clf:
    #     algo.fit(X,y)
    #     predictedClass=algo.predict(user_data)
    #     st.write("The"+str(algo)+" predicts ", predictedClass,action(predictedClass))
    #     score = cross_val_score( algo,X,y,cv = 5,scoring = 'accuracy')
    #     st.write("The accuracy score of {} is:".format(algo),score.mean())

    clf = [('dtc',dtc),('rfc',rfc),('knn',knn),('mlp',mlp)] #list of (str, estimator) ('mlp',mlp),('nb',nb)

    stack_model = StackingClassifier( estimators = clf,final_estimator = svc)
    stack_model.fit(X,y)
    user_result =stack_model.predict(user_data)
    st.subheader('Your Report: ')
    st.write(user_result,action(user_result))

    score = cross_val_score(stack_model,X,y,cv = 5,scoring = 'accuracy')
    st.write("The accuracy score of is:",score.mean())
