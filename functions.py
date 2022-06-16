import pandas as pd
import numpy as np
import re
import os
from sklearn.externals import joblib


emails_clean = pd.DataFrame()

def readxlsx(file):
    
    xldata = pd.ExcelFile(file)
    emails = xldata.parse()
    
    emails = pd.DataFrame(emails)
    return emails


#Function to combine subject and mail body, and drop columns not being used
def selectdata(emails):
    emails_select = emails.filter(['Subject', 'Body', 'Category'])

    return emails_select


#Function to remove the trailing mails except the last 2
def removetrailmail(emails_clean, n=2):       
    
    #Finding indices (of forward or reply) to separate mails (as required) from the trail. 
    trail_mail_indices = []
    for i in range(len(emails_clean)):
        if(isinstance(emails_clean.Body[i], str)):
            trail_mail = [m.start() for m in re.finditer('From:', emails_clean['Body'][i])]
        else:
            trail_mail = []
        trail_mail_indices.append(trail_mail)
    
    #Removing the trail mails based on value of n
    #Using trail mail indices to remove unwanted mail trails
    emails_single = emails_clean

    #Extracting the required mails only
    for i in range(len(emails_clean)):
        if(len(trail_mail_indices[i])>=n):
            emails_single['Body'][i] = emails_clean['Body'][i][:trail_mail_indices[i][n-1]]
    
    return emails_single


#Function to join email subject and body together
def joinsubject(emails_single):
    for i in range(0,len(emails_single)):
        emails_single['Body'][i]=str(emails_single['Subject'][i])+" "+str(emails_single['Body'][i])
    return emails_single


#Retraining data
def retrain(predictions, data_test, model_file):
    
    #Finding indices where data is wrongly classified
    retrain_indices = data_test[predictions != data_test.Category].index
    retrain_data = data_test.iloc[retrain_indices]
    
    #loading existing classification model
    text_model = joblib.load(os.path.join(".\\", model_file))
    
    #Retraining
    text_model_new = text_model.fit(retrain_data.Body.values.astype('U'), retrain_data.Category)
    model_file_new = model_file + "_new"
    
    #Saving model retrained with new data
    joblib.dump(text_model_new, os.path.join(".\\", model_file_new))