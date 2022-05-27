import pandas as pd
import os
import numpy as np
import csv
import tldextract
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ietfdata.mailarchive as ma

import email.header
import email.utils
import string
import sys
from dataclasses import dataclass, field
import datetime
from datetime import datetime
import json
from datetime import date

archive = ma.MailArchive()

ren = r'(?:\.?)([\w\-_+#~!$&\'\.]+(?<!\.)(@|[ ]?\(?[ ]?(at|AT)[ ]?\)?[ ]?)(?<!\.)[\w]+[\w\-\.]*\.[a-zA-Z-]{2,3})(?:[^\w])'
ren2 = r'([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'

emailID_pid_dict = json.load(open('data/emailID_pid_dict.json'))
pid_emailID_dict = json.load(open('data/pid_emailID_dict.json'))

role_based_emailIDs = set()
with open('data/role_based_emailIDs_newData.txt') as f:
#with open('./role_based_emailIDs_new.txt') as f:
    Lines = f.readlines()
    for line in Lines:
        line = line.strip()
        role_based_emailIDs.add(line)
        #print(line)
print("loaded role_based_emailIDs- ",len(role_based_emailIDs))

automated_list = set()
#with open('./automated_email_IDs.txt') as f2:
with open('data/automated_email_IDs_newData.txt') as f2:
    Lines2 = f2.readlines()
    for line in Lines2:
        line = line.strip()
        automated_list.add(line)
        #print(line)

print("loaded automated_list- ",len(automated_list))

emailID_first_email = json.load(open('data/emailID_first_email.json'))
emailID_yearly_monthly_vol_dict = json.load(open('data/emailID_yearly_monthly_vol_dict.json'))

def round_to(n, precision):
    correction = 0.5 if n >= 0 else -0.5
    return int( n/precision+correction ) * precision

def dfs(c, parent_node, id_age, personID_personID_type_mapping_df, mlist):
    
    #if len(parent_node.successors)>0:
    if len(parent_node.children)>0:
        #sister_nodes = []
        
        #for i in parent_node.successors:
        for i in parent_node.children:
            
            #if i.data is not None:
            if i is not None:
                
                e = i.from_addr
                e = e.replace("'", "__apostrophe__")
                x = re.findall(ren, str(e))
                if len(x) == 0:
                    x = re.findall(ren2, str(e))
                    if len(x) > 0:
                        email = x[0]
                else:
                    email = x[0][0]

                email = email.replace("__apostrophe__", "'").lower()
                
                if email in emailID_pid_dict and email not in automated_list:
                    personID = emailID_pid_dict[email]
                    
                    yrs = []
                    years_months = set([])
                    
                    for eid in pid_emailID_dict[str(personID)]:

                        if eid in emailID_first_email and eid in emailID_yearly_monthly_vol_dict:
                            date_f = emailID_first_email[eid]
                            yrs.append(date(datetime.strptime(date_f,"%Y-%m-%d").year,datetime.strptime(date_f,"%Y-%m-%d").month,datetime.strptime(date_f,"%Y-%m-%d").day))
                            
                            for yr in emailID_yearly_monthly_vol_dict[eid]:
                                for month in emailID_yearly_monthly_vol_dict[eid][yr]:
                                    years_months.add(date(int(yr),int(month),1))

                    if len(yrs) > 0 and len(years_months) > 0:
                        personID_first_email = min(yrs)
                        current_date = date(i.date.year, i.date.month, i.date.day)
                        age = (current_date - personID_first_email).days
                        age = round_to(age/365,0.5)
                        
                        personID_last_email = max(years_months)
                        max_age = (personID_last_email - min(years_months)).days #or can use personID_first_email
                        max_age = round_to(max_age/365,0.5)
                        
                        if personID != id_age[0]:
                            personID_personID_type_mapping_df.loc[0 if pd.isnull(personID_personID_type_mapping_df.index.max()) else personID_personID_type_mapping_df.index.max() + 1] = [personID,id_age[0],'reply_to',age,id_age[1],max_age,id_age[2],current_date,i.message_id,parent_node.message_id,mlist,maillist_type[mlist]]
                        else:
                            personID_personID_type_mapping_df.loc[0 if pd.isnull(personID_personID_type_mapping_df.index.max()) else personID_personID_type_mapping_df.index.max() + 1] = [personID,id_age[0],'reply_self',age,id_age[1],max_age,max_age,current_date,i.message_id,parent_node.message_id,mlist,maillist_type[mlist]]
                        
                        personID_personID_type_mapping_df = dfs(c, i, [personID, age, max_age], personID_personID_type_mapping_df, mlist)
                        
    return(personID_personID_type_mapping_df)


maillist_type = dict()
f = open('data/maillist_yearly_monthly_type_active_status.json')
maillist_yearly_monthly_type_active_status = json.load(f)
c = 0
for grp in maillist_yearly_monthly_type_active_status:
    flag = True
    c += 1
    for year in maillist_yearly_monthly_type_active_status[grp]:
        if flag:
            for month in maillist_yearly_monthly_type_active_status[grp][year]:
                if flag:
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'wg':
                        #list_WG.append(grp)
                        maillist_type[grp] = 'wg'
                        flag = False
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'rg':
                        #list_RG.append(grp)
                        maillist_type[grp] = 'rg'
                        flag = False
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'meeting':
                        #list_WG.append(grp)
                        maillist_type[grp] = 'meeting'
                        flag = False
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'dir':
                        #list_WG.append(grp)
                        maillist_type[grp] = 'dir'
                        flag = False
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'review':
                        #list_WG.append(grp)
                        maillist_type[grp] = 'review'
                        flag = False
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'announcements':
                        #list_WG.append(grp)
                        maillist_type[grp] = 'announcements'
                        flag = False
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'iab':
                        #list_WG.append(grp)
                        maillist_type[grp] = 'iab'
                        flag = False
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'rag':
                        #list_WG.append(grp)
                        maillist_type[grp] = 'rag'
                        flag = False
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'ag':
                        #list_WG.append(grp)
                        maillist_type[grp] = 'ag'
                        flag = False
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'team':
                        #list_WG.append(grp)
                        maillist_type[grp] = 'team'
                        flag = False
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'ietf@ietf.org':
                        #list_WG.append(grp)
                        maillist_type[grp] = 'ietf@ietf.org'
                        flag = False
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'program':
                        #list_WG.append(grp)
                        maillist_type[grp] = 'program'
                        flag = False
                    if maillist_yearly_monthly_type_active_status[grp][year][month]['type'] == 'nomcom':
                        #list_WG.append(grp)
                        maillist_type[grp] = 'nomcom'
                        flag = False
    
                else:
                    break
        else:
            break
            
    if flag:
        maillist_type[grp] = 'other'
        flag = False
print(c, len(maillist_type))

personID_personID_type_mapping_df = pd.DataFrame(columns=['A_PersonID_From','B_PersonID_To','Type','Time_since_1st_mail_A','Time_since_1st_mail_B','Max_Time_A','Max_Time_B','Interaction_timestamp','MessageID_A','MessageID_B','Mailing_list','Mailinglist_type'])

c = 0
d = 0

for mailing_list_name in archive.mailing_list_names():
    ml = archive.mailing_list(mailing_list_name)
    #mlist = mailing_list_name
    d += 1
    if d <= 1000:
        continue

    if d > 1200:
        print('break at ',d,mailing_list_name)
        break
    if ml:
        if ml._num_messages > 0:
            try:
                for t in ml.threads():
                    try:
                        nodes_list = []
                        #c += 1

                        if t.root is not None:

                            e = t.root.from_addr
                            e = e.replace("'", "__apostrophe__")
                            x = re.findall(ren, str(e))

                            if len(x) == 0:
                                x = re.findall(ren2, str(e))
                                if len(x) > 0:
                                    email = x[0]
                            else:
                                email = x[0][0]

                            email = email.replace("__apostrophe__", "'").lower()

                            if email in emailID_pid_dict and email not in automated_list:
                                personID = emailID_pid_dict[email]
                                #print(t.root.id,email,personID)
                                #nodes_list.append(personID)

                                yrs = []
                                years_months = set([])
                                
                                for eid in pid_emailID_dict[str(personID)]:

                                    if eid in emailID_first_email and eid in emailID_yearly_monthly_vol_dict:
                                        
                                        date_f = emailID_first_email[eid]
                                        yrs.append(date(datetime.strptime(date_f,"%Y-%m-%d").year,datetime.strptime(date_f,"%Y-%m-%d").month,datetime.strptime(date_f,"%Y-%m-%d").day))
                                        
                                        for yr in emailID_yearly_monthly_vol_dict[eid]:
                                            for month in emailID_yearly_monthly_vol_dict[eid][yr]:
                                                years_months.add(date(int(yr),int(month),1))

                                if len(yrs) > 0 and len(years_months) > 0:
                                    personID_first_email = min(yrs)
                                    
                                    #current_date = date(t.root.data.Date.year, t.root.data.Date.month, t.root.data.Date.day)
                                    current_date = date(t.root.date.year, t.root.date.month, t.root.date.day)
                                    age = (current_date - personID_first_email).days
                                    age = round_to(age/365,0.5)
                                    
                                    personID_last_email = max(years_months)
                                    max_age = (personID_last_email - min(years_months)).days #or can use personID_first_email
                                    max_age = round_to(max_age/365,0.5)

                                    personID_personID_type_mapping_df = dfs(c, t.root,[personID,age,max_age], personID_personID_type_mapping_df, mailing_list_name)

                    except Exception as e:
                        #print(e)
                        continue

                print(d, mailing_list_name, personID_personID_type_mapping_df.shape[0])

            except Exception as e:
                print('B- ZERO MESSAGE ',e,d, mailing_list_name,personID_personID_type_mapping_df.shape[0])
                continue

personID_personID_type_mapping_df.to_csv('data/personID_to_personID_graph_new_with_mlisttype.csv',mode='a',sep='\t',index=False,header=False)
#personID_personID_type_mapping_df.to_csv('./personID_to_personID_graph_new.csv',mode='a',sep='\t',index=False,header=False)
