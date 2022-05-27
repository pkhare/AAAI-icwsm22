import numpy as np
import pandas as pd
import csv
import re
import json

from ietfdata.datatracker import *
from ietfdata.mailarchive import *

from ietfdata.datatracker_ext import * #DataTrackerExt

import matplotlib.pyplot as plt

import datetime
from datetime import datetime
from datetime import date
import pytz
import seaborn as sns

import ietfdata.mailarchive as ma

import tldextract

import pickle

plt.rc('font',**{'family':'serif','serif':['Helvetica']})
plt.rc('axes', axisbelow=True)
plt.rcParams['pdf.fonttype'] = 42

year_original_datatracker_emailID = dict()
year_nodatatracker_emailID = dict()
year_mapped_datatracker_emailID = dict()
year_automated_emailID = dict()
year_rolebased_emailID = dict()

ren = r'(?:\.?)([\w\-_+#~!$&\'\.]+(?<!\.)(@|[ ]?\(?[ ]?(at|AT)[ ]?\)?[ ]?)(?<!\.)[\w]+[\w\-\.]*\.[a-zA-Z-]{2,3})(?:[^\w])'
ren2 = r'([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'

archive = ma.MailArchive()

#email_all_set = set()
email_all_dict = dict()
empty_lists = set()
error_maillists = dict()

for mailing_list_name in archive.mailing_list_names():
    
    ml = archive.mailing_list(mailing_list_name)
    if ml:
        
        if ml._num_messages > 0:
            ml_df = ml.messages_dataframe()
            for message in ml.messages():
                try:
                    e = message.from_addr
                    e = e.replace("'","__apostrophe__")
                    x = re.findall(ren,str(e))
                    
                    if len(x) == 0:
                        x = re.findall(ren2,str(e))
                        if len(x) > 0:
                            email = x[0]
                    else:
                        email = x[0][0]
                    email = email.replace("__apostrophe__","'")#.lower()
                    
                    e = e.replace(email,'')
                    
                    email = email.lower()
                    
                    if '@' not in email:
                        email = email.replace(' at ','@').lower()
                    
                    if email in email_all_dict:
                        email_all_dict[email] = [email_all_dict[email][0],email_all_dict[email][1]+1]
                        continue
                            
                    e = e.strip('>')
                    e = e.strip('<')
                    e = e.strip()
                    e = e.strip('"')
                    e = e.strip()
                    e = e.lower()
                        
                    if email != '':
                        #email_all_set.add(email)
                        email_all_dict[email] = [e,1]
                    
                    if mailing_list_name in error_maillists:
                        error_maillists[mailing_list_name] = [error_maillists[mailing_list_name][0]+1,error_maillists[mailing_list_name][1]]
                    else:
                        error_maillists[mailing_list_name] = [1,0]
                except:
                    if mailing_list_name in error_maillists:
                            error_maillists[mailing_list_name] = [error_maillists[mailing_list_name][0],error_maillists[mailing_list_name][1]+1]
                    else:
                        error_maillists[mailing_list_name] = [0,1]
                    continue
            print(mailing_list_name,str(len(email_all_dict)))
            
        else:
            print(mailing_list_name, "ZERO MESSAGE", len(email_all_dict))
            empty_lists.add(mailing_list_name)
            continue

colnames = ['Email','URI','ID','Primary Email','Emails','Name','Ascii Name','Draft Name','Alias']
df5 = pd.read_csv('data/overall_email_uri_mapping.csv',delimiter='\t',names=colnames)

temp_set = set()
c = 0
with open('data/overall_unique_email_uri_mapping.csv','w') as wfile:

        writer = csv.writer(wfile, delimiter='\t', escapechar='\\', quotechar=None)
        writer.writerow(['Email','Name_In_Archive','Email_Count','URI','ID','Primary_Email','Emails','Datatracker_Name','Ascii_Name','Draft_Name','Alias'])

        for index, row in df5.iterrows():

            email = row['Email'].lower()
            uri = str(row['URI']).lower() if pd.notnull(row['URI']) else ''
            ID = int(row['ID']) if pd.notnull(row['ID']) else ''
            primary_email = str(row['Primary Email']).lower() if pd.notnull(row['Primary Email']) else ''
            emails = str(row['Emails']).lower() if pd.notnull(row['Emails']) else ''
            name = row['Name'] if pd.notnull(row['Name']) else ''
            draft_name = row['Draft Name'] if pd.notnull(row['Draft Name']) else ''
            alias = row['Alias'] if pd.notnull(row['Alias']) else ''
            ascii_name = row['Ascii Name'] if pd.notnull(row['Ascii Name']) else ''

            if email != '' and email not in temp_set:
                writer.writerow([email,email_all_dict[email][0].replace('\n','').replace('\t',' '),email_all_dict[email][1],uri,ID,primary_email,emails,name,ascii_name,draft_name,alias])
                temp_set.add(row['Email'].lower())
                c += 1
print(len(temp_set))
print(c)

dframe = pd.read_csv('data/overall_unique_email_uri_mapping.csv',delimiter='\t')

name_pid_dict = dict()
pid_name_dict = dict()
pid_emailID_dict = dict()
emailID_pid_dict = dict()
datatracker_pid_dict = dict()
pid_datatracker_dict = dict()

names_set = set()
emailID_set = set()

person_ID = 100000
c = 0

v_1 = 0
v_2 = 0
v_3 = 0
v_4 = 0
v_5 = 0
v_6 = 0
v_7 = 0

v_7_emailIDs = set([])

for index,row in dframe.iterrows():
    person_ID += 1
    #print(c,person_ID)
    c += 1
    names = set()
    eID_set = set()
    
    if row['Datatracker_Name'] == row['Datatracker_Name']:
        names.add(row['Datatracker_Name'].lower())
        
    if row['Name_In_Archive'] == row['Name_In_Archive']:
        names.add(row['Name_In_Archive'].lower())
        
    if row['Draft_Name'] == row['Draft_Name']:
        names.add(row['Draft_Name'].lower())
        
    if row['Ascii_Name'] == row['Ascii_Name']:
        names.add(row['Ascii_Name'].lower())
        
    if row['Alias'] == row['Alias']:
        alias_names = row['Alias'].split(';')
        alias_names.remove('')
        alias_names = [v.lower() for v in alias_names]
        names.update(alias_names)
        
    row_eid = row['Email'].lower()
    if '@' not in row_eid:
        row_eid = row_eid.replace(' at ','@').lower()
    #eID_set.add(row['Email'].lower())
    eID_set.add(row_eid)
    
    if row['Primary_Email'] == row['Primary_Email']:
        eID_set.add(row['Primary_Email'].lower())
        
    if row['Emails'] == row['Emails']:
        emails = row['Emails'].split(';')
        emails.remove('')
        emails = [v.replace(' at ','@').lower() if '@' not in v else v.lower() for v in emails]
        eID_set.update(emails)
    

    if '' in names:
        names.remove('')
        
    if 'none' in names:
        names.remove('none')
        
    if '' in eID_set:
        eID_set.remove('')
        
    if 'none' in eID_set:
        eID_set.remove('none')
    
    if row['URI'] == row['URI']:
        
        if row['URI'] in datatracker_pid_dict:
            
            v_1 += 1
            
            p_ID = datatracker_pid_dict[row['URI']]
            print('True',row['URI'],datatracker_pid_dict[row['URI']])
            
            if names:
                pid_name_dict[p_ID].update(names)
                names_set.update(names)
                for name in names:
                    name_pid_dict[name] = p_ID
                    #if name not in name_pid_dict:
                    
                        #names_set.add(name)
            if eID_set:
                pid_emailID_dict[p_ID].update(eID_set)
                
                for eID in eID_set:
                    emailID_pid_dict[eID] = p_ID
                    #if eID not in emailID_pid_dict:
                    
                        #emailID_set.add(eID)
                        
                emailID_set.update(eID_set)
                
                    
        else:
            if len(names_set.intersection(names))>0:
                names_set.update(names)
                
                v_2 += 1
                for name in names:
                    if name in name_pid_dict:
                        p_ID = name_pid_dict[name]
                        
                        datatracker_pid_dict[row['URI']] = p_ID
                        pid_datatracker_dict[p_ID] = row['URI']
                        
                        pid_name_dict[p_ID].update(names)
                        pid_emailID_dict[p_ID].update(eID_set)
                        
                        for name2 in names:
                            if name2 != name:
                                name_pid_dict[name2] = p_ID
                            
                        for eID in eID_set:
                            emailID_pid_dict[eID] = p_ID
                            
                        emailID_set.update(eID_set)
                            
                        break
            else:
                
                if len(emailID_set.intersection(eID_set))>0:
                    
                    v_3 += 1
                    emailID_set.update(eID_set)
                    names_set.update(names)
                    
                    for eID in eID_set:
                        if eID in emailID_pid_dict:
                            p_ID = emailID_pid_dict[eID]
                            
                            datatracker_pid_dict[row['URI']] = p_ID
                            pid_datatracker_dict[p_ID] = row['URI']

                            if names:
                                if p_ID in pid_name_dict:
                                    pid_name_dict[p_ID].update(names)
                                else:
                                    pid_name_dict[p_ID] = names
                                
                            pid_emailID_dict[p_ID].update(eID_set)

                            for name in names:
                                name_pid_dict[name] = p_ID

                            for eID2 in eID_set:
                                if eID2 != eID:
                                    emailID_pid_dict[eID2] = p_ID

                            break
                            
                else:
                    
                    v_4 += 1
                
                    datatracker_pid_dict[row['URI']] = person_ID
                    pid_datatracker_dict[person_ID] = row['URI']

                    #pid_emailID_dict[person_ID] = eID_set

                    if names:
                        names_set.update(names)
                        pid_name_dict[person_ID] = names
                        for name in names:
                            name_pid_dict[name] = person_ID

                    for eID in eID_set:
                        emailID_pid_dict[eID] = person_ID

                    if eID_set:
                        pid_emailID_dict[person_ID] = eID_set
                        
                    emailID_set.update(eID_set)

                    #person_ID += 1
                    
    else:
        if len(names_set.intersection(names))>0:
            
            v_5 += 1
            names_set.update(names)
            
            for name in names:
                if name in name_pid_dict:
                    p_ID = name_pid_dict[name]
                    pid_name_dict[p_ID].update(names)
                    pid_emailID_dict[p_ID].update(eID_set)
                    
                    for name2 in names:
                        name_pid_dict[name2] = p_ID
                    for eID in eID_set:
                        emailID_pid_dict[eID] = p_ID
                    
                    emailID_set.update(eID_set)
                    
                    break
        else:
            
            if len(emailID_set.intersection(eID_set))>0:
                
                v_6 += 1
                
                emailID_set.update(eID_set)
                names_set.update(names)

                for eID in eID_set:
                    if eID in emailID_pid_dict:
                        p_ID = emailID_pid_dict[eID]

                        if names:
                            if p_ID in pid_name_dict:
                                pid_name_dict[p_ID].update(names)
                            else:
                                pid_name_dict[p_ID] = names

                        pid_emailID_dict[p_ID].update(eID_set)

                        for name in names:
                            name_pid_dict[name] = p_ID

                        for eID2 in eID_set:
                            emailID_pid_dict[eID2] = p_ID

                        break
            else:
            
                if names:
                    
                    names_set.update(names)
                    emailID_set.update(eID_set)
                    pid_name_dict[person_ID] = names
                    for name in names:
                        name_pid_dict[name] = person_ID

                if eID_set:
                    pid_emailID_dict[person_ID] = eID_set

                for eID in eID_set:
                    emailID_pid_dict[eID] = person_ID
                    
                    v_7_emailIDs.add(eID)
                v_7 += 1

                #person_ID += 1

#print(v_1,v_2,v_3,v_4,v_5,v_6,v_7,c,len(pid_emailID_dict),len(v_7_emailIDs))
print('v_1',v_1)
print('v_2',v_2)
print('v_3',v_3)
print('v_4',v_4)
print('v_5',v_5)
print('v_6',v_6)
print('v_7',v_7)
print('c',c)
print(len(pid_emailID_dict))
print(len(v_7_emailIDs))

###

dframe.loc[dframe.Email.isin(list(v_7_emailIDs))].sort_values(by='Email_Count',ascending=False).head()#50)

pid_emailID_list_dict = dict()
for k in pid_emailID_dict:
    pid_emailID_list_dict[k] = list(pid_emailID_dict[k])
    
pid_name_list_dict = dict()
for k in pid_name_dict:
    pid_name_list_dict[k] = list(pid_name_dict[k])

json.dump(pid_emailID_list_dict, open( "data/pid_emailID_dict.json", 'w' ) )
json.dump(pid_name_list_dict, open( "data/pid_name_dict.json", 'w' ) )

json.dump(name_pid_dict, open( "data/name_pid_dict.json", 'w' ) )
json.dump(emailID_pid_dict, open( "data/emailID_pid_dict.json", 'w' ) )
json.dump(datatracker_pid_dict, open( "data/datatracker_pid_dict.json", 'w' ) )
json.dump(pid_datatracker_dict, open( "data/pid_datatracker_dict.json", 'w' ) )

emailID_yearly_monthly_vol_dict = dict(dict(dict()))
emailID_first_email = dict()

for mailing_list_name in archive.mailing_list_names():
    
    ml = archive.mailing_list(mailing_list_name)
    if ml:
        if ml._num_messages > 0:
            ml_df = ml.messages_dataframe()
            for msg in ml.messages():
                try:
                    e = msg.from_addr
                    e = e.replace("'", "__apostrophe__")
                    x = re.findall(ren, str(e))

                    if len(x) == 0:
                        x = re.findall(ren2, str(e))
                        if len(x) > 0:
                            email = x[0]
                    else:
                        email = x[0][0]

                    email = email.replace("__apostrophe__", "'").lower()
                    email = email.lower()
                    
                    if '@' not in email:
                        email = email.replace(' at ','@').lower()

                    if email:

                        if email not in emailID_first_email:
                            emailID_first_email[email] = date(msg.date.year, msg.date.month, msg.date.day)
                        else:
                            emailID_first_email[email] = min(emailID_first_email[email],date(msg.date.year, msg.date.month, msg.date.day))

                        if email not in emailID_yearly_monthly_vol_dict:
                            emailID_yearly_monthly_vol_dict[email] = {msg.date.year : {msg.date.month : 1}}
                        else:
                            if msg.date.year in emailID_yearly_monthly_vol_dict[email]:
                                emailID_yearly_monthly_vol_dict[email][msg.date.year][msg.date.month] = emailID_yearly_monthly_vol_dict[email][msg.date.year].get(msg.date.month,0)+1
                            else:
                                emailID_yearly_monthly_vol_dict[email][msg.date.year] = {msg.date.month : 1}
                        
                except Exception as e:
                    print(e)
                    continue
            print(mailing_list_name, len(emailID_yearly_monthly_vol_dict),len(emailID_first_email))
        else:
            print(mailing_list_name,"ZERO MESSAGE",len(emailID_yearly_monthly_vol_dict),len(emailID_first_email))
            continue

def json_serial_f(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

json.dump(emailID_yearly_monthly_vol_dict, open( "data/emailID_yearly_monthly_vol_dict.json", 'w' ) )
json.dump(emailID_first_email, open( "data/emailID_first_email.json", 'w' ),default=json_serial_f)

emailID_yearly_monthly_vol_dict = json.load(open('data/emailID_yearly_monthly_vol_dict.json'))
emailID_first_email = json.load(open('data/emailID_first_email.json'))

#instead of creating the above object, loading from earlier created json
f = open('data/maillist_yearly_monthly_status.json')
maillist_yearly_monthly_status = json.load(f)

wg_rg_emailID_yearly_monthly_vol_dict = dict(dict(dict()))

for mailing_list_name in archive.mailing_list_names():
    flag = False
    for year in maillist_yearly_monthly_status[mailing_list_name]:
        if flag:
            break
        else:
            for month in maillist_yearly_monthly_status[mailing_list_name][year]:
                if flag:
                    break
                else:
                    if maillist_yearly_monthly_status[mailing_list_name][year][month] == 'wg':
                        flag = True
                    if maillist_yearly_monthly_status[mailing_list_name][year][month] == 'rg':
                        flag = True
                        
    if flag:
    
        ml = archive.mailing_list(mailing_list_name)
        if ml:
            if ml._num_messages > 0:
                ml_df = ml.messages_dataframe()
                for msg in ml.messages():
                    try:
                        e = msg.from_addr
                        e = e.replace("'", "__apostrophe__")
                        x = re.findall(ren, str(e))

                        if len(x) == 0:
                            x = re.findall(ren2, str(e))
                            if len(x) > 0:
                                email = x[0]
                        else:
                            email = x[0][0]

                        email = email.replace("__apostrophe__", "'").lower()
                        
                        email = email.lower()
                    
                        if '@' not in email:
                            email = email.replace(' at ','@').lower()

                        if email:

                            #if email not in wg_rg_emailID_first_email:
                            #    wg_rg_emailID_first_email[email] = date(row['Date'].year, row['Date'].month, row['Date'].day)
                            #else:
                            #    wg_rg_emailID_first_email[email] = min(wg_rg_emailID_first_email[email],date(row['Date'].year, row['Date'].month, row['Date'].day))

                            if email not in wg_rg_emailID_yearly_monthly_vol_dict:
                                wg_rg_emailID_yearly_monthly_vol_dict[email] = {msg.date.year : {msg.date.month : 1}}
                            else:
                                if msg.date.year in wg_rg_emailID_yearly_monthly_vol_dict[email]:
                                    wg_rg_emailID_yearly_monthly_vol_dict[email][msg.date.year][msg.date.month] = wg_rg_emailID_yearly_monthly_vol_dict[email][msg.date.year].get(msg.date.month,0)+1
                                else:
                                    wg_rg_emailID_yearly_monthly_vol_dict[email][msg.date.year] = {msg.date.month : 1}

                    except Exception as e:
                        #print(e)
                        continue
                print(mailing_list_name, len(wg_rg_emailID_yearly_monthly_vol_dict))#,len(emailID_first_email))
            else:
                print(mailing_list_name,"ZERO MESSAGE",len(wg_rg_emailID_yearly_monthly_vol_dict))#,len(emailID_first_email))
                continue

def wg_roles(wg:str):
    wg = wg.strip()
    chair_id = None
    ads_id = None
    if wg:
        chair_id = wg+'-chairs@ietf.org'
        ads_id = wg+'-ads@ietf.org'
        
    return(chair_id, ads_id)

def rg_roles(wg:str):
    wg = wg.strip()
    chair_id = None
    if wg:
        chair_id = wg+'-chairs@ietf.org'
        
    return(chair_id)

def draft_roles(dft:str):
    dft = dft.strip()
    dft_id = None
    dft_authors = None
    dft_chairs = None
    dft_ad = None
    dft_notify = None
    dft_shepherd = None
    dft_all = None
    if dft:
        dft_id = dft+'@ietf.org'
        dft_authors = dft+'.authors@ietf.org'
        dft_chairs = dft+'.chairs@ietf.org'
        dft_ad = dft+'.ad@ietf.org'
        dft_notify = dft+'.notify@ietf.org'
        dft_shepherd = dft+'.shepherd@ietf.org'
        dft_all = dft+'.all@ietf.org'
        
    return(dft_id, dft_authors, dft_chairs, dft_ad, dft_notify, dft_all)

emailid_dtracker = set(list(dframe.loc[dframe.URI.notna()].Email))
len(emailid_dtracker)

for i,r in dframe.loc[dframe.URI.notna()].iterrows():
    if r['Emails'] == r['Emails']:
        emails_ = r['Emails'].split(';')
        emails_.remove('')
        for e in emails_:
            emailid_dtracker.add(e.replace(' at ','@').lower() if '@' not in e else e.lower())

for i,r in dframe.loc[dframe.URI.notna()].iterrows():
    if r['Primary_Email'] == r['Primary_Email']:
        emailid_dtracker.add(r['Primary_Email'].lower())

print(len(emailid_dtracker))

automated_list = set()

for i,r in dframe.loc[dframe.URI.isna()].iterrows():
    email_id = r['Email']
    
    if email_id not in emailid_dtracker:
    
        if any(re.findall(r'^noreply|noreply@|^reply@|-archive@|notification|trac\+|^trac\+|trac@|@trac.tools|-secretary@|-secretariat|internet-drafts|-request@|tools.ietf|-bounces@|-owner@', email_id, re.IGNORECASE)):
            automated_list.add(email_id)
print(len(automated_list))

with open('data/automated_email_IDs_newData.txt', 'w') as f:
    for item in automated_list:
        f.write("%s\n" % item)

#print(len(emailID_set))
c = 0
role_based_emailIDs = set()
for i,r in dframe.iterrows():
    eid = r['Email'].lower()
    
    if '@' not in eid:
        eid = eid.replace(' at ','@').lower()
    domain = tldextract.extract(eid).domain
    try:
        if domain == 'ietf' or domain == 'irtf' or domain == 'iana' or domain == 'rfc-editor':
            if eid not in automated_list and '-admin' not in eid and eid not in emailid_dtracker and '.' not in eid[0:eid.index('@')]:
                if '-' in eid[0:eid.index('@')]:
                    c += 1
                    role_based_emailIDs.add(eid)
                elif eid == 'chair@ietf.org':
                    c += 1
                    role_based_emailIDs.add(eid)
    except:
        print(eid)
print(c)
print(len(role_based_emailIDs))

list_WG = []
list_RG = []

#f = open('../analysis_revision3/maillist_yearly_monthly_status.json')
#maillist_yearly_monthly_status = json.load(f)
c = 0
for grp in maillist_yearly_monthly_status:
    flag = True
    c += 1
    for year in maillist_yearly_monthly_status[grp]:
        if flag:
            for month in maillist_yearly_monthly_status[grp][year]:
                if flag:
                    if maillist_yearly_monthly_status[grp][year][month] == 'wg':
                        list_WG.append(grp)
                        flag = False
                    if maillist_yearly_monthly_status[grp][year][month] == 'rg':
                        list_RG.append(grp)
                        flag = False
                else:
                    break
        else:
            break
print(c)

wg_rg_role_based_IDs = []

for grp in list_WG:
    for role_eid in wg_roles(grp):
        wg_rg_role_based_IDs.append(role_eid.lower())
        
for grp in list_RG:
    role_eid = rg_roles(grp)
    wg_rg_role_based_IDs.append(role_eid.lower())
    
print(len(wg_rg_role_based_IDs))

for eid in wg_rg_role_based_IDs:
    role_based_emailIDs.add(eid)

print(len(role_based_emailIDs))

with open('data/role_based_emailIDs_newData.txt', 'w') as f:
    for item in role_based_emailIDs:
        f.write("%s\n" % item)

### spam email proc here

spam_emails  = pickle.load(open("data/spam-emails.pickle", "rb"))

spam_emails_set = set([])
for e in spam_emails:
    spam_emails_set.add(e.replace(' at ','@').lower() if '@' not in e else e.lower())
len(spam_emails_set)

list(pid_emailID_dict[100004])[0]

class_A_B_dict = dict()
class_C_dict = dict()
class_D_dict = dict()
class_E_dict = dict()

for pid in pid_emailID_dict:
    if len(pid_emailID_dict[pid]) > 1:
        if len(emailid_dtracker.intersection(set(pid_emailID_dict[pid])))>0:
            class_A_B_dict[pid] = list(pid_emailID_dict[pid])
        else:
            class_C_dict[pid] = list(pid_emailID_dict[pid])
    else:
        if len(emailid_dtracker.intersection(set(pid_emailID_dict[pid])))>0:
            class_D_dict[pid] = list(pid_emailID_dict[pid])
        else:
            class_E_dict[pid] = list(pid_emailID_dict[pid])
            
class_A_B_emailID_set = set([eid for pid in class_A_B_dict for eid in class_A_B_dict[pid]])

year_original_datatracker_emailID = dict()
year_nodatatracker_emailID = dict()
year_mapped_datatracker_emailID = dict()
year_automated_emailID = dict()
year_rolebased_emailID = dict()

original_datatracker_emailID_sets = set([])
nodatatracker_emailID_sets = set([])
mapped_datatracker_emailID_sets = set([])
automated_emailID_sets = set([])
rolebased_emailID_sets = set([])
total_emailID_sets = set([])

year_emailID_count = dict()
year_email_count = dict()

for mailing_list_name in archive.mailing_list_names():
    ml = archive.mailing_list(mailing_list_name)
    if ml:
        if ml._num_messages > 0:
            ml_df = ml.messages_dataframe()
            for msg in ml.messages():
                
                try:
                    
                    email = ''
                    e = msg.from_addr
                    e = e.replace("'", "__apostrophe__")
                    x = re.findall(ren, str(e))

                    if len(x) == 0:
                        x = re.findall(ren2, str(e))
                        if len(x) > 0:
                            email = x[0]
                    else:
                        email = x[0][0]

                    email = email.replace("__apostrophe__", "'").lower()
                    
                    email = email.lower()
                    
                    # date counters
                    if '@' not in email:
                        email = email.replace(' at ','@').lower()

                    if msg.date.year in year_emailID_count:
                        year_emailID_count[msg.date.year].add(email)
                    else:
                        year_emailID_count[msg.date.year] = set([email])

                    if msg.date.year in year_email_count:
                        year_email_count[msg.date.year] = year_email_count[msg.date.year] + 1
                    else:
                        year_email_count[msg.date.year] = 1
                    
                    total_emailID_sets.add(email)
                    
                    #check if automated

                    if any(re.findall(r'^noreply|noreply@|^reply@|-archive@|notification|trac\+|^trac\+|@trac.tools|-secretary@|-secretariat|internet-drafts|-request@|tools.ietf|-bounces@|-owner@', email, re.IGNORECASE)):
                        year_automated_emailID[msg.date.year] = year_automated_emailID.get(msg.date.year, 0)+1
                        automated_emailID_sets.add(email)
                        continue

                    #check role based
                    if email in role_based_emailIDs:
                        year_rolebased_emailID[msg.date.year] = year_rolebased_emailID.get(msg.date.year, 0)+1
                        rolebased_emailID_sets.add(email)
                        continue

                    #check if Cat A-B

                    if email in emailid_dtracker:
                        year_original_datatracker_emailID[msg.date.year] = year_original_datatracker_emailID.get(msg.date.year, 0)+1
                        original_datatracker_emailID_sets.add(email)
                    else:
                        if email in class_A_B_emailID_set:
                            year_mapped_datatracker_emailID[msg.date.year] = year_mapped_datatracker_emailID.get(msg.date.year, 0)+1
                            mapped_datatracker_emailID_sets.add(email)
                        else:
                            year_nodatatracker_emailID[msg.date.year] = year_nodatatracker_emailID.get(msg.date.year, 0)+1
                            nodatatracker_emailID_sets.add(email)

                except Exception as e:
                    continue

            print(mailing_list_name, len(year_original_datatracker_emailID), len(year_mapped_datatracker_emailID), len(year_nodatatracker_emailID), len(year_automated_emailID))
            print(mailing_list_name, len(total_emailID_sets), len(original_datatracker_emailID_sets), len(mapped_datatracker_emailID_sets), len(nodatatracker_emailID_sets), len(automated_emailID_sets))

year_personID_count = dict()

for yr in year_emailID_count:
    
    if yr not in year_personID_count:
        year_personID_count[yr] = set([])
        
    for email in year_emailID_count[yr]:
        if email in emailID_pid_dict:
            pid = emailID_pid_dict[email]
            year_personID_count[yr].add(pid)
            
print(len(year_personID_count))
print(len(year_emailID_count))

personIDs_count = []
Years = []
for i in range(1995,2021):
    personIDs_count.append(len(year_personID_count.get(i,[])))
    Years.append(i)
print('Done')
personID_count_df2 = pd.DataFrame({
    "PersonID_count":personIDs_count,
    "Years":Years
    }
)
personID_count_df2.head(2)

Original_DT = []
Mapped_DT = []
No_DT = []
Automated = []
Years = []
Role_based = []

for i in range(1995,2021):
    Original_DT.append(year_original_datatracker_emailID.get(i,0))
    Mapped_DT.append(year_mapped_datatracker_emailID.get(i,0))
    No_DT.append(year_nodatatracker_emailID.get(i,0))
    Automated.append(year_automated_emailID.get(i,0))
    Role_based.append(year_rolebased_emailID.get(i,0))
    Years.append(i)
    
plotdata = pd.DataFrame({
    "Original_DT":Original_DT,
    "Mapped_DT":Mapped_DT,
    "No_DT":No_DT,
    "Automated":Automated,
    "Role_based":Role_based
    }, index=Years
)
plotdata.head()

x = []
x1 = []
y = []
for yr in range(1995,2021):
    x.append(year_email_count[yr])
    x1.append(len(year_emailID_count[yr]))
    y.append(yr)

df_emails_yearly = pd.DataFrame({'Years':y,'Number_of_emails':x})

dt_mapped = []
for i in range(1995,2021):
    dt_mapped.append(year_original_datatracker_emailID.get(i,0)+year_mapped_datatracker_emailID.get(i,0))

plotdata_supplement = pd.DataFrame({
    "Datatracker Person-ID":dt_mapped,
    "Automated":Automated,
    "Role-based":Role_based,
    "New Person-ID":No_DT,
    }, index=Years
)
plotdata_supplement.head()

years = list(range(1995,2021))

with open("data/frequency_emails_yearly_categories2.csv", "w") as freqYearlyFile:
    print("year,dt_mapped_count,automated_count,role_based_count,no_dt_count", file=freqYearlyFile)
    for i in range(len(years)):
        print(f"{years[i]},{dt_mapped[i]},{Automated[i]},{Role_based[i]},{No_DT[i]}", file=freqYearlyFile)

with open("data/pid_count_emailing_yearly.csv", "w") as pidCountYearlyFile:
    print("year,pid_count,message_count", file=pidCountYearlyFile)
    for i in range(1995,2021):
        pid_count = personID_count_df2.loc[personID_count_df2['Years'] == i].PersonID_count.tolist()[0]
        msg_count = df_emails_yearly.loc[df_emails_yearly['Years'] == i].Number_of_emails.tolist()[0]
        print(f"{i},{pid_count},{msg_count}", file=pidCountYearlyFile)