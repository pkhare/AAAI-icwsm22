import pandas as pd
import os
import numpy as np
import csv
import tldextract
import matplotlib.pyplot as plt
import seaborn as sns
import re

import email.header
import email.utils
import string
import sys
from dataclasses import dataclass, field
from ietfdata.datatracker import *
from ietfdata.mailarchive import *
import datetime
from datetime import datetime
from datetime import date
import json


dt = DataTracker(use_cache = True)
#dt = DataTracker()

pid_datatracker_dict = json.load(open('./analysis_revision3/pid_datatracker_dict.json'))

pid_yearly_document = dict()
pid_document = dict()
pid_affiliation = dict()

for pid in pid_datatracker_dict:
    try:
        for doc in list(dt.documents_authored_by_person(dt.person(PersonURI(pid_datatracker_dict[pid])))):
            doc_ = dt.document(doc.document)
            #print(doc_.name,doc_.resource_uri.uri, doc_.time.year)
            year = doc_.time.year
            doc_uri = doc_.resource_uri.uri
            #print(doc_uri, len(pid_yearly_document),len(pid_document))
            aff = doc.affiliation
            aff = aff.strip().lower()
            if len(aff.strip()) == 0:
                aff = doc.email.uri
                aff = aff.replace(doc.email.root,'')
                aff = aff.lower()
                aff = aff.strip('/')
                aff = tldextract.extract(aff).domain
            else:
                if 'university' in aff or aff == 'china mobile' or aff == 'inside products':
                    aff = aff.strip('/')
                else:
                    aff = aff.lower().strip().split()[0].strip('/')

            if pid in pid_yearly_document:
                if year in pid_yearly_document[pid]:
                    if doc_uri not in pid_document[pid]:
                        pid_yearly_document[pid][year].append(doc_uri)
                        pid_document[pid].append(doc_uri)
                        pid_affiliation[pid][year].append(aff)
                else:
                    if doc_uri not in pid_document[pid]:
                        pid_yearly_document[pid][year] = [doc_uri]
                        pid_document[pid].append(doc_uri)
                        pid_affiliation[pid][year] = [aff]
            else:
                pid_yearly_document[pid] = {year: [doc_uri]}
                pid_document[pid] = [doc_uri]
                pid_affiliation[pid] = {year: [aff]}
            
    except Exception as e:
        print(e)
        #continue
    print(len(pid_yearly_document),len(pid_document))
    
json.dump(pid_yearly_document, open( "pid_yearly_document.json", 'w' ))
json.dump(pid_document, open( "pid_document.json", 'w' ))
json.dump(pid_affiliation, open( "pid_affiliation.json", 'w' ))