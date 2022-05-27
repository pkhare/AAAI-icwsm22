import string
import sys
from dataclasses import dataclass, field
from ietfdata.datatracker import *
from ietfdata.mailarchive import *
import datetime
from datetime import datetime
from ietfdata.datatracker_ext import * #DataTrackerExt
import ietfdata.mailarchive as ma
import json

dt      = DataTracker()

group_histories = list(dt.group_histories())
group_histories = group_histories + list(dt.groups())

group_list_mapping = {}

for group_history in group_histories:
    archive_url = group_history.list_archive.strip().strip("\u200b")
    if archive_url == "":
        continue
    list_url_regexes = [r'^(https?://mailarchive.ietf.org/arch/(browse/|search/\?email_list=))(?P<name>[^/]*)/?$',
                        r'^(https?://(www.)?ietf.org/mail-archive/web/)(?P<name>[^/]*)(/|$)',
                        r'^((ftp://)?ftp.ietf.org/ietf-mail-archive/)(?P<name>[^/]*)(/|$)',
                        r'^(https?://(www.)?ietf.org/mailman/(private|listinfo)/)(?P<name>[^/]*)(/|$)',
                        r'^(https?://(www.)?irtf.org/mail-archive/web/)(?P<name>[^/]*)(/|$)',
                        r'^(https?://(www.)?rfc-editor.org/pipermail/)(?P<name>[^/]*)(/|$)',
                        r'^(https?://ops.ietf.org/lists/)(?P<name>[^/]*)(/|$)']
    for list_url_regex in list_url_regexes:
        list_url_match = re.search(list_url_regex, archive_url)
        if list_url_match is not None:
            list_name = list_url_match.group("name").lower()
            group_list_mapping[list_name] = group_list_mapping.get(list_name, []) + [group_history]
            break

def label_mailing_list(mailing_list_name, at_date=None):
    mailing_list_name = mailing_list_name.lower()
    if mailing_list_name in group_list_mapping:
        group_histories = group_list_mapping[mailing_list_name]
        labels = []
        for group_history in group_histories:
            if group_history.parent is not None:
                parent = dt.group(group_history.parent)
                label = f"{parent.acronym}-{dt.group_type_name(group_history.type).slug}-{dt.group_state(group_history.state).slug}"
            else:
                label = f"{dt.group_type_name(group_history.type).slug}-{dt.group_state(group_history.state).slug}"
            if len(labels) == 0 or label != labels[-1][0]:
                labels.append((label, group_history.time))
        if at_date is None:
            return labels[-1][0]
        else:
            for label in labels:
                if at_date < label[1]:
                    return label[0]
            return label[0]
    else:
        if re.match("(ietf)?\d+(attendees|all|onsite|companions?|hackathon|-mentees|-mentors|-1st-timers|-team)", mailing_list_name) is not None:
            return "meeting"
        elif re.match("nomcom\d+", mailing_list_name) is not None:
            return "nomcom"
    return None
    
archive = ma.MailArchive()

maillist_yearly_monthly_status = dict()

for mailing_list_name in archive.mailing_list_names():
    #print(mailing_list_name)
    #ml = archive.mailing_list(mailing_list_name)
    #if ml:
    #    if ml._num_messages > 0:
    for i in range(1987,2021):
        for j in range(1,13):
            date_ = str(i)+'-'+str(j)+'-'+str(1)

            type_response = label_mailing_list(mailing_list_name,datetime.strptime(date_, "%Y-%m-%d"))

            if isinstance(type_response, tuple):
                type_ = type_response[1]
            elif isinstance(type_response, str):
                type_ = type_response
            else:
                type_ = 'None'

            if mailing_list_name in maillist_yearly_monthly_status:
                if i in maillist_yearly_monthly_status[mailing_list_name]:
                    maillist_yearly_monthly_status[mailing_list_name][i][j] = type_
                else:
                    maillist_yearly_monthly_status[mailing_list_name][i] = {j:type_}
            else:
                maillist_yearly_monthly_status[mailing_list_name] = {i:{j:type_}}

        print(mailing_list_name,i,len(maillist_yearly_monthly_status))
        
json.dump(maillist_yearly_monthly_status, open( "data/maillist_yearly_monthly_status.json", 'w' ))
	