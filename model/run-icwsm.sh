# LR ADOPT

python model.py --feats "draft_authors_interaction.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/SJ.txt
python model.py --feats "SNA-BWC-ALL.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/BWC.txt
python model.py --feats "SNA-EC-ALL.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/EC.txt
python model.py --feats "SNA-AGE-ALL.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/AGE.txt
python model.py --feats "SNA-NUM-AUTH.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/NAU.txt
python model.py --feats "SNA-TOP-AUTH.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/TOP.txt
python model.py --feats "SNA-DRAFTS-ALL.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/DRAFTS.txt
python model.py --feats "SNA-AREAS-ALL.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/AREAS.txt
python model.py --feats "SNA-MLIST-ALL.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/MLIST.txt
python model.py --feats "SNA-AFF.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/AFF.txt
python model.py --feats "TDIV.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/TDIV.txt

python model.py --feats "emails-text-feats.pickle" --do_fs f1t --mode adopt --model lr > output-lr-adopt/TEXT.txt

python model.py --feats "emails-text-feats.pickle draft_authors_interaction.csv SNA-BWC-ALL.csv SNA-EC-ALL.csv SNA-AGE-ALL.csv SNA-NUM-AUTH.csv SNA-TOP-AUTH.csv SNA-DRAFTS-ALL.csv SNA-AREAS-ALL.csv SNA-MLIST-ALL.csv SNA-AFF.csv TDIV.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/ALL.txt

python model.py --feats "draft_authors_interaction.csv SNA-BWC-ALL.csv SNA-EC-ALL.csv SNA-AGE-ALL.csv SNA-NUM-AUTH.csv SNA-TOP-AUTH.csv SNA-DRAFTS-ALL.csv SNA-AREAS-ALL.csv SNA-MLIST-ALL.csv SNA-AFF.csv TDIV.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/ALL-NOTEXT.txt


python model.py --feats "emails-text-feats.pickle draft_authors_interaction.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/SJ-T.txt
python model.py --feats "emails-text-feats.pickle SNA-BWC-ALL.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/BWC-T.txt
python model.py --feats "emails-text-feats.pickle SNA-EC-ALL.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/EC-T.txt
python model.py --feats "emails-text-feats.pickle SNA-AGE-ALL.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/AGE-T.txt
python model.py --feats "emails-text-feats.pickle SNA-NUM-AUTH.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/NAU-T.txt
python model.py --feats "emails-text-feats.pickle SNA-TOP-AUTH.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/TOP-T.txt
python model.py --feats "emails-text-feats.pickle SNA-DRAFTS-ALL.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/DRAFTS-T.txt
python model.py --feats "emails-text-feats.pickle SNA-AREAS-ALL.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/AREAS-T.txt
python model.py --feats "emails-text-feats.pickle SNA-MLIST-ALL.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/MLIST-T.txt
python model.py --feats "emails-text-feats.pickle SNA-AFF.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/AFF-T.txt
python model.py --feats "emails-text-feats.pickle TDIV.csv" --do_fs f1t --mode adopt --model lr > output-lr-adopt/TDIV-T.txt


# Do the jargon filtered text exepriments
python model.py --feats "emails-text-feats.pickle" --do_fs f1t --do_wf 1 --mode adopt --model lr > output-lr-adopt/TEXT-S.txt
python model.py --feats "emails-text-feats.pickle draft_authors_interaction.csv SNA-BWC-ALL.csv SNA-EC-ALL.csv SNA-AGE-ALL.csv SNA-NUM-AUTH.csv SNA-TOP-AUTH.csv SNA-DRAFTS-ALL.csv SNA-AREAS-ALL.csv SNA-MLIST-ALL.csv SNA-AFF.csv TDIV.csv" --do_fs f1t --do_wf 1 --mode adopt --model lr > output-lr-adopt/ALL-S.txt



