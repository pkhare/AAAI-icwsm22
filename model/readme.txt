** Installation **
Go to the model folder and issue these commands:
conda create -n icwsmenv python=3.9.4     # create environment
pip install -r requirements.txt           # install all required libraries
cat feats.tar.gz.* | tar xvzf -           # uncompress the pregenerated feature files



** Running the model **
Feature files are, for conveninece, pregenerated in the ./model/feats folder (if you want to generate them yourself, see below)

Run the experiments
./run-icwsm.sh
./run-icwsm-stats.sh

Generate latex table with results and the table with statistics about the coefficients
python output-to-latex.py
python output-to-stats.py

Due to small implementation differences from the original paper the numbers may be slightly different, but the main trends remain.


** Generating the features **

To generate feature csv files follow these instructions for each feature grop:

Comm. patterns: draft_authors_interaction.csv
You can generate this file as follows:
TODO: how to generate this file (you may need to specify a different virtual env)

--------------------
Centrality:  SNA-BWC-ALL.csv
Email count: SNA-EC-ALL.csv
N. years active: SNA-AGE-ALL.csv
Number of authors: SNA-NUM-AUTH.csv
Proportion in top: SNA-TOP-AUTH.csv
Draft count: SNA-DRAFTS-ALL.csv
N. Areas:  SNA-AREAS-ALL.csv
N Mailing lists: SNA-MLIST-ALL.csv
Affiliations: SNA-AFF.csv

You can generate all these files by going to the model/feats folder and running
python convert.py drafts_authors_SNA_features.csv

You can generate drafts_authors_SNA_features.csv as follows:
TODO: how to generate this file (you may need to specify a different virtual env)

--------------------
Topic diversity: TDIV.csv
Text: emails-text-feats.pickle, emails-text-feats.colnames, emails-text-feats.docnames
- generating these is a very long, computationally intensive process that requires downloading all emails, we strongly recommend using the pregenerated files. However, if you need the uncurated code please let us know by email.

--------------------
Text (S)
- these are not actual extra features but are implemented by passing standard text features and the '--do_wf 1' flag to model.py

