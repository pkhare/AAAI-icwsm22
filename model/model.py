from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import statsmodels.api as sm
import argparse
import sys
import os
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parser = argparse.ArgumentParser(description = 'Runs ICWSM prediction models.')
parser.add_argument("-verbose", "--verbose", help = "Whether to print out some additional info - 1 for yes or 0 for no. Default is 0.", default=0, type=int)
parser.add_argument("-do_fs", "--do_fs", help = "Whether to do feature selection - f1 or f1t for f1/f1 (2 class) optimiziation, auc for auc optimization, or none for no selection. If turned off all features are used. Default is f1t", default="f1t", type=str)
parser.add_argument("-feats", "--feats", help = "Whitespace separated list of feature file names, they should be in the feats subfolder. For example --feats 'orig-feats.csv example-feats.csv'. There is no default, this is required. Feature files are csv files with one column containing draft name (should be called 'doc_name') and all other columns containing feature values. The first row should contain feature names, which must end in _cat for categorical features and _num for numerical features. There are some example feature files in the feats subfolder.", type = str, required = True)
parser.add_argument("-model","--model", help = "Model - lr (logistic regression) or mfc (most frequent class) or dt (decision tree) or mlp (multilayer perceptron)", default = "lr", type = str)
parser.add_argument("-mode","--mode", help = "Mode of operation - 'adopt' for the 'does draft get assigned to working group' task and 'rfc' for the 'does draft become published as rfc' task. Default is 'rfc'", default = "rfc", type = str)
parser.add_argument("-parallel","--parallel", help = "Parallelize the feature selection loop, 1 for yes and 0 for no. Paralellizing will speed things up considerably when including text features, but will waste A LOT of ram (best used when running on a server). Default is 0.", default = 0, type = int)
parser.add_argument("-final_feats_filename","--final_feats_filename", help = "Name of the file (will be in csv format) to which the final feature set and label used in the statistical analysis will be written. If empty no file will be created. Default is empty.", default = "", type = str)
parser.add_argument("-weights_filename","--weights_filename", help = "Name of the file (will be in csv format) to which the final feature weights will be written. If empty no file will be created. Default is empty.", default = "", type = str)

parser.add_argument("-do_wf", "--do_wf", help = "Whether to remove domain specific stop words (WG names, last names, and jargon words. 1 means do it 0 means skip it.. Default is 0", default=0, type=int)
parser.add_argument("-do_stats", "--do_stats", help = "A 0 or 1 value indicating whether to print the logreg coefficients of the model with the final selected feature set. Default is 0 (if including thousands of text features this might take a really really long time).", default=0, type=int)

args=parser.parse_args(sys.argv[1:])
args.verbose = args.verbose == 1



# load up everything
label_df = pd.read_csv("./feats/gold-labels.csv")



print(label_df.shape)
if args.mode == "adopt":
    adopt_df = pd.read_csv("../data/ids_adoption.csv")
    adopt_ids = set(adopt_df["doc_name"].tolist())
    label_df = label_df[label_df.doc_name.isin(adopt_ids)]
elif args.mode == "rfc":
    pub_df = pd.read_csv("../data/ids_pub.csv")
    pub_ids = set(pub_df["doc_name"].tolist())
    label_df = label_df[label_df.doc_name.isin(pub_ids)]
else:
    raise Exception("Uknonwn mode :" + args.mode)


#print(label_df.shape)

df, feat_cols = label_df, []
#ignore_scaling_feats = set()

for f in args.feats.split():
  if not "pickle" in f:
    feats_df = pd.read_csv("./feats/" + f)
  else:
    feats_mat = pickle.load(open("./feats/" + f, "rb"))
    feats_mat = feats_mat.todense()
    feats_colnames = pickle.load(open("./feats/" + f + ".colnames", "rb"))
    feats_docnames = pickle.load(open("./feats/" + f + ".docnames", "rb"))
    
    #feats_df = pd.DataFrame.sparse.from_spmatrix(feats_mat)
    feats_df = pd.DataFrame(feats_mat)
    feats_df.columns = feats_colnames
    feats_df["doc_name"] = feats_docnames

    # load up all the stopwords
    stops_wg = pickle.load(open("wg-list.pickle", "rb"))
    stops_surname = set()
    for line in open("last_names_top20_each_year.txt", "r"):
        stops_surname.add(line.lower().strip())
    stops_jargon = set()
    for line in open("jargon2.txt", "r"):
        stops_jargon.add(line.lower().strip())
    
    stops_all = stops_wg.union(stops_surname).union(stops_jargon)
 
    #print(stops_all)
    # do the stopword filteringa
    if args.do_wf == 0:
        print("SKIPPING FILTER OF THE DOMAIN SPECIFIC WORDS (non alpha words will still get removed)")
        stops_all = set() 
    fcols = []
    for cname in feats_df.columns:
        if cname[6:-4] in stops_all:
            continue
        if cname[11:-4] in stops_all:
            continue
        if not cname[6:-4].replace(" ","").isalpha() and not cname == "doc_name":
            continue
        ok = True
        for w in cname[6:-4].split(" "):
            if w in stops_all:
                ok = False
                break
        if not ok:
            continue
        fcols.append(cname)

    print("NUMBER OF COLS BEFORE FILTERING " + str(feats_df.shape[1]))
    feats_df = feats_df[fcols]
    print("NUMBER OF COLS AFTER FILTERING " + str(feats_df.shape[1]))


  print("Shape is " + str(feats_df.shape))
  feat_cols_current = list(feats_df.columns)
  feat_cols_current.remove("doc_name")
  feat_cols += feat_cols_current
  
  print("Merging into main feature set ...")
  print("Shape before merge")
  print(df.shape, file = sys.stderr)
  print("Shape of new features")
  print(feats_df.shape, file = sys.stderr)
  df = df.merge(feats_df, on = ["doc_name"], how = "inner")
  print("After merge shape")
  print(df.shape, file = sys.stderr)
  #print(str("revisions_num" in feat_cols))
  #print(str("revisions_num" in df.columns))
  #print(df.columns[0:5])

# assert label_df.shape[0] == df.shape[0]

print("Final shape")
print(df.shape)

# prepare the input for the prediction model (handle categorical and numerical inputs differently)
def binlab(x):
    return 1 if (str(x) == "yes" or str(x) == "1" or str(x).lower() == "true") else 0

if args.mode == "rfc":
    y = [str(x) for x in df.is_rfc]
    y = [binlab(x) for x in y]
elif args.mode == "adopt":
    y = [str(x) for x in df.is_expired] 
    y = [binlab(x) for x in y]
    y = [1 - x for x in y]
else:
    raise Exception("Unknown mode: " + args.mode)


print("Processing columns of different types ...")
X = df[feat_cols].copy()
nonbinary_cols = []
col_norms = {}
for col in X.columns:
    if args.verbose:
        print("Loading column " + col, end = "")
    if col.endswith("_cat"):
        numvals = len(set(X[col]))
        if args.verbose:
            print(" Type: categorical Number of possible values: " + str(numvals), end = "")
        if numvals > 2:
            if args.verbose:
                print(" will turn to dummy variables.")
            nonbinary_cols.append(col)
        else:
            if args.verbose:
                print(" leaving as a binary variable.")
            res = X[col].apply(binlab)
            X.loc[:, col] = res
    elif col.endswith("_num"):
        if args.verbose:
            print(" Type: numerical, turning to float" + str(do_norm))
        X[col] = X[col].astype(float)
    else:
        raise Exception("Undefined column type (cat/num) for column : " + col)


#print(X)

X = pd.get_dummies(X, columns = nonbinary_cols)

print("Normalizing all variables ...")


X = X.fillna(0)
print(X.shape)
scaler = StandardScaler()
cnames = X.columns
X = pd.DataFrame(scaler.fit_transform(X))
X.columns = cnames

print(type(X))
print(X.shape)
print(len(y))






# building the model and fitting the data
print("Train test splitting ...")
random_seed = 42
X_traindev, X_test, y_traindev, y_test = train_test_split(X, y, test_size=0.20, random_state=random_seed, shuffle = True)
X_train, X_dev, y_train, y_dev = train_test_split(X_traindev, y_traindev, test_size=0.125, random_state=random_seed, shuffle = True)


def do_eval(train_set, train_labs, test_set, hparam):
    if args.model == "lr":
      clf = LogisticRegression(C = hparam, random_state = 9001, max_iter = 10000, penalty = "l1", solver = "liblinear", class_weight = "balanced")
      #clf = LogisticRegression(C = hparam, random_state = 9001, max_iter = 10000, penalty = "l1", solver = "liblinear")
    #clf = SVC(C = args.cost, random_state = 9001, probability = True, kernel = "rbf")
    elif args.model == "dt":
      clf = DecisionTreeClassifier(random_state = 9001)
    elif args.model == "mfc":
        clf = DummyClassifier(strategy = "most_frequent")
    elif args.model == "mlp":
        clf = MLPClassifier(hidden_layer_sizes = (hparam), early_stopping = True)
        #clf = MLPClassifier(hidden_layer_sizes = (50, 5), early_stopping = True, class_weight = "balanced")
    elif args.model == "rf":
        clf = RandomForestClassifier(n_estimators = 300, class_weight = "balanced", max_depth = hparam, random_state = 9001)

    clf.fit(train_set,train_labs)
    preds = clf.predict(test_set).tolist()
    probs = clf.predict_proba(test_set)[:,1].tolist()
    
    if args.model == "lr":
        weighted_feats = list(zip(list(train_set.columns), list(clf.coef_[0])))
        feat_list = [wf[0] for wf in weighted_feats if abs(wf[1]) >= 0.0000001] # feature name for features with nonzero weight
        feat_weights = [wf[1] for wf in weighted_feats if abs(wf[1]) >= 0.0000001] # feature name for features with nonzero weight
        if len(feat_list) == 0:
            feat_list = [x[0] for x in weighted_feats]
            feat_weights = [x[1] for x in weighted_feats]
    else:
        feat_list = train_set.columns
        feat_weights = [1] * train_set.shape[1]

    print("Finished eval!")
    return preds, probs, feat_list, feat_weights




#print(do_loo_xval(X, y))
#print(args.do_fs, file = sys.stderr)
if args.do_fs != "none":
    print("Feature selection ...")
    best_score, best_c = -1, -1
     

    def generate_score(h):
        print("Fitting model for hparam = " + str(h))
        preds, probs, nonzero_feats, nz_weights  = do_eval(X_train, y_train, X_dev, h)
        
        if args.do_fs == "f1":
            score = f1_score(y_true = y_dev, y_pred = [int(x) for x in preds], average = "binary")
        elif args.do_fs == "auc":
            score = roc_auc_score(y_true = y_dev, y_score = probs)
        elif args.do_fs == "f1t":
            score = f1_score(y_true = y_dev, y_pred = [int(x) for x in preds], average = "macro")
        else:
            raise Exception("Uknnown FS method: " + args.do_fs)
        
        print("HPARAM = %.7f, score = %.3f" % (h, score))
        return (h, score, nonzero_feats, nz_weights)

    if args.model == "lr":
        hparam_range = [2**i for i in range(-10,5)]
    elif args.model == "dt" or args.model == "mfc":
        hparam_range = [1]
    elif args.model == "mlp":
        hparam_range = [5, 10, 50, 100,500]
    elif args.model == "rf":
        hparam_range = [2,3,4,5]

    if args.parallel == 1:
        score_list = Parallel(n_jobs = 30, prefer = "threads")(delayed(generate_score)(i) for i in hparam_range)
    else:
        score_list = []
        for i in hparam_range:
            score_list.append(generate_score(i))
 
    for cval, score, nonzero_feats, nz_weights  in score_list:
        print("Considering c = " + str(cval))
        if score > best_score:
            print("*** UPDATING BEST SCORE *** ")
            best_score, best_c, feats_for_stats, weights_for_stats = score, cval, nonzero_feats, nz_weights
        
    print(feats_for_stats)

else: # no feat sel
    feats_for_stats = list(X.columns)
    best_c = 1

# fit one big model on all data


pd.set_option('display.max_rows', 3000)

# printouts
#print(feats_for_stats)
#feats_for_stats = [x for x in feats_for_stats if x.startswith("is_active")]
print("Running the final prediction model ...")
preds, probs, _, _ = do_eval(X_train[feats_for_stats], y_train, X_test[feats_for_stats], best_c)
print([f[6:-4] for f in feats_for_stats])
print(X_test.shape)
print(len(preds))

# print feat weights into a file
if args.weights_filename != "":
  flist = [f[6:-4] + "   " for f in feats_for_stats]
  wdf = pd.DataFrame(zip(flist, weights_for_stats), columns = ["name","weight"])
  print("****************")
  print(flist)
  print(weights_for_stats)
  print(wdf)
  wdf = wdf.sort_values(by = ["weight"], ascending = False)
  wdf.to_csv(args.weights_filename, index = False, sep = ",")

#print(preds[0:100])
#print(y_test[0:100])

f1 = f1_score(y_true = y_test, y_pred = [int(x) for x in preds], average = "binary")
auc = roc_auc_score(y_true = y_test, y_score = probs)
f1t = f1_score(y_true = y_test, y_pred = [int(x) for x in preds], average = "macro")
prec_t = precision_score(y_true = y_test, y_pred = [int(x) for x in preds], average = "macro")
recall_t = recall_score(y_true = y_test, y_pred = [int(x) for x in preds], average = "macro")

print("\n\n\n********** Test set results ***********************************")
print("F1 = %.3f, AUC = %.3f, F1T = %.3f, PT = %.3f, RT = %.3f " % (f1, auc, f1t, prec_t, recall_t))
print("****************************************************************************")

f1_pos = f1_score(y_true = y_test, y_pred = [int(x) for x in preds], pos_label = 1)
f1_neg = f1_score(y_true = y_test, y_pred = [int(x) for x in preds], pos_label = 0)
print(f1_pos)
print(f1_neg)
print(sum(y_test))
print(len(y_test))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred = [int(x) for x in preds] ))


if args.do_stats:
    #print(X[feats_for_stats])


    log_reg = sm.Logit(y, X[feats_for_stats]).fit(maxiter = 1000000)
    
    # building the model and fitting the data
    summary = log_reg.summary()
    #print(summary.tables[0].as_latex_tabular())
    print(summary)
    print("****************************************************************************")

    print("\n\n\n********** Statistics results ***********************************")
    results_as_html = summary.tables[1].as_html()
    table_df = pd.read_html(results_as_html, header=0 )[0]
    table_df = table_df.rename(columns = {"Unnamed: 0" : "feature"})
    table_df = table_df[["feature", "coef","P>|z|"]]
    table_df = table_df[table_df["P>|z|"] <= 0.05000]
    table_df["abs_coef"] = [abs(x) for x in table_df.coef]
    table_df.sort_values(by = ['abs_coef'], ascending = False, inplace = True)
    del table_df["abs_coef"]
    print(table_df)
    if args.final_feats_filename != "":
        out_df = X_test[feats_for_stats]
        out_df["label"] = y_test
        out_df["prediction"] = [int(x) for x in preds]
        out_df.to_csv(args.final_feats_filename, index = False)
    print("****************************************************************************")


