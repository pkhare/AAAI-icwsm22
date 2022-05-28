# TODO change feature names to account for CONEXT features
import sys

f1 = 0
auc = 0
feature_stats = []


filenames_per_run = [
        ("SJ.txt","Comm. patterns"),
        ("BWC.txt","Centrality"),
        ("EC.txt", "Email count"),
        ("AGE.txt","Age"),
        ("NAU.txt","Number of authors"),
        ("TOP.txt", "Top percentiles"),
        ("DRAFTS.txt", "Draft count"),
        ("AREAS.txt", "N. Areas"),
        ("MLIST.txt", "N. Mailing lists"),
        ("AFF.txt","Affiliations"),
        ("TDIV.txt","Topic entropy ($\eta$)"),
        ("TEXT.txt","Text"),
        ("TEXT-S.txt","Text (S)"),
        ("ALL-NOTEXT.txt","Everything except text"),
        ("ALL.txt","Everything"),
        ("ALL-S.txt","Everything (S)")
        ]

rownames = [x[1] for x in filenames_per_run]
result_lists = {}

def modify(n, x):
    if x == "notext" or n == "TEXT.txt" or n == "ALL.txt" or n == "ALL-NOTEXT.txt" or n == "ALL-S.txt" or n == "TEXT-S.txt":
        return n
    if x == "text":
        return n.replace(".txt", "-T.txt")

for model in ["notext", "text"]:    
    result_lists[model] = {}
    for dataset in ["adopt"]:
      result_lists[model][dataset] = []
      for fname, lab in filenames_per_run:
          try:
              f1t = None
              file_to_process = "./output-lr-" + dataset + "/" + modify(fname, model)
              with open(file_to_process) as modelOutputFile:
                for line in modelOutputFile:
                    if "Test set" in line:
                        line = next(modelOutputFile)
                        line = line[:-1].split(",")
                        f1 = float(line[0].split()[-1])
                        auc = float(line[1].split()[-1])
                        f1t = float(line[2].split()[-1])
                        pt = float(line[3].split()[-1])
                        rt = float(line[4].split()[-1])
                        break
              result_lists[model][dataset].append((f1t,pt,rt,auc))
          except:
              result_lists[model][dataset].append(None)

#print("\\begin{tabular}{lrr}")
#print("\\toprule")
#print("Feature Name & Coef. & P>|Z| \\\\")
#print("\\midrule")
def form(x):
    if x is None:
        return " & .???  & .???  & .??? & .???"
    p = "%.3f" % (x[1])
    p = p[1:]
    r = "%.3f" % (x[2])
    r = r[1:]
    f = "%.3f" % (2 * x[1]* x[2] / (x[1] + x[2])) # fixes macro F1 being slightly higher than the average of macro precision and macro recall
    f = f[1:]
    auc = "%.3f" % (x[3])
    auc = auc[1:]
    return " &  " + auc +" & " + f + " & " + p + " & " + r


for i in range(len(rownames)):
    print(rownames[i], end = "")
    for dataset in ["adopt"]:
        for model in ["notext", "text"]:
            if (rownames[i] == "Text" or rownames[i] == "Everything" or rownames[i]=="Everything (S)" or rownames[i] == "Text (S)") and model == "notext":
                print(" & - & - & - & - ", end = "")
            elif (rownames[i] == "Everything except text") and model != "notext":
                print(" & - & - & - & - ", end = "")
            else:
                print(form(result_lists[model][dataset][i]), end = "")
    print(" \\\\ ")
#print("\\bottomrule")
#print("\\end{tabular}")

