
with open("./output-lr-adopt/STATS.txt", "r") as infile:
    start_printing = False
    for line in infile:
        if "Statistics results" in line:
            start_printing = True
        if start_printing:
            print(line, end = "")
