import os
import pandas as pd
if __name__ == "__main__":
    candidates_df = pd.read_csv("./../../../candidate_compounds_100.csv")
    affinity_list = []
    IDs = []
    with open("5_IDs.txt","r") as file:
        lines = file.readlines()
        for line in lines:
            ID = line[2:].strip().split(".")[0]
            IDs.append(ID)
    for ID in IDs:
        with open("{}.pdb".format(ID),"r") as file:
            lines = file.readlines()
            line = lines[1]
            affinity = float(line.split("    ")[1])
            affinity_list.append([ID,affinity])
    affinity_df = pd.DataFrame(affinity_list,columns = ["ID","Affinity(kcal/mol)"])
    candidates_df = pd.merge(candidates_df, affinity_df, on='ID', how='left')
    candidates_df.to_csv("./candidate_compounds_100.csv")
    print("Done")