drug_list = set()
protein_list = set()
drug_dict = {}
protein_dict = {}
with open("./demo_data.txt","r") as file:
    for pair in file.readlines():
        cid,pid,label = pair.strip().split()
        drug_list.add(cid)
        protein_list.add(pid)
with open("./../../Datasets/BindingDB_AIBind/feature/drug_list.txt","r") as file:
    for pair in file.readlines():
        cid,smiles = pair.strip().split()
        drug_dict[cid] = smiles
with open("./../../Datasets/BindingDB_AIBind/feature/protein_list.txt","r") as file:
    for pair in file.readlines():
        pid,aas = pair.strip().split()
        protein_dict[pid] = aas
with open("drug_list.txt","w") as file:
    for cid in drug_list:
        file.writelines("{} {}\n".format(cid,drug_dict[cid]))
with open("protein_list.txt","w") as file:
    for pid in protein_list:
        file.writelines("{} {}\n".format(pid,protein_dict[pid]))