# ColdstartCPI
ColdstartCPI：Improving compound-protein interaction prediction under cold-start scenarios by pretrain feature engineering
This repository contains the source code and the data.

Predicting compound-protein interactions (CPIs) is a critical step in drug discovery. Existing deep-learning-based methods show promising performance, but generally fail to generalize well to novel compounds and proteins due to the high sparsity of CPI data. To this end, we propose ColdstartCPI, a two-step framework that generates compound and protein representations with unsupervised pre-training, utilizes a Transformer-based structure to unify the pre-trained feature space with the CPI prediction space, and improves interactions for novel compounds and proteins. ColdstartCPI is evaluated under four realistic scenarios and achieves accurate and robust performance against state-of-the-art baselines. Furthermore, we validate the top predictions of ColdstartCPI through comparison with the experimental evidence and docking simulations. Our results indicate that ColdstartCPI provides a unified framework for integrating pre-trained models with CPI prediction tasks, which promises to be a powerful tool for drug discovery.

## ColdstartCPI framwork

<div align="center">
<p><img src="model.jpg" width="600" /></p>
</div>

## Setup and dependencies 

Dependencies:
- python = 3.8
- pytorch >= 1.12.0
- numpy = 1.18.1
- Pandas = 1.0.1
- RDKit = 2022.9.4
- tqdm
- Scikit-learn = 1.0.2
- prefetch_generator
- Mol2Vec
- ProtTrans

## Resources:
+ README.md: this file.
+ Datasets: The datasets used in paper.
	+ BindingDB: 
		+ Cross_domain: The datasets for blind start.
		+ feature: Contain the SMILES strings of compounds and amino acid sequences of proteins. 
			+ drug_list.txt: The SMILES strings of compounds
			+ protein_list.txt: Amino acid sequences of proteins
		+ In_domain: The datasets for warm start.
		+ full_pair.csv: The full dataset with positives and negatives for performance evaluation with scarce data.
	+ BindingDB_AIBind: 
		+ drug_coldstart: The datasets for compound cold start.
		+ feature: Contain the SMILES strings of compounds and amino acid sequences of proteins. 
			+ drug_list.txt: The SMILES strings of compounds
			+ protein_list.txt: Amino acid sequences of proteins
		+ pair_coldstart: The datasets for blind start.
		+ protein_coldstart: The datasets for protein cold start.
		+ warm_start: The datasets for warm start.
		+ drug_without_feature.txt: Contain the compounds of which the SMILES cannot be recongnized by Mol2Vec.
		+ full_pair.csv: The full dataset with positives and negatives for performance evaluation with scarce data.
		+ full_pair.txt: The full dataset with positives and negatives for performance evaluation with scarce data.
		+ protein_without_feature.txt: Contain the proteins of which the amino acid sequence cannot be recongnized by ProtTrans.
	+ BioSNAP
		+ Cross_domain: The datasets for blind start.
		+ feature: Contain the SMILES strings of compounds and amino acid sequences of proteins. 
			+ drug_list.txt: The SMILES strings of compounds
			+ protein_list.txt: Amino acid sequences of proteins
		+ In_domain: The datasets for warm start.
		+ full_pair.csv: The full dataset with positives and negatives for performance evaluation with scarce data.
	+ luo's_dataset
		+ data_folds
			+ warm_start_1_1: The datasets for warm start with Positives：Negatives=1:1.
			+ warm_start_1_10: The datasets for warm startwith Positives：Negatives=1:10.
			+ drug_coldstart: The datasets for compound cold start.
			+ protein_coldstart: The datasets for protein cold start.
		+ feature: Contain the SMILES strings of compounds and amino acid sequences of proteins. 
			+ drug_smiles.csv: The SMILES strings of compounds
			+ proseq.csv: Amino acid sequences of proteins
	+ yamanishi_08
		+ data_folds
			+ warm_start_1_1: The datasets for warm start with Positives：Negatives=1:1.
			+ warm_start_1_10: The datasets for warm startwith Positives：Negatives=1:10.
			+ drug_coldstart: The datasets for compound cold start.
			+ protein_coldstart: The datasets for protein cold start.
		+ feature: Contain the SMILES strings of compounds and amino acid sequences of proteins. 
			+ drug_smiles.csv: The SMILES strings of compounds
			+ proseq.csv: Amino acid sequences of proteins
+ Pretrian_models
	+ Mol2Vec
	
	Mol2Vec is customised version of Mol2Vec(https://github.com/samoturk/mol2vec). We recode the mol2vec/feature.py to generate feature matrices of compounds.
	
	You will obtain the feature vectors and matrices of the compounds by following the steps below:
	1. Place the SMILES sequence file of compounds, e.g. drug_list.txt, in the Mol2Vec folder;
	2. modify the code in lines 13-17 of the Mol2Vec.py file to obtain the dictionary "Drug" with key as Compound ID and Value as SMILES.
		
		python Mol2Vec.py
	
	+ ProtTrans
	You will obtain the feature vectors and matrices of the proteins by following the steps below:
	1. Place the amino acid sequence file of portiens, e.g. protein_list.txt, in the ProtTrans folder;
	2. modify the filepath in line 12 of the generator.py file and run generator.py.

		python generator.py
	
	
+ Train
	+ ColdstartCPI: The codes of training, testing, and model.
		+ ablation
			+ model.py: The codes of WOPretrain, WODecouple, WOTransformer, MolTrans_pretrain, and DrugBAN_pretrain.
		+ dataset.py
		+ model.py: The code of ColdstartCPI.
		+ train_BindingDB_AIBind.py: The code of evaluation in BindingDB_AIBind under warm start, compound cold start, protein cold start, and blind start.
		+ train_BindingDB_AIBind_missing.py: The code of evaluation in BindingDB_AIBind with scarce data.
		+ train_BindingDB_CrossDomain.py: The code of evaluation in BindingDB under blind start.
		+ train_BindingDB_InDomain.py: The code of evaluation in BindingDB under warm start.
		+ train_BindingDB_missing.py: The code of evaluation in BindingDB with scarce data.
		+ train_BioSNAP_CrossDomain.py: The code of evaluation in BioSNAP under blind start.
		+ train_BioSNAP_InDomain.py: The code of evaluation in BioSNAP under warm start.
		+ train_BioSNAP_missing.py: The code of evaluation in BioSNAP with scarce data.
		+ train_Luo_Yamanishi.py: The code of evaluation in the Luo and Yamanishi datasets under warm start, compound cold start and protein cold start.


# Run:
+ step 1: Generate the feature matrices of compounds and proteins
	+ 1.1 For compounds:
		+ Move the Mol2Vec-generated compound_Mol2Vec300.pkl and compound_Atom2Vec300.pkl to the feature folder in the corresponding dataset.
	+ 1.2 For proteins:
		+ Move the ProtTrans-generated aas_ProtTransBertBFD1024.pkl to the feature folder in the corresponding dataset.
+ setp 2: Training and testing
	+ python train_Luo_Yamanishi.py
	
	The results are saved in the Results folder.
	
	
