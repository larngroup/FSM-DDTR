# -*- coding: utf-8 -*-
"""


"""

import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
import json
from utils import *
from sascorer_calculator import SAscore
from rdkit.Chem import MolFromSmiles, AllChem, QED, Descriptors,rdMolDescriptors, FindMolChiralCenters
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit import DataStructs
import os
import csv



def smiles_str_to_tokens(molecule):
	""" This function returns the list of tokens in a molecule given in SMILES notation.
	-------
	Args:
	- molecule (str) -> SMILES molecule

	-------
	Returns:
	- molecule_tokens (list of str) -> list of SMILES tokens
	"""

	molecule_tokens=[]
	indx_mol =0
	while indx_mol < len(molecule):
		if indx_mol<=len(molecule)-2 and molecule[indx_mol]=='B' and molecule[indx_mol+1]=='r':
			molecule_tokens.append(molecule[indx_mol]+molecule[indx_mol+1])
			indx_mol+=1
		elif indx_mol<=len(molecule)-2 and molecule[indx_mol]=='C' and molecule[indx_mol+1]=='l':
			molecule_tokens.append(molecule[indx_mol]+molecule[indx_mol+1])
			indx_mol+=1
		else:
			molecule_tokens.append(molecule[indx_mol])
		indx_mol+=1
	return molecule_tokens



def smiles_tokens_to_idx_and_padding(smiles_token, max_len, dict_mol):
	""" This function transforms each SMILES token in the respective integer given by the dictionary token and applies padding.
	-------
	Args:
	- smiles_token (list of lists of int) -> list of smiles token
	- max_len (int) -> maximum number of tokens in a SMILES
	- dict_mol (dict) -> dictionary of SMILES 
  
	-------
	Returns:
	- smiles_emb (list of lists of int) -> list of smiles with the tokens represented as integers
	- smiles_tokens (list of lists of tokens) -> list of SMILES token of molecules with length less than max_len
	- smiles_str (list of lists of str) -> list of SMILES strings with length less than max_len
	- smiles_ind (list of int) -> list of SMILES indexes with length less than max_len
	"""

	smiles_emb = []
	smiles_tokens = []
	smiles_str = []
	smiles_ind = []

	for ind_smile in range(len(smiles_token)):
		if len(smiles_token[ind_smile]) <= max_len:

			smiles_ind.append(ind_smile)
			smiles_tokens.append(smiles_token[ind_smile])
			smiles_str.append(''.join(smiles_token[ind_smile]))
			
			temp = [dict_mol['[CLS]']]
			distance_vector = max_len-len(smiles_token[ind_smile])
			
			for token_smile in smiles_token[ind_smile]:
				temp.extend([dict_mol[token_smile]])
			temp.extend([dict_mol['[SEP]']])
			temp.extend([dict_mol['[PAD]']]*distance_vector) 
			smiles_emb.append(temp)

	return smiles_emb, smiles_tokens, smiles_str, smiles_ind



def int_to_string_smiles(predictions, smiles_dict):   
	"""This function transforms each SMILES integer token in the respective string token given by the dictionary token. 
	-------
	Args:
	- predictions (list of list of int) ->  smiles integers tokens predicted by the model
	- smiles_dict (dict) -> dictionary of SMILES
  
	-------
	Returns:
	- smiles_pred (list of lists of str) -> list of SMILES strings without [CLS] and [SEP] 
	
	"""
	inv_dict = {v: k for k, v in smiles_dict.items()}   
	smiles_pred=[]
	for mol in range(len(predictions)):
		smile = ''
		for token in range(1,len(predictions[mol][:])):
			if (predictions[mol][token] == smiles_dict['[SEP]'] or predictions[mol][token] == smiles_dict['[PAD]']):
				 break
			smile += f'{inv_dict[predictions[mol][token]]}'
		smiles_pred.append(smile)
	return smiles_pred



def index_remove_tokens(smiles_tokens): #smiles_molecule tokens
	""" This function finds the indexes that are not atoms in molecules.
	-------
	Args:
	- smiles_token (list of lists) -> list of smiles tokens  
  
	-------
	Returns:
	- remove_ind (list of lists of int) -> list of indexes that are not atoms
	
	"""
	remove = ["#", "]", "(", "[", "=", ")"]
	remove_ind = []
	for index_smile in range (len(smiles_tokens)):
		if smiles_tokens[index_smile] in remove:
			remove_ind.append(index_smile)
		elif smiles_tokens[index_smile].isnumeric():
			remove_ind.append(index_smile)
	   
	return remove_ind



def boolean_vector_fg(smiles_string, smiles_tokens):
	""" This function gives a boolean matrix with 1 in the indexes of functional groups.
	-------
	Args:
	- smiles_token (list of int) -> list of smiles tokens  
	- smiles_string (list of str) -> list of smiles strings
  
	-------
	Returns:
	- boolean_vector (lists of int) -> boolean matrix, it length is len(smiles_token)+2 because the start and the end tokens
	
	"""

	remove = index_remove_tokens(smiles_tokens)
	mol = MolFromSmiles(smiles_string)
	groups = identify_functional_groups(mol) 
	index_fg = [y for x in groups for y in x]
	
	not_remove = [x for x in range(len(smiles_tokens)) if x not in remove]
	
	idx = []
	for i in index_fg:
		idx.append(not_remove[i]+1) #+1 because the start token [CLS]
	
	boolean_vector = np.zeros(len(smiles_tokens)+2) # because the start token [CLS] and the end token [SEP]
	boolean_vector[idx] = 1
	boolean_vector = tf.convert_to_tensor(boolean_vector)
	
	return boolean_vector



def index_vector_fg(smiles_string, smiles_tokens): 
	""" This function gives the indexes of functional groups (FGs).
	-------
	Args:
	- smiles_token (list of int) -> list of smiles tokens  
	- smiles_string (list of str) -> list of smiles strings
  
	-------
	Returns:
	- idx_vector_fg (lists of int) -> matrix with FGs indexes, it considers all positions in smiles_token and the start token
	
	"""
	remove = index_remove_tokens(smiles_tokens)
	mol = Chem.MolFromSmiles(smiles_string)
	groups = identify_functional_groups(mol) 
	index_fg = [y for x in groups for y in x]
	
	not_remove = [x for x in range(len(smiles_tokens)) if x not in remove]
	
	idx_vector_fg = []
	for i in index_fg:
		idx_vector_fg.append(not_remove[i]+1) #+1 because the start token [CLS]
	
	return idx_vector_fg


	
def normalization(data, q1 = None, q3 = None):
	""" This function performs the interquartile normalization of data.
	-------
	Args:
	- data (list of float) -> list of data 
	- q1 (float) -> lower quartile of data (none if q1 is calculated based on data)
	- q3 (float) -> upper quartile of data (none if q3 is calculated based on data)
  
	-------
	Returns:
	- data_norm () -> normalized data
	- q1 (float)  
	- q3 (float)  
	"""

	if q1 == None and q3 == None:
		q1 = np.percentile(data, 5)
		q3 = np.percentile(data, 90)

	data_norm = (data - q1) / (q3 - q1)
	return data_norm, q1, q3



def denormalization(data_norm, q1, q3):
	""" This function performs the denormalization of data.
	-------
	Args:
	- data_norm (list of float) -> list of normalized data
	- q1 (float) -> lower quartile of data
	- q3 (float) -> upper quartile of data
  
	-------
	Returns:
	- data () -> denormalized data
	"""

	data = data_norm * (q3 - q1) + q1
	return data



def q2(y_true,y_pred):
    ss_res = tf.reduce_sum((y_true-y_pred)**2)
    y_mean = tf.reduce_mean(y_true)
    ss_tot = tf.reduce_sum((y_true-y_mean)**2)
    r2 = 1.0-ss_res/ss_tot
    return r2
 


def ccc(y_true,y_pred):
    ''' Concordance Correlation Coefficient'''

    x = tf.cast(y_true, dtype=tf.float32)
    y = tf.cast(y_pred, dtype=tf.float32)

    x_mean = tf.reduce_mean(x)
    y_mean = tf.reduce_mean(y)

    x_var =  tf.math.reduce_variance(x)
    y_var =  tf.math.reduce_variance(y)

    error_x = x-x_mean
    error_y = y-y_mean
    
    covar = tf.reduce_sum(error_x*error_y)/tf.cast(len(x), dtype=tf.float32)
    rhoc = 2*covar/(x_var + y_var + (x_mean - y_mean)**2 )
    return rhoc



def save_func(file_path,values, header_list):
    if os.path.isfile(file_path):            
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(values)
    else:
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header_list)
            writer.writerow(values)



def find_clusters(data_path):
    clusters = []
    for i in data_path:
        clusters.append(('train', pd.read_csv(i,header=None)))
    return clusters



def inference_metrics(model,data):
    pred_values = model.predict([data[0]])
    mse = tf.keras.losses.MeanSquaredError()
    metrics = {'MSE': mse(data[1],pred_values),'R2': q2(data[1],pred_values),'CCC':ccc(data[1],pred_values)}
    return metrics



def validity(smiles_list):  
	"""
	Function that takes as input a list containing SMILES strings and determine which ones are valid.
	----------
	Args:
	- smiles_list (list of str) -> List with SMILES strings
	
	-------
	Returns:
	- valid_smiles (list of str) ->  list of valid SMILES molecules in smiles_list input
	- perc_valid (float) -> percentage of valid molecules in input list
	"""
	total = len(smiles_list)
	valid_smiles =[]
	count = 0
	for sm in smiles_list:
		if MolFromSmiles(sm) != None and sm !='':
			valid_smiles.append(sm)
			count = count +1
	perc_valid = count*100/total
	return valid_smiles, perc_valid    



def uniqueness(smiles_list):	
	"""
	Function that takes as input a list containing valid SMILES strings and determine the uniqueness of the input.
	----------
	Args:
	- smiles_list (list of str) -> List with valid SMILES strings
	
	-------
	Returns:
	- perc_unique (float) -> percentage of unique molecules in input list
	"""  

	unique_smiles=list(set(smiles_list))
	perc_unique = (len(unique_smiles)/len(smiles_list))*100
	return perc_unique



def diversity(smiles_list):
	"""
	Function that takes as input a list containing SMILES strings to compute
	its internal diversity
	----------
	Args:
	- smiles_list (list of str) -> List with valid SMILES strings
	
	-------
	Returns:
	- td (float) ->  internal diversity of the list given as input, 
	based on the computation Tanimoto similarity
	"""
	td = 0
	
	fps_A = []
	for i, row in enumerate(smiles_list):
		try:
			mol = MolFromSmiles(row)
			fps_A.append(AllChem.GetMorganFingerprint(mol, 3))
		except:
			print('ERROR: Invalid SMILES!')
			
		
	for ii in range(len(fps_A)):
		for xx in range(len(fps_A)):
			tdi = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
			td += tdi          
	  
	td = td/len(fps_A)**2
	return td



def smiles2mol(smiles_list):
	"""
	Function that converts a list of SMILES strings to a list of RDKit molecules 
	----------
	Args:
	- smiles_list (list of str) -> List of SMILES strings
	----------
	Returns:
	- mol_list (list of str) -> list of molecules objects 
	"""

	mol_list = []
	if isinstance(smiles_list,str):
		mol = MolFromSmiles(smiles_list, sanitize=True)
		mol_list.append(mol)
	else:
		for smi in smiles_list:
			mol = MolFromSmiles(smi, sanitize=True)
			mol_list.append(mol)
	return mol_list



def qed_calculator(mols):
	"""
	Function that takes as input a list of SMILES to predict its qed value
	
	Parameters
	----------
	Args:
	mols (list of str) -> list of molecules
	----------
	Returns:
	- qed_values (list of float) -> list of qed values 
	"""

	qed_values = []
	for mol in mols:
		try:
			q = QED.qed(mol)
			qed_values.append(q)
		except: 
			pass
		
	return qed_values



def scatter_plot(y_test, pred_values,title,filepath):
	fig, ax = plt.subplots()
	ax.scatter(y_test, pred_values, c= mcolors.XKCD_COLORS['xkcd:lavender'].upper(), alpha=0.6, edgecolors='black')
	ax.plot(y_test, y_test, 'k--', lw=2)
	ax.set_xlabel('True', fontsize=10)
	ax.set_ylabel('Predicted', fontsize=10)
	plt.title(title)
	plt.savefig(filepath, dpi=300)



def predict_property(molecules, predictor_model, max_smiles_len, smiles_dictionary, norm_q1, norm_q3):
	tokens_mol = [smiles_str_to_tokens(x) for x in molecules]
	smiles_emb, _,smiles_str,_ = smiles_tokens_to_idx_and_padding(tokens_mol, max_smiles_len, smiles_dictionary)      
	norm_cls_predictions = predictor_model.predict(tf.convert_to_tensor(smiles_emb))
	cls_predictions = norm_cls_predictions*(norm_q3 - norm_q1) + norm_q1 
	return cls_predictions, smiles_str



def compare_models(molecules_unbiased, molecules_biased, predictor_model, norm_q1, norm_q3, max_smiles_len, smiles_dictionary, filepath):

	pred_unbiased, smiles_unb = predict_property(molecules_unbiased, predictor_model, max_smiles_len, smiles_dictionary, norm_q1, norm_q3)
	pred_biased, smiles_bias = predict_property(molecules_biased, predictor_model, max_smiles_len, smiles_dictionary, norm_q1, norm_q3)

	ax = sns.kdeplot(pred_unbiased[:,0], shade=True,color='r')
	sns.kdeplot(pred_biased[:,0], shade=True,color='g')
	ax.set(xlabel='Predicted pIC50 for AA2A receptor', title='Comparison between unbiased and biased Generators')
	plt.legend(['unbiased', 'biased'])
	plt.savefig(f'{filepath}_Unbiased_Biased_Generators_pIC50.png', dpi=300)


	sas_unb= SAscore(smiles2mol((smiles_unb)))
	logp_unb = [rdMolDescriptors.CalcCrippenDescriptors(m)[0] for m in smiles2mol(smiles_unb)]
	mw_unb = [rdMolDescriptors._CalcMolWt(m) for m in smiles2mol(smiles_unb)]


	sas_bias = SAscore(smiles2mol(smiles_bias))
	logp_bias = [rdMolDescriptors.CalcCrippenDescriptors(m)[0] for m in smiles2mol(smiles_bias)]
	mw_bias = [rdMolDescriptors._CalcMolWt(m) for m in smiles2mol(smiles_bias)]

	
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

	ax1.scatter(pred_biased[:,0],sas_bias, c='g', alpha=0.6, edgecolors='black')
	ax1.set_xlabel('pIC50', fontsize=10)
	ax1.set_ylabel('SAS', fontsize=10)   

	ax2.scatter(pred_unbiased[:,0],sas_unb, c='r', alpha=0.6, edgecolors='black')
	ax2.set_xlabel('pIC50', fontsize=10)
	ax2.set_ylabel('SAS', fontsize=10)

	ax3.scatter(mw_bias,logp_bias, c='g', alpha=0.6, edgecolors='black')
	ax3.set_xlabel('Mw', fontsize=10)
	ax3.set_ylabel('LogP', fontsize=10)    

	ax4.scatter(mw_unb,logp_unb, c='r', alpha=0.6, edgecolors='black')
	ax4.set_xlabel('Mw', fontsize=10)
	ax4.set_ylabel('LogP', fontsize=10)   

	ax1.set_title("Biased Generator")
	ax2.set_title("Unbiased Generator")

	plt.suptitle("Comparison between biased and unbiased Generators")
	plt.savefig(f'{filepath}_Unbiased_Biased_Generators_pIC50_SAS_LogP_Mw.png', dpi=300)
	plt.show()

