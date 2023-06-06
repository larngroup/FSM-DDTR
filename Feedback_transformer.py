# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:47:18 2022

@author: Ana Catarina
"""



import pandas as pd
import numpy as np
import json

#import matplotlib.pyplot as plt
#import seaborn as sns

from prepare_data_classification import token, embedding_token_id_and_padding, normalization
from predict_mol import Generator_
import tensorflow as tf
import tensorflow_addons as tfa
#from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MolFromSmiles, AllChem, QED, Descriptors,rdMolDescriptors, FindMolChiralCenters
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

import time

from sascorer_calculator import SAscore
from pareto_front_TiagoC import nsga, nsga3, nsga5

def smiles2mol(smiles_list):
   
	mol_list = []
	if isinstance(smiles_list,str):
		mol = MolFromSmiles(smiles_list, sanitize=True)
		mol_list.append(mol)
	else:
		for smi in smiles_list:
			mol = MolFromSmiles(smi, sanitize=True)
			mol_list.append(mol)
	return mol_list



def listStr_to_listInt(df_column):
	list_int = []
	for row in range(len(df_column)):           
		x = df_column[row]
		
		x = x.replace(",","")
		x = x.replace("[","")
		x = x.replace("]","")
		x = x.split()
		list_int.append([int(x[i]) for i in range(len(x))])
	return list_int

def q2(y_true,y_pred):
	ss_res = tf.reduce_sum((y_true-y_pred)**2)
	y_mean = tf.reduce_mean(y_true)
	ss_tot = tf.reduce_sum((y_true-y_mean)**2)
	r2 = 1.0-ss_res/ss_tot
	#print(r2)
	return r2
	

def ccc(y_true,y_pred):
	''' Concordance Correlation Coefficient'''

	#print(f'y {y_true} ,{y_pred}')
	x = tf.cast(y_true, dtype=tf.float32)
	y = tf.cast(y_pred, dtype=tf.float32)
	
	#print(f'y {x} ,{y}')
	x_mean = tf.reduce_mean(x)
	y_mean = tf.reduce_mean(y)
	#print(f'mean {x_mean} ,{y_mean}')
	x_var =  tf.math.reduce_variance(x)
	y_var =  tf.math.reduce_variance(y)
	#print(f'var {x_var} ,{y_var}')
	error_x = x-x_mean
	error_y = y-y_mean
	
	#print(f'Error {error_x} ,{error_y}')
	covar = tf.reduce_sum(error_x*error_y)/tf.cast(len(x), dtype=tf.float32)
	#print(covar)
	rhoc = 2*covar/(x_var + y_var + (x_mean - y_mean)**2 )
	#print(rhoc)
	#sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
	#rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
	return rhoc


def validity(smiles_list):  
	total = len(smiles_list)
	valid_smiles =[]
	count = 0
	for sm in smiles_list:
		if MolFromSmiles(sm) != None and sm !='':
			valid_smiles.append(sm)
			count = count +1
	perc_valid = count*100/total
	return valid_smiles, perc_valid    
	

class Feedback_transformers():
	def __init__(self, path_predictor, path_generator, loss_object, optimizer, smiles_dictionary, norm_q1, norm_q3, q2, ccc):
		self.model_generator = tf.keras.models.load_model(path_generator)
		self.model_predictor = tf.keras.models.load_model(path_predictor,custom_objects={'q2':q2, 'ccc':ccc})
		self.loss_object = loss_object
		self.optimizer = optimizer
		self.smiles_dictionary = smiles_dictionary
		self.norm_q1 = norm_q1
		self.norm_q3 = norm_q3
		
	def loss_function(self, y_true,y_pred):  
		loss_ = self.loss_object(y_true, y_pred)
		mask = tf.math.logical_not(tf.math.equal(y_true, 0)) # true para elementos diferentes de 0
		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask
		#print(f'loss {loss_}')
		return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

	
	def accuracy_function(self, y_true,y_pred):
		mask = tf.math.logical_not(tf.math.equal(y_true, 0)) # true para elementos diferentes de 0
		mask = tf.cast(mask, dtype=tf.float32)
		pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.int32)
		#print(f'output_real {output_real.shape} and argmax_indices {argmax_indices.shape}')
		accuracy = tf.equal(y_true,pred)
		#print('b')
		acc = tf.cast((accuracy), dtype=tf.float32)
		acc *= mask
		return tf.reduce_sum(acc)/tf.reduce_sum(mask)
	 
	
	def train_generator(self, data):
		
		x = data[:,:-1]
		target = data[:,1:]
		with tf.GradientTape() as tape:
			predictions, _ = self.model_generator(x, training = True)
			#print(f'pred {predictions}')
			loss = self.loss_function(target, predictions) #####ver loss function
		
		gradients = tape.gradient(loss, self.model_generator.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model_generator.trainable_variables))       
		acc = self.accuracy_function(target, predictions)
		return loss, acc
	
   
	def train_feedback(self, x_train, x_train_smiles, epochs, batch_size, num_gen_mol, max_molec_threshold, sample_temperature, num_molec):
		
		for epoch in range(epochs):
			
			print(f'Epoch {epoch+1}/{epochs}')
			
			start = time.time()
			data = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder = False)
			
			loss_epoch = []
			acc_epoch = []
			
			for num, batch in enumerate(data):
				loss_generator, acc_generator = self.train_generator(batch)
				loss_epoch.append(loss_generator)
				acc_epoch.append(acc_generator)
				
				if num == len(data)-1:
					
					#smiles_generator = SMILES_Generator_(self.model_generator, 73, '[CLS]','[SEP]', self.smiles_dictionary, sample_temperature=sample_temperature) 
					smiles_generator = Generator_(self.model_generator, 73, '[CLS]','[SEP]', self.smiles_dictionary, sample_temperature=sample_temperature) 
					smiles_list, valid_smiles, valid = smiles_generator(num_gen_mol)
					print(f'Num valid smiles:  {len(valid_smiles)}, % valid:{valid}')
					#x_train, x_train_smiles = self.update_mol(valid_smiles, x_train_smiles,x_train, max_molec_threshold,epoch+1,num_gen_mol, order = 'desc')
					
					x_train, x_train_smiles = self.update_nsga5(valid_smiles, x_train_smiles,x_train, max_molec_threshold,epoch+1,num_gen_mol, order = 'desc')
					print(f'{num+1}/{len(data)} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f} - accuracy: {np.mean(acc_epoch):.4f}')    
		
					print('Saving model...')
					#self.model_generator.save_weights(f'../model/feedback/path_{epochs}epochs_{batch_size}batch_{num_gen_mol}molec_{max_molec_threshold}new_{sample_temperature}T_train{num_molec}molec.h5')
					self.model_generator.save(f'../model/feedback/model_kop_{epochs}epochs_{batch_size}batch_{num_gen_mol}molec_{max_molec_threshold}new_{sample_temperature}T_train{num_molec}molec_pIC50_SAS_Mw_TPSA_LogP.h5py')
					
					
					
					
	def update_data(self, gen_smiles, train_smiles, x_train, max_molec_threshold, epoch, order = 'asc'):
		
		 pred_smiles = [molec for molec in gen_smiles if molec not in train_smiles]
		 #print(pred_smiles)
		 tokens_mol = [token(x) for x in pred_smiles]
		 pred_smiles_emb, smiles_mol = embedding_token_id_and_padding(tokens_mol, 72, self.smiles_dictionary) 
		 
		 pred_smiles = [''.join(m) for m in smiles_mol]
		 
		 #print(f'pred {len(pred_smiles_emb)}, {len(pred_smiles_emb[0])}')
		 #print(tf.convert_to_tensor(pred_smiles_emb))
		 #print(f'shape pred_smiles_emb {tf.convert_to_tensor(pred_smiles_emb).shape}')
		 norm_cls_predictions = self.model_predictor.predict(tf.convert_to_tensor(pred_smiles_emb))
		 #print(f'norm_cls_predictions {norm_cls_predictions.shape}')
		 cls_predictions = norm_cls_predictions*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 #print(f'cls_predictions {np.array(cls_predictions[:,0])}')
		 sort_ind_pred =  np.array(cls_predictions[:,0]).argsort()
		 #print(sort_ind_pred)
		 
		 
		 norm_cls_data_train = self.model_predictor.predict(x_train)
		 cls_data_train = norm_cls_data_train*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 #print(f'cls_data_train {np.array(cls_data_train[:,0])}')
		 sort_ind_data_train =  np.array(cls_data_train[:,0]).argsort()
		 
		 
		 if order == 'desc':
			 sort_ind_pred = sort_ind_pred[::-1]
			 #print(sort_ind_pred)
			 sort_ind_data_train = sort_ind_data_train[::-1]             

		 
		 if len(sort_ind_pred) < max_molec_threshold:
			 max_molec_threshold = len(sort_ind_pred)
		 
		 sorted_pred_smiles_emb = np.array(pred_smiles_emb)[sort_ind_pred]
		 #print(sorted_pred_smiles_emb, sorted_pred_smiles_emb.shape)
		 sorted_pred_smiles = np.array(pred_smiles)[sort_ind_pred]
		 #print(sorted_pred_smiles, sorted_pred_smiles.shape)
		 
		 sorted_train_smiles = np.array(train_smiles)[sort_ind_data_train]
		 #print(sorted_train_smiles.shape)
		 sorted_train_smiles_emb = np.array(x_train)[sort_ind_data_train] 
		 #print(sorted_train_smiles_emb.shape)
		 
		 new_smiles_train = np.concatenate((sorted_train_smiles[0:len(sort_ind_data_train)-max_molec_threshold],
											sorted_pred_smiles[0:max_molec_threshold]), axis = 0)
		 #print(new_smiles_train.shape)
		 new_x_train = np.concatenate((sorted_train_smiles_emb[0:len(sort_ind_data_train)-max_molec_threshold, :],
									  sorted_pred_smiles_emb[0:max_molec_threshold, :]), axis = 0)
		 #print(new_x_train.shape)
		 #print(f'sorted_pred_smiles_emb {tf.convert_to_tensor(sorted_pred_smiles_emb[0:max_molec_threshold, :]).shape}')
		 new_norm_data_cls = self.model_predictor.predict(tf.convert_to_tensor(sorted_pred_smiles_emb[0:max_molec_threshold, :]))
		 #print(new_norm_data_cls)
		 new_data_cls = new_norm_data_cls*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 #print(new_data_cls)
		 
		 #print(np.array([[epoch]]*max_molec_threshold))
		 #print(sorted_pred_smiles_emb[0:max_molec_threshold,:])
		 
		 df = pd.DataFrame(list(zip(np.array([[epoch]]*max_molec_threshold),sorted_pred_smiles[0:max_molec_threshold], new_data_cls )),
			   columns =['epoch','smiles','pIC50'])
		 
		 df.to_csv('../molecules/feedback/bias_molecules.csv',mode='a',header=False, index=False)
		 #escrever no csv new_data_cls para fazer a distribuicao
		 new_x_train = tf.cast(new_x_train, dtype=tf.int32)
		 return new_x_train, new_smiles_train
		
	def update_mol(self, gen_smiles, train_smiles, x_train, max_molec_threshold, epoch, num_gen_mol, order = 'asc'):
		
		 pred_smiles = [molec for molec in gen_smiles if molec not in train_smiles]
		 #print(pred_smiles)
		 tokens_mol = [token(x) for x in pred_smiles]
		 pred_smiles_emb, smiles_mol = embedding_token_id_and_padding(tokens_mol, 72, self.smiles_dictionary) 
		 
		 pred_smiles = [''.join(m) for m in smiles_mol]
		 
		 smiles_pred = [MolFromSmiles(s) for s in pred_smiles]
		 
		 hba = [rdMolDescriptors.CalcNumHBA(m) for m in smiles_pred]
		 hbd = [rdMolDescriptors.CalcNumHBD(m) for m in smiles_pred]
		 rtb = [rdMolDescriptors.CalcNumRotatableBonds(m) for m in smiles_pred]
		 psa = [rdMolDescriptors.CalcTPSA(m) for m in smiles_pred]
		 mw = [rdMolDescriptors._CalcMolWt(m) for m in smiles_pred]
		 logp = [rdMolDescriptors.CalcCrippenDescriptors(m)[0] for m in smiles_pred]
			 
		 ind_opt_mol = []
		 for indx_mol in range(len(pred_smiles)):             
			 if (hba[indx_mol]<10) and (hbd[indx_mol]<5) and (rtb[indx_mol]<10) and (psa[indx_mol]<140) and (mw[indx_mol]<500) and (logp[indx_mol]<5):
					  ind_opt_mol.append(indx_mol) 
				 
		 
		 emb_smiles = np.array(pred_smiles_emb)
		 print(f'emb_smiles {emb_smiles.shape}, ind_opt_mol {len(ind_opt_mol)}')
		 emb_smiles = emb_smiles[ind_opt_mol,:]
		 print(f'emb_smiles {emb_smiles.shape}')
		 #print(f'pred {len(pred_smiles_emb)}, {len(pred_smiles_emb[0])}')
		 #print(tf.convert_to_tensor(pred_smiles_emb))
		 norm_cls_predictions = self.model_predictor.predict(tf.convert_to_tensor(emb_smiles))
		 cls_predictions = norm_cls_predictions*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 #print(f'cls_predictions {np.array(cls_predictions[:,0])}')
		 sort_ind_pred =  np.array(cls_predictions[:,0]).argsort()
		 #print(sort_ind_pred)
		 
		 
		 norm_cls_data_train = self.model_predictor.predict(x_train)
		 cls_data_train = norm_cls_data_train*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 #print(f'cls_data_train {np.array(cls_data_train[:,0])}')
		 sort_ind_data_train =  np.array(cls_data_train[:,0]).argsort()
		 
		 
		 if order == 'desc':
			 sort_ind_pred = sort_ind_pred[::-1]
			 #print(sort_ind_pred)
			 sort_ind_data_train = sort_ind_data_train[::-1]             

		 
		 if len(sort_ind_pred) < max_molec_threshold:
			 max_molec_threshold = len(sort_ind_pred)
		 
		 sorted_pred_smiles_emb = np.array(pred_smiles_emb)[sort_ind_pred]
		 #print(sorted_pred_smiles_emb, sorted_pred_smiles_emb.shape)
		 sorted_pred_smiles = np.array(pred_smiles)[sort_ind_pred]
		 #print(sorted_pred_smiles, sorted_pred_smiles.shape)
		 
		 sorted_train_smiles = np.array(train_smiles)[sort_ind_data_train]
		 #print(sorted_train_smiles.shape)
		 sorted_train_smiles_emb = np.array(x_train)[sort_ind_data_train] 
		 #print(sorted_train_smiles_emb.shape)
		 
		 new_smiles_train = np.concatenate((sorted_train_smiles[0:len(sort_ind_data_train)-max_molec_threshold],
											sorted_pred_smiles[0:max_molec_threshold]), axis = 0)
		 #print(new_smiles_train.shape)
		 new_x_train = np.concatenate((sorted_train_smiles_emb[0:len(sort_ind_data_train)-max_molec_threshold, :],
									  sorted_pred_smiles_emb[0:max_molec_threshold, :]), axis = 0)
		 #print(new_x_train.shape)
		 new_norm_data_cls = self.model_predictor.predict(tf.convert_to_tensor(sorted_pred_smiles_emb[0:max_molec_threshold, :]))
		 #print(new_norm_data_cls)
		 new_data_cls = new_norm_data_cls*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 #print(new_data_cls)
		 
		 #print(np.array([[epoch]]*max_molec_threshold))
		 #print(sorted_pred_smiles_emb[0:max_molec_threshold,:])
		 
		# df = pd.DataFrame(list(zip(np.array([[epoch]]*max_molec_threshold),sorted_pred_smiles[0:max_molec_threshold], new_data_cls )),
		#	   columns =['epoch','smiles','pIC50'])
		 
		 #df.to_csv('../molecules/feedback/bias_molecules.csv',mode='a',header=False, index=False)
		 #escrever no csv new_data_cls para fazer a distribuicao
		 new_x_train = tf.cast(new_x_train, dtype=tf.int32)
		 return new_x_train, new_smiles_train
		 
	def update_nsga2(self, gen_smiles, train_smiles, x_train, max_molec_threshold, epoch, num_gen_mol, order = 'asc'):
		
		 pred_smiles = [molec for molec in gen_smiles if molec not in train_smiles]
		 #print(pred_smiles)
		 tokens_mol = [token(x) for x in pred_smiles]
		 pred_smiles_emb, smiles_mol = embedding_token_id_and_padding(tokens_mol, 72, self.smiles_dictionary) 
		 
		 pred_smiles = [''.join(m) for m in smiles_mol]
		 
		# smiles_pred = [MolFromSmiles(s) for s in pred_smiles]
		 
		 sas_bias = SAscore(smiles2mol((pred_smiles)))
		 sas_unb = SAscore(smiles2mol((train_smiles)))
		 
		 
		 if len(sas_bias) < max_molec_threshold:
			 max_molec_threshold = len(sas_bias)
		 
		 norm_cls_predictions = self.model_predictor.predict(tf.convert_to_tensor(pred_smiles_emb))
		 cls_predictions = norm_cls_predictions*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 
		 df_bias = pd.DataFrame(list(zip(-1*cls_predictions[:,0],sas_bias)),
			   columns =['pIC50','SAS'])
		 
		 norm_cls_predictions_org = self.model_predictor.predict(tf.convert_to_tensor(x_train))
		 cls_predictions_org = norm_cls_predictions_org*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 
		 df = pd.DataFrame(list(zip(-1*cls_predictions_org[:,0],sas_unb)),
			   columns =['pIC50','SAS'])
		 
		 
		 order_data_inp = nsga(df, len(df))
		 argmax_inp = order_data_inp.iloc[0:len(train_smiles)-max_molec_threshold]['index_col_org']
		 argmax_inp = [int(x) for x in argmax_inp]
		
		 order_data_pred = nsga(df_bias, len(df_bias)) 
		 argmax_pred = order_data_pred.iloc[0:max_molec_threshold]['index_col_org']
		 argmax_pred = [int(x) for x in argmax_pred]
		 
		 
		 pred_smiles = np.array(pred_smiles)
		 emb_smiles = np.array(pred_smiles_emb)
		 x_train = np.array(x_train)
		 train_smiles = np.array(train_smiles)
		 
		 print(train_smiles[argmax_inp].shape,pred_smiles[argmax_pred].shape)
		 new_smiles_train = np.concatenate((train_smiles[argmax_inp],pred_smiles[argmax_pred]), axis = 0)
		 #print(new_smiles_train.shape)
		 new_x_train = np.concatenate((x_train[argmax_inp],emb_smiles[argmax_pred]), axis = 0)
		 new_x_train = tf.cast(new_x_train, dtype=tf.int32)
		 return new_x_train, new_smiles_train
	 		
	def update_nsga3(self, gen_smiles, train_smiles, x_train, max_molec_threshold, epoch, num_gen_mol, order = 'asc'):
		
		 pred_smiles = [molec for molec in gen_smiles if molec not in train_smiles]
		 #print(pred_smiles)
		 tokens_mol = [token(x) for x in pred_smiles]
		 pred_smiles_emb, smiles_mol = embedding_token_id_and_padding(tokens_mol, 72, self.smiles_dictionary) 
		 
		 pred_smiles = [''.join(m) for m in smiles_mol]
		 
		# smiles_pred = [MolFromSmiles(s) for s in pred_smiles]
		 
		 sas_bias = SAscore(smiles2mol((pred_smiles)))
		 mw_bias = [rdMolDescriptors._CalcMolWt(m) for m in smiles2mol(pred_smiles)]
		 #tpsa_bias = [rdMolDescriptors.CalcTPSA(m) for m in smiles2mol(pred_smiles)]
		 #logp_bias = [rdMolDescriptors.CalcCrippenDescriptors(m)[0] for m in smiles2mol(pred_smiles)]
		 
		 sas_unb = SAscore(smiles2mol((train_smiles)))
		 mw_unb = [rdMolDescriptors._CalcMolWt(m) for m in smiles2mol(train_smiles)]	
		 #tpsa_unb = [rdMolDescriptors.CalcTPSA(m) for m in smiles2mol(train_smiles)]	 
		 #logp_unb = [rdMolDescriptors.CalcCrippenDescriptors(m)[0] for m in smiles2mol(train_smiles)]
		 
		 if len(sas_bias) < max_molec_threshold:
			 max_molec_threshold = len(sas_bias)
		 
		 norm_cls_predictions = self.model_predictor.predict(tf.convert_to_tensor(pred_smiles_emb))
		 cls_predictions = norm_cls_predictions*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 
		 df_bias = pd.DataFrame(list(zip(-1*cls_predictions[:,0],sas_bias, mw_bias)),columns =['pIC50','SAS','Mw'])
			  # columns =['pIC50','SAS', 'Mw'])  ,columns =['pIC50','SAS', 'TPSA'], columns =['pIC50','SAS', 'LogP']
		 
		 norm_cls_predictions_org = self.model_predictor.predict(tf.convert_to_tensor(x_train))
		 cls_predictions_org = norm_cls_predictions_org*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 
		 df = pd.DataFrame(list(zip(-1*cls_predictions_org[:,0],sas_unb, mw_unb)),columns =['pIC50','SAS','Mw'])
			   #columns =['pIC50','SAS','Mw']) columns =['pIC50','SAS', 'TPSA'], columns =['pIC50','SAS', 'LogP']
		 
		 order_data_inp = nsga3(df, len(df))
		 argmax_inp = order_data_inp.iloc[0:len(train_smiles)-max_molec_threshold]['index_col_org']
		 argmax_inp = [int(x) for x in argmax_inp]
		
		 order_data_pred = nsga3(df_bias, len(df_bias)) 
		 argmax_pred = order_data_pred.iloc[0:max_molec_threshold]['index_col_org']
		 argmax_pred = [int(x) for x in argmax_pred]
		 
		 
		 pred_smiles = np.array(pred_smiles)
		 emb_smiles = np.array(pred_smiles_emb)
		 x_train = np.array(x_train)
		 train_smiles = np.array(train_smiles)
		 
		 print(train_smiles[argmax_inp].shape,pred_smiles[argmax_pred].shape)
		 new_smiles_train = np.concatenate((train_smiles[argmax_inp],pred_smiles[argmax_pred]), axis = 0)
		 #print(new_smiles_train.shape)
		 new_x_train = np.concatenate((x_train[argmax_inp],emb_smiles[argmax_pred]), axis = 0)
		 new_x_train = tf.cast(new_x_train, dtype=tf.int32)
		 return new_x_train, new_smiles_train



	def update_nsga5(self, gen_smiles, train_smiles, x_train, max_molec_threshold, epoch, num_gen_mol, order = 'asc'):
		
		 pred_smiles = [molec for molec in gen_smiles if molec not in train_smiles]
		 #print(pred_smiles)
		 tokens_mol = [token(x) for x in pred_smiles]
		 pred_smiles_emb, smiles_mol = embedding_token_id_and_padding(tokens_mol, 72, self.smiles_dictionary) 
		 
		 pred_smiles = [''.join(m) for m in smiles_mol]
		 
		# smiles_pred = [MolFromSmiles(s) for s in pred_smiles]
		 
		 sas_bias = SAscore(smiles2mol((pred_smiles)))
		 mw_bias = [rdMolDescriptors._CalcMolWt(m) for m in smiles2mol(pred_smiles)]
		 tpsa_bias = [rdMolDescriptors.CalcTPSA(m) for m in smiles2mol(pred_smiles)]
		 logp_bias = [rdMolDescriptors.CalcCrippenDescriptors(m)[0] for m in smiles2mol(pred_smiles)]
		 
		 sas_unb = SAscore(smiles2mol((train_smiles)))
		 mw_unb = [rdMolDescriptors._CalcMolWt(m) for m in smiles2mol(train_smiles)]	
		 tpsa_unb = [rdMolDescriptors.CalcTPSA(m) for m in smiles2mol(train_smiles)]	 
		 logp_unb = [rdMolDescriptors.CalcCrippenDescriptors(m)[0] for m in smiles2mol(train_smiles)]
		 
		 if len(sas_bias) < max_molec_threshold:
			 max_molec_threshold = len(sas_bias)
		 
		 norm_cls_predictions = self.model_predictor.predict(tf.convert_to_tensor(pred_smiles_emb))
		 cls_predictions = norm_cls_predictions*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 
		 df_bias = pd.DataFrame(list(zip(-1*cls_predictions[:,0],sas_bias, mw_bias,tpsa_bias,logp_bias)),columns =['pIC50','SAS','Mw','TSPA','LogP'])
			  # columns =['pIC50','SAS', 'Mw'])  ,columns =['pIC50','SAS', 'TPSA'], columns =['pIC50','SAS', 'LogP']
		 
		 norm_cls_predictions_org = self.model_predictor.predict(tf.convert_to_tensor(x_train))
		 cls_predictions_org = norm_cls_predictions_org*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 
		 df = pd.DataFrame(list(zip(-1*cls_predictions_org[:,0],sas_unb, mw_unb,tpsa_unb,logp_unb)),columns =['pIC50','SAS','Mw','TSPA','LogP'])
			   #columns =['pIC50','SAS','Mw']) columns =['pIC50','SAS', 'TPSA'], columns =['pIC50','SAS', 'LogP']
		 
		 order_data_inp = nsga5(df, len(df))
		 argmax_inp = order_data_inp.iloc[0:len(train_smiles)-max_molec_threshold]['index_col_org']
		 argmax_inp = [int(x) for x in argmax_inp]
		
		 order_data_pred = nsga5(df_bias, len(df_bias)) 
		 argmax_pred = order_data_pred.iloc[0:max_molec_threshold]['index_col_org']
		 argmax_pred = [int(x) for x in argmax_pred]
		 
		 
		 pred_smiles = np.array(pred_smiles)
		 emb_smiles = np.array(pred_smiles_emb)
		 x_train = np.array(x_train)
		 train_smiles = np.array(train_smiles)
		 
		 print(train_smiles[argmax_inp].shape,pred_smiles[argmax_pred].shape)
		 new_smiles_train = np.concatenate((train_smiles[argmax_inp],pred_smiles[argmax_pred]), axis = 0)
		 #print(new_smiles_train.shape)
		 new_x_train = np.concatenate((x_train[argmax_inp],emb_smiles[argmax_pred]), axis = 0)
		 new_x_train = tf.cast(new_x_train, dtype=tf.int32)
		 return new_x_train, new_smiles_train

if __name__ == '__main__':
	
	
	data_path={'data':'../dataset/data_for_generation_notduplicate.csv',
			   'data_train':'../dataset/../dataset/data_kop_train.csv', #data_classification_a2a_train.csv  ,     '../dataset/data_kop_train.csv'
			   'smiles_dic':'../dictionary/dictionary_smiles.txt',
				'generator': '../model/generation/train_without_compile_8_4_relu_RAdam_batch_256_data_1046964.h5py',
				'predictor': '../model/classification/model_4heads_4encoders_RAdam'}
		
	smiles_dictionary = json.load(open(data_path['smiles_dic']))        
			 
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
		from_logits=True, reduction='none')
	optimizer = tfa.optimizers.RectifiedAdam(lr=1e-4)
	
	
	df_cls = pd.read_csv(data_path['data_train'], index_col = False)
	_, norm_q1, norm_q3 = normalization(df_cls['pIC50'].tolist())
	
	print(f'norm_q1 {norm_q1} e norm_q3 {norm_q3}')
	
	
	file='../molecules/molecules_without_compile_0.9T_10000.txt' #'molecules_without_feedback_500.txt'
	f = open(file, "r")
	molecules_org = f.read().splitlines()
	f.close()
	
	valid_org_smiles, perc_valid_org  = validity(molecules_org)
	print(perc_valid_org)
	unique_valid_org_smiles = set(valid_org_smiles)
	tokens_mol = [token(x) for x in unique_valid_org_smiles]
	smiles_emb, smiles_mol = embedding_token_id_and_padding(tokens_mol, 72, smiles_dictionary)  
	smiles_str_mol = [''.join(k) for k in smiles_mol]
	
	max_num_mol = 8500
	
	x_train = smiles_emb[0:max_num_mol]
	x_train_smiles = smiles_str_mol[0:max_num_mol]
	
	transformer  = Feedback_transformers('../model/classification/BERT/model_kop_4_4_RAdam', #'../model/classification/BERT/model_4heads_4encoders_batch_size256_RAdam_linear_1_32_200_RAdam', model_4heads_4encoders_RAdam',
										 '../model/generation/train_without_compile_8_4_relu_RAdam_batch_256_data_1046964.h5py',
										 loss_object, optimizer, smiles_dictionary, norm_q1, norm_q3, q2, ccc)         
	#num_molec = 10000
	#df = pd.read_csv(data_path['data']).iloc[0:num_molec]       
	#x_train = listStr_to_listInt(df['smiles_emb'].to_list())
	#x_train_smiles = df['smiles'].to_list()
	transformer.train_feedback(x_train, x_train_smiles, 60, 128, 1200, 450, 0.9, max_num_mol)              
			
		
		
"""
model_predictor = tf.keras.models.load_model('../model/classification/model_4heads_4encoders_RAdam',custom_objects={'q2':q2, 'ccc':ccc})

model_generator = tf.keras.models.load_model('../model/generation/train_without_compile_8_4_relu_RAdam_batch_256_data_1046964.h5py')
smiles_generator = SMILES_Generator_(model_generator, 73, '[CLS]','[SEP]', smiles_dictionary, top_k=10)
 
miles_list, valid_smiles, valid = smiles_generator(100)
pred_smiles = [molec for molec in valid_smiles if molec not in x_train_smiles]
tokens_mol = [token(x) for x in pred_smiles]
pred_smiles_emb, smiles_mol = embedding_token_id_and_padding(tokens_mol, 72, smiles_dictionary) 
pred_smiles = [''.join(m) for m in smiles_mol]
norm_cls_predictions = model_predictor.predict(tf.convert_to_tensor(pred_smiles_emb))
cls_predictions = norm_cls_predictions*(norm_q3 - norm_q1) + norm_q1 
sort_ind_pred =  np.array(cls_predictions[:,0]).argsort()
sort_ind_pred = sort_ind_pred[::-1]
max_molec_threshold = 70
if len(sort_ind_pred) < max_molec_threshold:
	max_molec_threshold = len(sort_ind_pred)
		 
sorted_pred_smiles_emb = np.array(pred_smiles_emb)[sort_ind_pred]
sorted_pred_smiles = np.array(pred_smiles)[sort_ind_pred]

new_norm_data_cls = model_predictor.predict(tf.convert_to_tensor(sorted_pred_smiles_emb[0:max_molec_threshold, :]))
new_data_cls = new_norm_data_cls*(norm_q3 - norm_q1) + norm_q1 












df = pd.read_csv('../molecules/feedback/bias_molecules.csv', index_col = False)#pd.read_csv('../molecules/feedback/bias_molecules_20epochs_32batch_100moleculesgen_30new.csv', index_col = False)
#df = df.iloc[60::]

df = df.groupby('epoch')
df_epochs = df['pIC50'].apply(list)


def Str_to_listInt(df_column):
	list_int = []
	for row in range(len(df_column)):           
		x = df_column[row]
		x = x.replace("[","")
		x = x.replace("]","")
		#print(x)
		list_int.append(float(x) )
	return list_int

pIC50_list_epochs = [Str_to_listInt(df_epochs.to_list()[i]) for i in range(len(df_epochs.to_list()))]

#indx = [10,11,12,13,14,15,16,17,18,19,1,20,2,3,4,5,6,7,8,9]

indx = [10,1,20,2,3,4,5,6,7,8,9]


sns.displot(pIC50_list_epochs, kind="kde", fill=True, legend= False)
plt.legend(labels=indx)

list_org_1_10_20_epochs = [new_data_cls[:,0].tolist()] +[pIC50_list_epochs[indx.index(1)]]+[pIC50_list_epochs[indx.index(5)]]+[pIC50_list_epochs[indx.index(10)]]

sns.displot(list_org_1_10_20_epochs, kind="kde", fill=True, legend= False)
plt.legend(labels=["original","epoch1","epoch5", "epoch10"])
plt.title('Predicted pIC50 for AA2A receptor for the generated molecules')
plt.xlabel('Predicted pIC50 for AA2A receptor')



"""    
