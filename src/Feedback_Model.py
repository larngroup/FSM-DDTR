# -*- coding: utf-8 -*-
"""

"""



import pandas as pd
import numpy as np
import json

import tensorflow as tf
import tensorflow_addons as tfa

import time

import argparse
import sys

from utils import *
from Generator import Generator_Model

from rdkit.Chem import MolFromSmiles, AllChem, QED, Descriptors,rdMolDescriptors, FindMolChiralCenters
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit import DataStructs




def cmd_options():
	parser = argparse.ArgumentParser(description='Program that trains and generates molecules based on Feedback of pretrained generator and a predictor.')

	parser.add_argument('--train', action='store_true', help='Training')
	parser.add_argument('--generate', action='store_true', help='Generating')
	parser.add_argument('--compare', action='store_true', help='Compare biased and unbiased generators')

	parser.add_argument('--max_smiles_len', type = int,  default=72, metavar='', help='Maximum length of SMILES for training, needs to be larger than 0')
	parser.add_argument('--batch', type = int,  default=128, metavar='', help='Batch size')
	parser.add_argument('--epochs', type = int,  default=100, metavar='', help='Number of epochs')
	parser.add_argument('--optimizer', type = str,  default='RAdam', metavar='', help='Optimizer')
	parser.add_argument('--patience', type = int,  default=20, metavar='', help='Patience in EarlyStopping')
	parser.add_argument('--min_delta', type = float,  default=0.001, metavar='', help='EarlyStopping mininum delta')
	
	parser.add_argument('--num_gen', type=int, default = 1200, metavar='', help='Number of molecules to generate in each epoch')
	parser.add_argument('--num_add', type=int, default = 450, metavar='', help='Number of molecules with the best properties to add to the input in each epoch')
	parser.add_argument('--temperature', type=float, default = 0.9, metavar='', help='Sample temperature used to add randomness to generation')
	parser.add_argument('--asc', type=bool, default = False, metavar='', help='True if to order the molecules by ascending order')
	parser.add_argument('--rule5',  type=bool, default = True, metavar='', help='True if the optimization is based on the rule of 5')

	parser.add_argument('--inp_mol', type = str, default='../molecules/pre_train/molecules_without_compile_0.9T_10000.txt', metavar='', help='Path to the file with the molecules used as input in the model')
	parser.add_argument('--num_inp', type = int, default=8500 , metavar='', help='Number of input molecules used to train the model')
	parser.add_argument('--generator', type = str, default="../models/generation/train_without_compile_8_4_relu_RAdam_batch_256_data_1046964.h5py", metavar='', help='Path to the model file with the unbiased generator pretrained')
	parser.add_argument('--predictor', type = str, default = "../models/classification/model_4heads_4encoders_batch_size256_RAdam_linear_1_32_200_RAdam", metavar='', help='Path to the model file with the predictor model trained')
	parser.add_argument('--norm', type = str, default="../dataset/norm_q1_q3_data_classification_a2a_train.txt", metavar='', help='Path to file with the parameters used to normalize')
	parser.add_argument('--num_generation', type=int, default=500, metavar='', help='Amount of molecules to generate')
	parser.add_argument('--metrics',  type=bool, default = True, metavar='', help='True if to show diversity, validity, uniqueness metrics')
	parser.add_argument('--mol_path', type=str, default = "../molecules/feedback/", metavar='', help='Path to store the resulting molecules from the trained generator (include valid and non-valid molecules)')
	
	parser.add_argument('--dict', type = str, default="../dictionary/dictionary_smiles.txt", metavar='', help='dictionary smiles path')
	parser.add_argument('--save_path', type = str, default="../models/feedback/", metavar='', help='save model path')
	parser.add_argument('--save_csv', type = str, default = "../results/feedback/results.csv", metavar='', help='Path of csv to write the results of generation')
	parser.add_argument('--saved_model', required='--generate' in sys.argv, type = str, metavar='', help='saved model path')
	parser.add_argument('--path_results', type=str, default = "../results/feedback/", metavar='', help='Path to save the graphs results')

	parser.add_argument('--biased_mol', type=str, default = "../molecules/feedback/molecules_opt_rule5_0.9T_500_1200_450.txt", metavar='', help='Path with the biased molecules')
	parser.add_argument('--unbiased_mol', type=str, default = "../molecules/feedback/molecules_without_feedback_0.9T_500.txt", metavar='', help='Path with the unbiased molecules')

	args = parser.parse_args()
	return args


	

class Feedback_transformers():
	def __init__(self, path_predictor, path_generator, max_smiles_len, smiles_dictionary, norm_q1, norm_q3, q2, ccc, loss_object, optimizer, patience, min_delta):
		self.generator_model = Generator_Model(None, None, None, None, max_smiles_len, len(smiles_dictionary),
									 None, None, loss_object, optimizer, smiles_dictionary, None, None, path_saved_model = path_generator)
		self.predictor_model = tf.keras.models.load_model(path_predictor,custom_objects={'q2':q2, 'ccc':ccc})

		self.smiles_dictionary = smiles_dictionary
		self.norm_q1 = norm_q1
		self.norm_q3 = norm_q3
		self.max_smiles_len = max_smiles_len
		self.patience = patience
		self.min_delta = min_delta
				

   
	def train_feedback(self, x_train, x_train_smiles, epochs, batch_size, num_gen_mol, max_molec_threshold, sample_temperature, num_molec, save_path, rule5 = True):
		data = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder = False)
		last_loss = {'epoch':0,'value':1000}

		for epoch in range(epochs):
			
			print(f'Epoch {epoch+1}/{epochs}')
			
			start = time.time()
		
			
			loss_epoch = []
			acc_epoch = []
			
			for num, batch in enumerate(data):
				
				x = batch[:,:-1]
				target = batch[:,1:]

				loss_generator, acc_generator = self.generator_model.train_step(x,target)#self.train_generator(batch)
				loss_epoch.append(loss_generator)
				acc_epoch.append(acc_generator)
				
				if num == len(data)-1:
					smiles_list, valid_smiles, valid = self.generator_model.generate_molecules(num_gen_mol, '[CLS]', sample_temperature) 
					print(f'Num valid smiles:  {len(valid_smiles)}, % valid:{valid}')
					
					if rule5:
						x_train, x_train_smiles = self.update_rule_of_five(valid_smiles, x_train_smiles,x_train, max_molec_threshold, epoch+1, order = 'desc')

					else:
						x_train, x_train_smiles = self.update_only_by_property(valid_smiles, x_train_smiles,x_train, max_molec_threshold, epoch+1, order = 'desc')
					#x_train, x_train_smiles = self.update_rule_of_five(valid_smiles, x_train_smiles,x_train, max_molec_threshold, epoch+1, order = 'desc')
					
					print(f'{num+1}/{len(data)} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f} - accuracy: {np.mean(acc_epoch):.4f}')    
		
			
			if (last_loss['value'] - np.mean(loss_epoch)) >= self.min_delta: 
				last_loss['value'] = np.mean(loss_epoch)
				last_loss['epoch'] = epoch+1
				if save_path != None:
					print('Saving model...')
					if rule5:
						self.generator_model.model.save(f'{save_path}_rule5py') #model_{epochs}epochs_{batch_size}batch_{num_gen_mol}molec_{max_molec_threshold}new_{sample_temperature}T_train{num_molec}molec.h5py')
					
					else:
						self.generator_model.model.save(f'{save_path}.h5py') #model_{epochs}epochs_{batch_size}batch_{num_gen_mol}molec_{max_molec_threshold}new_{sample_temperature}T_train{num_molec}molec.h5py')
					
				   
			if ((epoch+1) - last_loss['epoch']) >= self.patience:
				break 		

					
		
					
	def update_only_by_property(self, gen_smiles, train_smiles, x_train, max_molec_threshold, epoch, order = 'asc'):
		
		 pred_smiles = [molec for molec in gen_smiles if molec not in train_smiles]

		 tokens_mol = [smiles_str_to_tokens(x) for x in pred_smiles]
		 pred_smiles_emb, _, pred_smiles_str,_  = smiles_tokens_to_idx_and_padding(tokens_mol, self.max_smiles_len-1, self.smiles_dictionary)  
		
		 norm_cls_predictions = self.predictor_model.predict(tf.convert_to_tensor(pred_smiles_emb))
		 cls_predictions = norm_cls_predictions*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 sort_ind_pred =  np.array(cls_predictions[:,0]).argsort()
		 
		 norm_cls_data_train = self.predictor_model.predict(x_train)
		 cls_data_train = norm_cls_data_train*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 sort_ind_data_train =  np.array(cls_data_train[:,0]).argsort()
		 
		 
		 if order == 'desc':
			 sort_ind_pred = sort_ind_pred[::-1]
			 sort_ind_data_train = sort_ind_data_train[::-1]             

		 
		 if len(sort_ind_pred) < max_molec_threshold:
			 max_molec_threshold = len(sort_ind_pred)
		 
		 sorted_pred_smiles_emb = np.array(pred_smiles_emb)[sort_ind_pred]
		 sorted_pred_smiles = np.array(pred_smiles_str)[sort_ind_pred]
		 
		 sorted_train_smiles = np.array(train_smiles)[sort_ind_data_train]
		 sorted_train_smiles_emb = np.array(x_train)[sort_ind_data_train] 
		 
		 
		 new_smiles_train = np.concatenate((sorted_train_smiles[0:len(sort_ind_data_train)-max_molec_threshold],
											sorted_pred_smiles[0:max_molec_threshold]), axis = 0)

		 new_x_train = np.concatenate((sorted_train_smiles_emb[0:len(sort_ind_data_train)-max_molec_threshold, :],
									  sorted_pred_smiles_emb[0:max_molec_threshold, :]), axis = 0)
		 
		 new_norm_data_cls = self.predictor_model.predict(tf.convert_to_tensor(sorted_pred_smiles_emb[0:max_molec_threshold, :]))
		 new_data_cls = new_norm_data_cls*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 new_x_train = tf.cast(new_x_train, dtype=tf.int32)

		 return new_x_train, new_smiles_train
		

	def update_rule_of_five(self, gen_smiles, train_smiles, x_train, max_molec_threshold, epoch, order = 'asc'):
		
		 pred_smiles = [molec for molec in gen_smiles if molec not in train_smiles]

		 tokens_mol = [smiles_str_to_tokens(x) for x in pred_smiles]
		 pred_smiles_emb, _, pred_smiles_str,_  = smiles_tokens_to_idx_and_padding(tokens_mol, self.max_smiles_len-1, self.smiles_dictionary)  
		
		 mol_smiles = [MolFromSmiles(s) for s in pred_smiles_str]
		 
		 hba = [rdMolDescriptors.CalcNumHBA(m) for m in mol_smiles]
		 hbd = [rdMolDescriptors.CalcNumHBD(m) for m in mol_smiles]
		 rtb = [rdMolDescriptors.CalcNumRotatableBonds(m) for m in mol_smiles]
		 psa = [rdMolDescriptors.CalcTPSA(m) for m in mol_smiles]
		 mw = [rdMolDescriptors._CalcMolWt(m) for m in mol_smiles]
		 logp = [rdMolDescriptors.CalcCrippenDescriptors(m)[0] for m in mol_smiles]
			 
		 ind_opt_mol = []
		 for indx_mol in range(len(pred_smiles_str)):             
			 if (hba[indx_mol]<10) and (hbd[indx_mol]<5) and (rtb[indx_mol]<10) and (psa[indx_mol]<140) and (mw[indx_mol]<500) and (logp[indx_mol]<5):
					  ind_opt_mol.append(indx_mol) 
				 
		 
		 emb_smiles = np.array(pred_smiles_emb)
		 emb_smiles = emb_smiles[ind_opt_mol,:]
		 norm_cls_predictions = self.predictor_model.predict(tf.convert_to_tensor(emb_smiles))
		 cls_predictions = norm_cls_predictions*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 sort_ind_pred =  np.array(cls_predictions[:,0]).argsort()

		 
		 
		 norm_cls_data_train = self.predictor_model.predict(x_train)
		 cls_data_train = norm_cls_data_train*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 sort_ind_data_train =  np.array(cls_data_train[:,0]).argsort()
		 
		 
		 if order == 'desc':
			 sort_ind_pred = sort_ind_pred[::-1]
			 sort_ind_data_train = sort_ind_data_train[::-1]             

		 
		 if len(sort_ind_pred) < max_molec_threshold:
			 max_molec_threshold = len(sort_ind_pred)
		 
		 sorted_pred_smiles_emb = np.array(pred_smiles_emb)[sort_ind_pred]
		 sorted_pred_smiles = np.array(pred_smiles_str)[sort_ind_pred]
		 
		 sorted_train_smiles = np.array(train_smiles)[sort_ind_data_train]
		 sorted_train_smiles_emb = np.array(x_train)[sort_ind_data_train] 
		 
		 new_smiles_train = np.concatenate((sorted_train_smiles[0:len(sort_ind_data_train)-max_molec_threshold],
											sorted_pred_smiles[0:max_molec_threshold]), axis = 0)
		 
		 new_x_train = np.concatenate((sorted_train_smiles_emb[0:len(sort_ind_data_train)-max_molec_threshold, :],
									  sorted_pred_smiles_emb[0:max_molec_threshold, :]), axis = 0)
		 
		 new_norm_data_cls = self.predictor_model.predict(tf.convert_to_tensor(sorted_pred_smiles_emb[0:max_molec_threshold, :]))
		 new_data_cls = new_norm_data_cls*(self.norm_q3 - self.norm_q1) + self.norm_q1 
		 new_x_train = tf.cast(new_x_train, dtype=tf.int32)

		 return new_x_train, new_smiles_train



if __name__ == '__main__':

	args = cmd_options()

	smiles_dictionary = json.load(open(args.dict))
	
	norm_parameters = json.load(open(args.norm))
	q1 = norm_parameters['q1']
	q3 = norm_parameters['q3']


	if args.train:

		f = open(args.inp_mol, "r")
		inp_molecules = f.read().splitlines()
		f.close()


		valid_smiles, _  = validity(inp_molecules)
		unique_valid_smiles = set(valid_smiles)
		tokens_mol = [smiles_str_to_tokens(x) for x in unique_valid_smiles]
		x_train, _, x_train_smiles,_  = smiles_tokens_to_idx_and_padding(tokens_mol, args.max_smiles_len, smiles_dictionary)  

		
		if len(x_train) > args.num_inp:
			x_train = x_train[0:args.num_inp]
			x_train_smiles = x_train_smiles[0:args.num_inp]

		loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


		filename = f"{args.save_path}model_{args.num_gen}gen_{args.num_add}add_{args.temperature}T_{args.optimizer}_batch{args.batch}_epochs{args.epochs}_inp{args.num_inp}"
		#filename = f"{args.save_path}model_{args.generator.split('/')[-1].split('.')[0]}_{args.predictor.split('/')[-1]}_{args.num_gen}_{args.num_add}_{args.temperature}T_batch{args.batch}_epoch{args.epochs}_{args.optimizer}_data_{args.num_inp}"
		#f"{args.save_path}model_trained.h5py"
		print(filename)
		if args.optimizer == 'RAdam':
			optimizer = tfa.optimizers.RectifiedAdam(
				 learning_rate = 1e-6,
				 beta_1 = 0.9,
				 beta_2 = 0.999,
				 weight_decay = 0.1)
		else:
			optimizer = optimizer


		transformer  = Feedback_transformers(args.predictor, args.generator,args.max_smiles_len+1, smiles_dictionary, q1, q3, q2, ccc, loss_object, optimizer, args.patience, args.min_delta)         
		transformer.train_feedback(x_train, x_train_smiles, args.epochs, args.batch, args.num_gen, args.num_add, args.temperature, args.num_inp, filename, rule5 = args.rule5)  


	if args.generate or (args.compare and args.biased_mol == ""):
		
		#Biased Generator
		biased_model = Generator_Model(None, None, None, None, args.max_smiles_len+1, len(smiles_dictionary),
									 None, None, None, None, smiles_dictionary, None, None, path_saved_model = args.saved_model)

		biased_smiles, biased_valid_smiles, perc_biased_valid = biased_model.generate_molecules(args.num_generation, '[CLS]', args.temperature)
		
		if biased_valid_smiles != []:
			unique_bias = uniqueness(biased_valid_smiles)
			div_bias = diversity(biased_valid_smiles)

		else:
			unique_bias = None
			div_bias = None

		if args.metrics:
			print('Biased Generator:')
			print(f'% valid: {perc_biased_valid}')
			print(f'% uniqueness: {unique_bias}')  
			print(f'diversity: {div_bias}')
		
		
		
		if args.rule5:
			filename = f"{args.mol_path}biased_molecules_rule5_{args.num_gen}_{args.num_add}_{args.temperature}T_from{args.num_generation}.txt"
		else:
			filename = f"{args.mol_path}biased_molecules_{args.num_gen}_{args.num_add}_{args.temperature}T_from{args.num_generation}.txt"
		
		with open(filename, "w") as outfile:
			outfile.write("\n".join(biased_smiles))


		predictor_model = tf.keras.models.load_model(args.predictor,custom_objects={'q2':q2, 'ccc':ccc})
		cls_predictions, _ = predict_property(biased_valid_smiles, predictor_model, args.max_smiles_len, smiles_dictionary, q1, q3)
		save_func(args.save_csv,(filename,args.saved_model, args.temperature, args.num_generation, perc_biased_valid,unique_bias,div_bias,np.mean(cls_predictions)),
			 ['model_path','file_mol','temperature','num_generate','%valid', '%uniqueness', 'diversity', 'Mean_pIC50'])

	
	if args.compare:

		if args.unbiased_mol == "":			
			unbiased_model = Generator_Model(None, None, None, None, args.max_smiles_len+1, len(smiles_dictionary),
										 None, None, None, None, smiles_dictionary, None, None, path_saved_model = args.generator)

			unbiased_smiles, unbiased_valid_smiles, perc_unbiased_valid = unbiased_model.generate_molecules(args.num_generation, '[CLS]', args.temperature)

			
			with open(f"{args.mol_path}unbiased_molecules{args.num_gen}_{args.num_add}_{args.temperature}T_from{args.num_generation}.txt", "w") as outfile:
				outfile.write("\n".join(unbiased_smiles))

		else:
			f = open(args.unbiased_mol, "r")
			molecules_unb = f.read().splitlines()
			f.close()

			unbiased_valid_smiles, perc_unbiased_valid = validity(molecules_unb)

		if unbiased_valid_smiles != []:
			unique_unb = uniqueness(unbiased_valid_smiles)
			div_unb = diversity(unbiased_valid_smiles)

		else:
			unique_unb = None
			div_unb = None

		if args.metrics:
			print('\nUnbiased Generator:')
			print(f'% valid: {perc_unbiased_valid}')
			print(f'% uniqueness: {unique_unb}')  
			print(f'diversity: {div_unb}')

			

			if args.biased_mol != "":
				f = open(args.biased_mol, "r")
				molecules_bias = f.read().splitlines()
				f.close()

				biased_valid_smiles, perc_biased_valid  = validity(molecules_bias)

				if biased_valid_smiles != []:
					unique_bias = uniqueness(biased_valid_smiles)
					div_bias = diversity(biased_valid_smiles)

				else:
					unique_bias = None
					div_bias = None

				print('\nBiased Generator:')
				print(f'% valid: {perc_biased_valid}')
				print(f'% uniqueness: {unique_bias}')  
				print(f'diversity: {div_bias}')

		predictor_model = tf.keras.models.load_model(args.predictor,custom_objects={'q2':q2, 'ccc':ccc})
		
		if args.rule5:
			filepath = f'{args.path_results}molecules_rule5_{args.num_gen}_{args.num_add}_{args.temperature}T_from{args.num_generation}mol'
		else:
			filepath = f'{args.path_results}molecules_{args.num_gen}_{args.num_add}_{args.temperature}T_from{args.num_generation}mol'
		
		if unique_unb == None and div_unb == None and unique_bias == None and div_bias == None:
			print('It is not possible to compare the generators.')
		else:
			compare_models(unbiased_valid_smiles, biased_valid_smiles, predictor_model, q1, q3, args.max_smiles_len, smiles_dictionary, filepath)