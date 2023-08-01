# -*- coding: utf-8 -*-
"""

"""


import pandas as pd
import numpy as np
import json
import tensorflow as tf
import tensorflow_addons as tfa
import random
import time
import argparse
import sys
from transformers import *
from utils import *


def cmd_options():
	parser = argparse.ArgumentParser(description='Program that trains and tests the Masked Learning Model')

	parser.add_argument('--train', action='store_true', help='Training')
	parser.add_argument('--test', action='store_true', help='Testing')

	parser.add_argument('--d_model', type = int,  default=512,  metavar='', help='D_model (dimensional embedding vector)')
	parser.add_argument('--dff', type = int,  default=1024,  metavar='', help='Dff (feed forward  projection size)')
	parser.add_argument('--num_heads', type = int,  default=4, metavar='', help='Number of heads')
	parser.add_argument('--num_encoders', type = int,  default=4, metavar='', help='Number of encoders')
	parser.add_argument('--rate', type = float,  default=0.1, metavar='', help='Dropout rate')
	parser.add_argument('--func', type = str,  default='relu', metavar='', help='Activation Function')
	parser.add_argument('--num_smiles', type = int,  default=1000000, metavar='', help='Number of SMILES for training or testing, needs to be larger than 0')
	parser.add_argument('--max_smiles_len', type = int,  default=72, metavar='', help='Maximum length of SMILES for training, needs to be larger than 0')
	parser.add_argument('--min_fg', type = int,  default=6, metavar='', help='Minimumm number of functional groups (FGs) where masking considers FGs')
	parser.add_argument('--batch', type = int,  default=256, metavar='', help='Batch size')
	parser.add_argument('--epochs', type = int,  default=100, metavar='', help='Number of epochs')
	parser.add_argument('--optimizer', type = str,  default='RAdam', metavar='', help='Optimizer')
	parser.add_argument('--patience', type = int,  default=20, metavar='', help='Patience in EarlyStopping')
	parser.add_argument('--min_delta', type = float,  default=0.001, metavar='', help='EarlyStopping mininum delta')
	parser.add_argument('--metrics',  type=bool, default = True, metavar='', help='True if to show the metrics of the model with the test dataset')
	parser.add_argument('--training_dataset', type = str, default="../dataset/data_pretrain_train.csv", metavar='', help='Training dataset path')
	parser.add_argument('--testing_dataset', type = str, default="../dataset/data_pretrain_test.csv", metavar='', help='Testing dataset path')
	parser.add_argument('--dict', type = str, default="../dictionary/dictionary_smiles.txt", metavar='', help='dictionary smiles path')
	parser.add_argument('--save_path', type = str, default="../models/masking_pre_train/", metavar='', help='save model path')
	parser.add_argument('--saved_model', required='--test' in sys.argv, default = None, metavar='', help='saved model path')
	parser.add_argument('--path_csv', type = str, default = "../results/masked/results.csv", metavar='', help='Path of csv to write the results of prediction')
	
	args = parser.parse_args()
	return args





class Masked_Learning_Model():
	def __init__(self, d_model, dff, num_heads, num_encoders,smiles_len,smiles_vocab_size, act_func, rate, loss_object, optimizer, smiles_dictionary, patience, min_delta, path_saved_model = None):        
		if path_saved_model == None:
			self.model = Masked_Smiles_Model(d_model, dff, num_heads, num_encoders, smiles_len,smiles_vocab_size, act_func,rate)
		else: 
			self.model = tf.keras.models.load_model(path_saved_model)
		
		self.loss_object = loss_object
		self.optimizer = optimizer
		self.patience = patience
		self.min_delta = min_delta
		self.smiles_dictionary = smiles_dictionary

		
	def loss_function(self, y_true,y_pred, mask):  
		loss_ = self.loss_object(y_true, y_pred)
		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask
		return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

	
	def accuracy_function(self, y_true,y_pred, mask):
		accuracies = tf.equal(y_true, tf.cast(tf.argmax(y_pred, axis=2), dtype=tf.int32))
		mask_b = tf.constant(mask > 0)
		accuracies = tf.math.logical_and(mask_b, accuracies)
		accuracies = tf.cast(accuracies, dtype=tf.float32)
		mask = tf.cast(mask, dtype=tf.float32)
		return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
	 
	
	def train_step(self, x, target, masked_positions):
	
		with tf.GradientTape() as tape:
			_, predictions, _, _ = self.model(x, training = True)
			loss = self.loss_function(target, predictions, masked_positions)
		
		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))       
		acc = self.accuracy_function(target, predictions, masked_positions)
		return loss, acc
	
   
	def train_model(self, inp_smiles_emb, inp_fgs, epochs, batch_size, threshold_min, filename):
		print('Training...')
		inp_smiles, masked_positions = self.masking(inp_smiles_emb, inp_fgs, threshold_min)   
		inp = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(inp_smiles), tf.convert_to_tensor(masked_positions))).batch(batch_size, drop_remainder = False)
		label = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(inp_smiles_emb)).batch(batch_size, drop_remainder = False)
		data = tf.data.Dataset.zip((inp, label))


		last_loss = {'epoch':0,'value':1000}

		for epoch in range(epochs):            
			print(f'Epoch {epoch+1}/{epochs}')
			
			start = time.time()           
			
			loss_epoch = []
			acc_epoch = []
			
			for num, ((x_train, fgs_train), y_train) in enumerate(data):
				loss_batch, acc_batch = self.train_step(x_train, y_train, fgs_train)
				loss_epoch.append(loss_batch)
				acc_epoch.append(acc_batch)
				
				if num == len(data)-1:
					print(f'{num+1}/{len(data)} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f} - accuracy: {np.mean(acc_epoch):.4f}')    
					
			if (last_loss['value'] - np.mean(loss_epoch)) >= self.min_delta: 
				last_loss['value'] = np.mean(loss_epoch)
				last_loss['epoch'] = epoch+1
				print('Saving model...')
				self.model.save(filename) #save_path+'trained_model.h5py')
				   
			if ((epoch+1) - last_loss['epoch']) >= self.patience:
				break 
	

	def predict_model(self, inp_smiles_emb, inp_fgs, threshold_min,batch_size):

		inp_smiles, masked_positions = self.masking(inp_smiles_emb, inp_fgs, threshold_min) 
		_, pred, _, _ = self.model(inp_smiles)    
		
		print('Predicting...')
		_, pred, _, _ = self.model.predict(inp_smiles) 
		loss_ = self.loss_function(inp_smiles_emb,pred, masked_positions)
		acc_ = self.accuracy_function(inp_smiles_emb,pred, np.array(masked_positions))
		return loss_, acc_


	def masking(self, inp, fgs, threshold_min):
		masked_positions = []
		
		x = [i.copy() for i in inp]
		
		for smile_index in range(len(x)):
			fg = fgs[smile_index]

			not_fg = [indx for indx in range(len(x[smile_index])) if (indx not in fg) and (x[smile_index][indx] not in  [self.smiles_dictionary['[PAD]'],
			self.smiles_dictionary['[CLS]'], self.smiles_dictionary['[SEP]'], self.smiles_dictionary['[MASK]'] ])]
			
			p_fg = 0.075
			p_not_fg = 0.075
			
			if len(fg) < threshold_min:
				p_not_fg = 0.15
			
			num_mask_fg = max(1, int(round(len(fg) * p_fg)))
			num_mask_not_fg = max(1, int(round(len(not_fg) * p_not_fg)))
			shuffle_fg = random.sample(range(len(fg)), len(fg))
			shuffle_not_fg = random.sample(range(len(not_fg)), len(not_fg))
			
			fg_temp = [fg[n] for n in shuffle_fg[:num_mask_fg]]
			not_fg_temp = [not_fg[n] for n in shuffle_not_fg[:num_mask_not_fg]] 
			
			mask_index = fg_temp + not_fg_temp#fg[shuffle_fg[:num_mask_fg]]+ shuffle_not_fg[:num_mask_not_fg]
			masked_pos =[0]*len(x[smile_index])
			
			for pos in mask_index:
				masked_pos[pos] = 1
				if random.random() < 0.8: 
					x[smile_index][pos] = self.smiles_dictionary['[MASK]']
				elif random.random() < 0.15: 
					index = random.randint(1, self.smiles_dictionary['[CLS]']-1) 
					x[smile_index][pos] = self.smiles_dictionary[list(self.smiles_dictionary.keys())[index]]
			masked_positions.append(masked_pos) 
		 
		return x, masked_positions
			



if __name__ == '__main__':

	args = cmd_options()

	smiles_dictionary = json.load(open(args.dict))
	
	if args.train:

		df = pd.read_csv(args.training_dataset)  
		df = df.drop_duplicates()   

		if args.num_smiles < len(df):
			df = df.iloc[0:args.num_smiles]

		smiles = df['smiles'].tolist()
		fgs = df['fgs'].apply(eval).tolist()

		tokens_mol = [smiles_str_to_tokens(x) for x in smiles]
		inp_smiles_emb, _,_, inp_ind  = smiles_tokens_to_idx_and_padding(tokens_mol, args.max_smiles_len, smiles_dictionary)  
		
		fgs = np.array(fgs)
		inp_fgs = list(fgs[inp_ind])

		filename = f"{args.save_path}model_{args.d_model}_{args.dff}_{args.num_heads}_{args.num_encoders}_{args.func}_batch{args.batch}_epoch{args.epochs}_{args.optimizer}_data_{fgs.shape[0]}.h5py"
		#filename = f"{args.save_path}model_trained.h5py"


		if args.optimizer == 'RAdam':
			optimizer = tfa.optimizers.RectifiedAdam(
			 learning_rate=1e-3,
			 beta_1 = 0.9,
			 beta_2 = 0.999,
			 weight_decay = 0.1)
		else:
			optimizer = args.optimizer


		loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

		model = Masked_Learning_Model(args.d_model, args.dff, args.num_heads, args.num_encoders, args.max_smiles_len+2, len(smiles_dictionary),
									 args.func, args.rate, loss_object, optimizer, smiles_dictionary, args.patience, args.min_delta, path_saved_model = args.saved_model)

		model.train_model(inp_smiles_emb, inp_fgs, args.epochs, args.batch, args.min_fg, filename)


	if args.test:

		df = pd.read_csv(args.training_dataset)  
		df = df.drop_duplicates()   

		if args.num_smiles < len(df):
			df = df.iloc[0:args.num_smiles]

		smiles = df['smiles'].tolist()
		fgs = df['fgs'].apply(eval).tolist()

		tokens_mol = [smiles_str_to_tokens(x) for x in smiles]
		inp_smiles_emb, _,_, inp_ind  = smiles_tokens_to_idx_and_padding(tokens_mol, args.max_smiles_len, smiles_dictionary)  
		
		fgs = np.array(fgs)
		inp_fgs = list(fgs[inp_ind])

		loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

		model = Masked_Learning_Model(None, None, None, None, args.max_smiles_len+2, len(smiles_dictionary),
									 None, None, loss_object, None, smiles_dictionary, None,None, path_saved_model = args.saved_model)

		loss, acc = model.predict_model(inp_smiles_emb, inp_fgs, args.min_fg, args.batch)
		save_func(args.path_csv,(args.saved_model, loss.numpy(), acc.numpy()), ['model_path','loss','acc'])
		
		if args.metrics:
			print(f'Metrics:\nLoss: {loss}\nAcc: {acc}')