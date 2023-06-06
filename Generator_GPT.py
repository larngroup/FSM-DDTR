# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:05:26 2022

@author: Ana Catarina
"""


import pandas as pd
import json

from transformers import Transformer_Decoder
#from prepare_data import *
from predict_mol import * #generateTransformer
import tensorflow as tf
import tensorflow_addons as tfa
import random
import time
#from tqdm import tqdm
#from progress.bar import Bar
#from keras_radam import RAdam

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
    
    
def create_batch(smiles_inp, smiles_out, batch_size):
    rand = random.sample(range(len(smiles_inp)), len(smiles_inp))
        
    batched_smiles_x = []
    batched_smiles_y = []
        
    for i in range(0, len(rand), batch_size):
            
        batched_smiles_x.append([smiles_inp[ii] for ii in rand[i:i+batch_size]])
        batched_smiles_y.append([smiles_out[ii] for ii in rand[i:i+batch_size]])
            
    return batched_smiles_x, batched_smiles_y



def loss_function(y_true,y_pred):  
  loss_ = loss_object(y_true, y_pred)
  mask = tf.math.logical_not(tf.math.equal(y_true, 0)) # true para elementos diferentes de 0
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  #print(f'loss {loss_}')
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)




def accuracy_function(y_true,y_pred):
  mask = tf.math.logical_not(tf.math.equal(y_true, 0)) # true para elementos diferentes de 0
  mask = tf.cast(mask, dtype=tf.float32)
  pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.int32)
  y_true = tf.cast(y_true, dtype=tf.int32)
  #print(f'output_real {output_real.shape} and argmax_indices {argmax_indices.shape}')
  accuracy = tf.equal(y_true,pred)
  #print('b')
  acc = tf.cast((accuracy), dtype=tf.float32)
  acc *= mask
  return tf.reduce_sum(acc)/tf.reduce_sum(mask)



data_path={'data':'../dataset/data_for_generation_notduplicate.csv', #data_generation_GPT_Tiago.csv',
            'smiles_dic':'../dictionary/dictionary_smiles.txt',
            'save_model': '../model/generation/gpt/',
            'checkpoint_path': '../model/generation/checkpoint_GPT'} #só 2 funciona bem
    

parameters = {'D_model': 512, 'Dff': 1024, 'Num_heads': 8, 'Num_decoders': 4, 'Act_func':'relu','Rate': 0.1, 'num_smiles': 1000000, 'batch_size': 256, 'epochs':100, 
			'optimizer':'RAdam', 'EarlyStopping_min_delta':0.001, 'EarlyStopping_patience':10}
        
smiles_dictionary = json.load(open(data_path['smiles_dic']))


train = False
generate = True

if train:
    
    df= pd.read_csv(data_path['data'])
    #df = df.iloc[0:parameters['num_smiles']]
    
    smiles_emb = listStr_to_listInt(df['smiles_emb'])
	
    #tokens_mol = [token(x) for x in df['smiles'].tolist()]
    #smiles_emb, smiles_mol = embedding_token_id_and_padding(tokens_mol, 72, smiles_dictionary)   
    smiles = tf.convert_to_tensor(smiles_emb)
    smiles_x = smiles[:,:-1]
    smiles_y = smiles[:,1:]

    #print(smiles_x[0])
    #print(smiles_y[0])
    
    #model = tf.keras.models.load_model(f'../model/generation/train_without_compile_8_4_relu_RAdam_batch_256_data_1046964.h5py')
    
    model = Transformer_Decoder(parameters['D_model'], parameters['Dff'], parameters['Num_heads'], parameters['Num_decoders'],73,len(smiles_dictionary), parameters['Act_func'], rate = 0.1)
    #model_path = f'{data_path["save_model"]}_train_{parameters["Num_heads"]}_{parameters["Num_decoders"]}_{parameters["Act_func"]}_{parameters["optimizer"]}_num_smiles_{parameters["num_smiles"]}_batch_{parameters["batch_size"]}.h5py'

    model_path = f'../model/generation/train_without_compile_{parameters["Num_heads"]}_{parameters["Num_decoders"]}_{parameters["Act_func"]}_{parameters["optimizer"]}_batch_{parameters["batch_size"]}_data_{smiles.shape[0]}.h5py'

    check_path = f'{data_path["checkpoint_path"]}_train_{parameters["Num_heads"]}_{parameters["Num_decoders"]}_{parameters["Act_func"]}_{parameters["optimizer"]}_batch_{parameters["batch_size"]}'
    
    optimizer = tfa.optimizers.RectifiedAdam(
     learning_rate = 1e-3,
     beta_1 = 0.9,
     beta_2 = 0.999,
     weight_decay = 0.1)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
	    from_logits=True, reduction='none')
     

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    
    def train_step(x,target):
        with tf.GradientTape() as tape:
    	    predictions, _ = model(x, training = True)
    	    #print(f'pred {predictions}')
    	    loss = loss_function(target, predictions)
    	
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        acc = accuracy_function(target, predictions)
        train_accuracy(acc)
        #return loss, acc

    
   # ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    
    #ckpt_manager = tf.train.CheckpointManager(ckpt, check_path, max_to_keep=1)


	# if a checkpoint exists, restore the latest checkpoint.
    #if ckpt_manager.latest_checkpoint:
        #ckpt.restore(ckpt_manager.latest_checkpoint)
        #print('\nLatest checkpoint restored!!')
	


    print('###############################################')
    print('>>>>>>>>>>>>> Training the Model <<<<<<<<<<<<<<')    
    print('###############################################')


    last_loss = {'epoch':0,'value':1000}
    
    for epoch in range(parameters['epochs']):
        print(f'Epoch {epoch+1}/{parameters["epochs"]}')
		
        start = time.time()
        
        train_loss.reset_states()
        train_accuracy.reset_states()
	   
        batched_smiles_x, batched_smiles_y = create_batch(smiles_x, smiles_y, parameters['batch_size'])

        #total_loss = 0
        #total_acc = 0
	    
       # with Bar(f'Epoch {epoch+1}/{parameters["epochs"]}',max=len(batched_smiles_x)) as bar:
        for  (batch,(inp_smiles,out_smiles)) in enumerate(zip(batched_smiles_x, batched_smiles_y)):
        #for  (batch,(inp_smiles,out_smiles)) in enumerate(tqdm(zip(batched_smiles_x, batched_smiles_y),total = len(batched_smiles_x), position=0, leave=True)):
               inp_smiles = tf.convert_to_tensor(inp_smiles)#tf.concat(inp_smiles, 0)
               out_smiles = tf.convert_to_tensor(out_smiles) #tf.concat(out_smiles, 0)
               #print(inp_smiles[0],out_smiles[0])
               train_step(inp_smiles, out_smiles)
               #loss, acc = train_step(inp_smiles,out_smiles)
    	       #print(loss, acc)
               #total_loss += loss
               #total_acc += acc    
               #bar.next()
               #pass
    
           
        if (last_loss['value'] - train_loss.result() >= parameters['EarlyStopping_min_delta']):
	     #print(f'last {last_loss} and {train_loss.result()}')
	        last_loss['value'] = train_loss.result()
	        last_loss['epoch'] = epoch+1
	        #ckpt_save_path = ckpt_manager.save()
	        model.save(model_path)
	        print(f'Saving checkpoint for epoch {epoch+1}') #at {ckpt_save_path}')

        
        print(f'{batch+1}/{len(batched_smiles_x)} - {round(time.time() - start)}s - loss: {train_loss.result():.4f} - accuracy: {train_accuracy.result():.4f}')    
	    
        if (epoch - last_loss['epoch']) > parameters['EarlyStopping_patience']:
            break 

	 

if generate:
    #model_path = f'../model/train_{parameters["Num_heads"]}_{parameters["Num_decoders"]}_{parameters["Act_func"]}_{parameters["optimizer"]}_num_smiles_{parameters["num_smiles"]}_batch_{parameters["batch_size"]}.h5py'

    check_path = f'{data_path["checkpoint_path"]}_train_{parameters["Num_heads"]}_{parameters["Num_decoders"]}_{parameters["Act_func"]}_{parameters["optimizer"]}_batch_{parameters["batch_size"]}'
    
    
    #model_trained = Transformer_Decoder(parameters['D_model'], parameters['Dff'], parameters['Num_heads'], parameters['Num_decoders'],73,len(smiles_dictionary), parameters['Act_func'], rate = 0.1)
    
    
    optimizer = tfa.optimizers.RectifiedAdam(
	     learning_rate=1e-3)#,
	     #total_steps=10000,
	     #warmup_proportion=0.1,
	     #min_lr=1e-5,
	 #)
    
    
    #ckpt = tf.train.Checkpoint(model=model_trained, optimizer=optimizer)
    
    #ckpt_manager = tf.train.CheckpointManager(ckpt, check_path, max_to_keep=1)


	# if a checkpoint exists, restore the latest checkpoint.
    #if ckpt_manager.latest_checkpoint:
        #ckpt.restore(ckpt_manager.latest_checkpoint)
        #print('\nLatest checkpoint restored!!')
        
    #transformer = generateTransformer(model_trained, '[CLS]', '[SEP]', smiles_dictionary, top_k = 2)
    
    #smiles, perc_valid = transformer(4, 73)
    
    #transformer = TextGenerator(model_trained, 73, [32],smiles_dictionary, top_k=10, print_every=1) #generateTransformer(model_trained, '[CLS]', '[SEP]', smiles_dictionary)
    
    #smiles_list, valid_smiles, perc_valid, perc_unique_smiles, td = transformer(10, 2)
    model_path = f'../model/train_without_compile_{parameters["Num_heads"]}_{parameters["Num_decoders"]}_{parameters["Act_func"]}_{parameters["optimizer"]}_num_smiles_{parameters["num_smiles"]}_batch_{parameters["batch_size"]}.h5py'
    model_trained = tf.keras.models.load_model('../model/generation/train_without_compile_8_4_relu_RAdam_batch_256_data_1046964.h5py')
    """
    transformer = SMILES_Generator_(model_trained, 73, '[CLS]','[SEP]', smiles_dictionary, top_k=10) #generateTransformer(model_trained, '[CLS]', '[SEP]', smiles_dictionary)
    
    smiles_list, valid_smiles, perc_valid = transformer(500)
    
    print(f'valid smiles {valid_smiles}')
    print(f'% valid: {perc_valid}')
    
    def uniqueness(smiles_list):
        unique_smiles=list(set(smiles_list))
        return (len(unique_smiles)/len(smiles_list))*100
    
    print(f'% uniqueness {uniqueness(valid_smiles)}')  
   # print(f'Diversity {td}')
    with open("../molecules/molecules_without_feedback_500.txt", "w") as outfile:
     outfile.write("\n".join(smiles_list))
    
   """
   
   
   
   
   
   
   
   #################################################
    #model_trained = Transformer_Decoder(parameters['D_model'], parameters['Dff'], parameters['Num_heads'], parameters['Num_decoders'],73,len(smiles_dictionary), parameters['Act_func'], rate = 0.1)
    #model_trained.load_weights('../model/feedback/path_10epochs_32batch_100molec_70new.h5') 
    temp = 0.9
    num_mol = 500
    model_trained = tf.keras.models.load_model('../model/feedback/model_kop_60epochs_128batch_1200molec_450new_0.9T_train8500molec_pIC50_SAS_Mw_TPSA_LogP.h5py') #('../model/generation/train_without_compile_8_4_relu_RAdam_batch_256_data_1046964.h5py')
    start = time.time()	
    transformer = Generator_(model_trained, 73, '[CLS]','[SEP]', smiles_dictionary, sample_temperature=temp) #generateTransformer(model_trained, '[CLS]', '[SEP]', smiles_dictionary)
    
    smiles_list, valid_smiles, perc_valid = transformer(num_mol)
    print(f'{round(time.time() - start)}s - %valid: {perc_valid}')
    #print(f'valid smiles {valid_smiles}')
    #print(f'% valid: {perc_valid}')
    
    def uniqueness(smiles_list):
        unique_smiles=list(set(smiles_list))
        return (len(unique_smiles)/len(smiles_list))*100
    print(f'% uniqueness {uniqueness(valid_smiles)}')  
   # print(f'Diversity {td}')
    with open(f"../molecules/molecules_kop_{temp}T_{num_mol}_1200_450_pIC50_SAS_Mw_TPSA_LogP.txt", "w") as outfile:
     outfile.write("\n".join(smiles_list))
   
