# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 00:12:46 2022

@author: Ana Catarina
"""

#D:\Universidade\5º Ano\Tese\SMILES_BERT\src


import pandas as pd
import json

#from transformer_encoder_b import Masked_Smiles_Model
from transformers import *
import tensorflow as tf
import tensorflow_addons as tfa
import random
import time
#from keras_radam import RAdam

data_path={'data':'../dataset/data_pretrain.csv',
            'smiles_dic':'../dictionary/dictionary_smiles.txt',
            'save_model': '../model/Pre_train/smiles_bert'}
    

#smiles_dictionary = json.load(open('../dictionary/dictionary_smiles.txt'))#('../dictionary/dictionary_smiles_kaggle.txt'))#

#df = pd.read_csv('../dataset/data.csv')#('../dataset/smiles_kaggle.csv')#('../dataset/data.csv')  


#df = df.head(1000)

parameters = {'D_model': 512, 'Dff': 1024, 'Num_heads': 4, 'Num_encoder': 4, 'Rate': 0.1, 'num_smiles': 1000000, 'batch_size': 256, 'epochs':100, 'optimizer':'RAdam', 'patience':20}
   
checkpoint_path = f'{data_path["save_model"]}_train_{parameters["Num_heads"]}_{parameters["Num_encoder"]}_{parameters["optimizer"]}_num_smiles_{parameters["num_smiles"]}_batch_{parameters["batch_size"]}'
     

smiles_dictionary = json.load(open(data_path['smiles_dic']))

df= pd.read_csv(data_path['data'])  
df = df.drop_duplicates()   
df = df.iloc[0:parameters['num_smiles']]
print(df.head(10))   
model = Masked_Smiles_Model(parameters['D_model'], parameters['Dff'], parameters['Num_heads'], parameters['Num_encoder'],74,len(smiles_dictionary), 'relu', parameters['Rate'])
#model = tf.keras.models.load_model('../model/pretrain_BERT/model_pretrained.h5py')

"""
def create_batch(fgs, smiles, batch_size):
    rand = random.sample(range(len(smiles)), len(smiles))
    
    batched_smiles = []
    batched_fgs = []
    
    for i in range(0, len(rand), batch_size):
        
        batched_smiles.append([smiles[ii] for ii in rand[i:i+batch_size]])
        batched_fgs.append([fgs[ii] for ii in rand[i:i+batch_size]])
        #print(batched_fgs)
        
    return batched_smiles, batched_fgs
"""

def create_batch(fgs, smiles_inp, smiles_out, batch_size):
        rand = random.sample(range(len(smiles_inp)), len(smiles_inp))
        
        batched_smiles_x = []
        batched_fgs = []
        batched_smiles_y = []
        
        for i in range(0, len(rand), batch_size):
            
            batched_smiles_x.append([smiles_inp[ii] for ii in rand[i:i+batch_size]])
            batched_fgs.append([fgs[ii] for ii in rand[i:i+batch_size]])
            batched_smiles_y.append([smiles_out[ii] for ii in rand[i:i+batch_size]])
            
        return batched_smiles_x, batched_fgs, batched_smiles_y

    

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


BATCH_SIZE = parameters['batch_size']

def masking(inp, fgs, dict_mol, threshold_min):
    #print(f' len {len(x)}')
    masked_positions = []
    
    x = [i.copy() for i in inp]
    
    for smile_index in range(len(x)):
        fg = fgs[smile_index]
        #print(f'fg {fg}')
        not_fg = [indx for indx in range(len(x[smile_index])) if (indx not in fg) and (x[smile_index][indx] not in [0, 32,33,34])]
        #print(f'not_fg {not_fg}')
        
        p_fg = 0.075
        p_not_fg = 0.075
        
        if len(fg) < threshold_min:
            p_not_fg = 0.15
        
        num_mask_fg = max(1, int(round(len(fg) * p_fg)))
        num_mask_not_fg = max(1, int(round(len(not_fg) * p_not_fg)))
        #print(f' num_mask_fg {num_mask_fg} and num_mask_not_fg {num_mask_not_fg} ')
        shuffle_fg = random.sample(range(len(fg)), len(fg))
        shuffle_not_fg = random.sample(range(len(not_fg)), len(not_fg))
        
        fg_temp = [fg[n] for n in shuffle_fg[:num_mask_fg]]
        not_fg_temp = [not_fg[n] for n in shuffle_not_fg[:num_mask_not_fg]] 
        
        mask_index = fg_temp + not_fg_temp#fg[shuffle_fg[:num_mask_fg]]+ shuffle_not_fg[:num_mask_not_fg]
        #print(f'mask_index {mask_index}')
        masked_pos =[0]*len(x[smile_index])
        
        for pos in mask_index:
            masked_pos[pos] = 1
            if random.random() < 0.8: 
                x[smile_index][pos] = dict_mol['[MASK]'] 
                #print('a')
            elif random.random() < 0.15: 
                index = random.randint(1, 31) 
                x[smile_index][pos] = dict_mol[list(dict_mol.keys())[index]]
                #print('b')
            #else:
                #print('c')
        masked_positions.append(masked_pos) 
     
    return x, masked_positions
            

"""
def masking(inp, fgs, dict_mol):
    #print(f' len {len(x)}')
    masked_positions = []
    masked_fgs_indx = []
    
    x = [i.copy() for i in inp]

    
    for batch in range (len(x)):
      fg = fgs[batch]
      n_pred = max(1, int(round(len(fg) * 0.3)))
      shuffle_list = random.sample(range(len(fg)), len(fg))
      
      masked_pos =[0]*len(x[batch])
      #print(f' m {masked_pos}')
      masked_fg_index = []
      for pos in shuffle_list[:n_pred]:
         masked_pos[pos] = 1
         masked_fg_index.append(fg[pos])
         if random.random() < 0.85: 
             x[batch][fg[pos]] = dict_mol['[MASK]'] 
         elif random.random() < 0.1: 
             index = random.randint(1, 31) 
             x[batch][fg[pos]] = dict_mol[list(dict_mol.keys())[index]]
      masked_positions.append(masked_pos)
      masked_fgs_indx.append(masked_fg_index)
      #print(f' ms {masked_positions}')
  
    for batch in range (len(x)):
      fg = fgs[batch]
      masked_pos =[0]*len(x[batch])
      
      for pos in fg:
         masked_pos[pos] = 1
         if random.random() < 0.85: 
             x[batch][pos] = dict_mol['[MASK]'] 
         elif random.random() < 0.1: 
             index = random.randint(1, 31) 
             x[batch][pos] = dict_mol[list(dict_mol.keys())[index]]
      masked_positions.append(masked_pos) 
      masked_fgs_indx.append(fg )
      
    #print(f'input {inp}')
    #print(f'x {x}')
    
    return x, masked_fgs_indx, masked_positions  
    
    """

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule): #########Alterar
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


#learning_rate = CustomSchedule(512)

#optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                     epsilon=1e-9) 

#optimizer = tfa.optimizers.RectifiedAdam()#RAdam() 
optimizer = tfa.optimizers.RectifiedAdam(
     learning_rate=1e-3,
     beta_1 = 0.9,
     beta_2 = 0.999,
     weight_decay = 0.1)
     

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none')
     
def loss_function(real, pred, mask):
  #print(f'mask {mask}')
  #print(f'pred {pred}')
  #print(f'real {real}')
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  #print(f'loss {loss_}')
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred, mask):
  #print(f'mask {mask}')
  #print(f'pred {tf.cast(tf.argmax(pred, axis=2), dtype=tf.int32)}')
  #print(f'real {real}')
  accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2), dtype=tf.int32))
  #print(f'acc {accuracies}')
  mask_b = tf.constant(mask > 0)
  accuracies = tf.math.logical_and(mask_b, accuracies)
  #print(accuracies)
  accuracies = tf.cast(accuracies, dtype=tf.float32)
  #print(f'accuracies {accuracies}')
  mask = tf.cast(mask, dtype=tf.float32)
  #print(f'mask {mask.shape}')
  #print(tf.reduce_sum(accuracies)/tf.reduce_sum(mask))
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    


"""
def train_step(inp_smiles,inp_fgs):

    with tf.GradientTape() as tape:
      predictions = model(inp_smiles,inp_fgs)
      loss = lossFunction(inp_smiles, predictions)
    
      #print(f'tar_real{tar_real.shape} and predictions{predictions.shape}')
      #print(np.argmax(predictions, axis=2).shape)
      acc = accuracyFunction(inp_smiles, predictions)
      #print(acc)
    
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
      #  train_loss(loss)
       # train_accuracy(acc)
    return loss, acc

"""


def train_step(x, masked_positions, target):
  #print(f'inp {inp_smiles}') # está bem
  #x,_ , masked_positions = masking(inp_smiles, inp_fgs, dict_mol) 
  #print(f'x {x.shape}') #mascara
  #print(f'inp {inp_smiles}')
  #x = tf.convert_to_tensor(x)
  #print(f'mask {masked_positions.shape}')
  #masked_positions = tf.convert_to_tensor(masked_positions)
  #target = tf.convert_to_tensor(inp_smiles)
  #print(f'target {target.shape}') #mascara
  #print(f' x {x.shape} and masked {masked_positions.shape}')
  with tf.GradientTape() as tape:
      
    transformer_output, predictions,_,_ = model(x)
    #print(f'pred {predictions}')
    loss = loss_function(target, predictions, masked_positions)
    #print(f'loss {loss}')

  gradients = tape.gradient(loss, model.trainable_variables)
  #print('gradients')
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  #print('opt')
  train_loss(loss)
  train_accuracy(accuracy_function(target, predictions, masked_positions)) ######


#checkpoint_path = "./checkpoints/train_smiles"#"_kaggle"

ckpt = tf.train.Checkpoint(model=model,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)


#if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')
 
  
fgs = listStr_to_listInt(df['fgs'].to_list()) 
smiles = listStr_to_listInt(df['smiles_emb'].to_list())#['emb_smiles'])#'SMILES'])
        
#print(fgs)
x, masked_positions = masking(smiles, fgs, smiles_dictionary, 6)    

smiles_x = tf.convert_to_tensor(x)
smiles_y = tf.convert_to_tensor(smiles) 
mask_fg = tf.convert_to_tensor(masked_positions)  

print('###############################################')
print('>>>>>>>>>>>>>>> Train the Model <<<<<<<<<<<<<<<')    
print('###############################################')

NUM_EPOCHS = parameters['epochs']

"""
for epoch in range(NUM_EPOCHS):
   print(f'{epoch+1}/{NUM_EPOCHS}')
   start = time.time()

   train_loss.reset_states()
   train_accuracy.reset_states()
   
   batched_smiles_x, batched_fgs, batched_smiles_y = create_batch(mask_fg, smiles_x, smiles_y, BATCH_SIZE)
  
   cont = 0     
   for  (batch,(inp_smiles,inp_fgs,out_smiles)) in enumerate(zip(batched_smiles_x, batched_fgs, batched_smiles_y)):
       inp_smiles = tf.convert_to_tensor(inp_smiles)#tf.concat(inp_smiles, 0)
       inp_fgs = tf.convert_to_tensor(inp_fgs)#tf.concat(inp_fgs, 0)
       out_smiles = tf.convert_to_tensor(out_smiles) #tf.concat(out_smiles, 0)
       #print(inp_smiles,inp_fgs,out_smiles)
       
       train_step(inp_smiles,inp_fgs,out_smiles)
       cont += 1
           
   print(f'{cont}/{len(batched_smiles_x)} - {round(time.time() - start)}s - loss: {train_loss.result():.4f} - accuracy: {train_accuracy.result():.4f}')    

   tf.saved_model.save(model, checkpoint_path)    
              
   #batched_smiles, batched_fgs = create_batch(listStr_to_listInt(df['FGs']), listStr_to_listInt(df['SMILES']), BATCH_SIZE)     
   #for  (batch,(inp_smiles,inp_fgs)) in enumerate(zip(batched_smiles, batched_fgs)):
      #print(f'input {len(inp_smiles)}, fgs {len(inp_fgs)}')
      
      #x, _ , masked_positions = masking(inp_smiles, inp_fgs, smiles_dictionary)
      #print(f'x {x}') #mascara
      #print(f'inp {inp_smiles}')
     # x = tf.convert_to_tensor(x)
      #masked_positions = tf.convert_to_tensor(masked_positions)
      #target = tf.convert_to_tensor(inp_smiles)
  
      #predictions = train_step(x, masked_positions, target)#[i][np.newaxis], output_seq[i][np.newaxis])
      #break
       #if ((batch+1) % 100 == 0):
        #print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
   #if (epoch + 1) % 5 == 0:
     #ckpt_save_path = ckpt_manager.save()
    # print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

   #print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

#model.load_weights(checkpoint_path)
"""

last_loss = {'epoch':0,'value':1000}

for epoch in range(NUM_EPOCHS):
   print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
	
   start = time.time()

   train_loss.reset_states()
   train_accuracy.reset_states()
   


   batched_smiles_x, batched_fgs, batched_smiles_y = create_batch(mask_fg, smiles_x, smiles_y, BATCH_SIZE)
  
   cont_batch = 0     
   for  (batch,(inp_smiles,inp_fgs,out_smiles)) in enumerate(zip(batched_smiles_x, batched_fgs, batched_smiles_y)):
       inp_smiles = tf.convert_to_tensor(inp_smiles)#tf.concat(inp_smiles, 0)
       inp_fgs = tf.convert_to_tensor(inp_fgs)#tf.concat(inp_fgs, 0)
       out_smiles = tf.convert_to_tensor(out_smiles) #tf.concat(out_smiles, 0)
       #print(inp_smiles,inp_fgs,out_smiles)
       
       train_step(inp_smiles,inp_fgs,out_smiles)
       cont_batch += 1
           
   
   
   if (last_loss['value'] > train_loss.result()):
     #print(f'last {last_loss} and {train_loss.result()}')
     last_loss['value'] = train_loss.result()
     last_loss['epoch'] = epoch+1
     #print(f'last {last_loss}')
     #ckpt_save_path = ckpt_manager.save()
     #rint(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
     #tf.saved_model.save(model, '../checkpoints/train/model_cls_2')   
     model.save(f'../model/pretrain_BERT/model__{parameters["Num_heads"]}heads_{parameters["Num_encoder"]}encoders_batch_size{parameters["batch_size"]}_{parameters["optimizer"]}.h5py')#('../model/pretrain_BERT/model_pretrained_8heads_4encoders.h5py')
     #model.save_weights('../model/pretrain_BERT/model')
   print(f'{cont_batch}/{len(batched_smiles_x)} - {round(time.time() - start)}s - loss: {train_loss.result():.4f} - accuracy: {train_accuracy.result():.4f}')    
    
   if ((epoch+1) -last_loss['epoch']) >= parameters['patience']:
    	break 
              
   #batched_smiles, batched_fgs = create_batch(listStr_to_listInt(df['FGs']), listStr_to_listInt(df['SMILES']), BATCH_SIZE)     
   #for  (batch,(inp_smiles,inp_fgs)) in enumerate(zip(batched_smiles, batched_fgs)):
      #print(f'input {len(inp_smiles)}, fgs {len(inp_fgs)}')
      
      #x, _ , masked_positions = masking(inp_smiles, inp_fgs, smiles_dictionary)
      #print(f'x {x}') #mascara
      #print(f'inp {inp_smiles}')
     # x = tf.convert_to_tensor(x)
      #masked_positions = tf.convert_to_tensor(masked_positions)
      #target = tf.convert_to_tensor(inp_smiles)
  
      #predictions = train_step(x, masked_positions, target)#[i][np.newaxis], output_seq[i][np.newaxis])
      #break
       #if ((batch+1) % 100 == 0):
        #print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
   #if (epoch + 1) % 5 == 0:
     #ckpt_save_path = ckpt_manager.save()
    # print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

   #print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

