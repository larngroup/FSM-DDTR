# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:43:08 2022

@author: Ana Catarina
"""

import pandas as pd
import json
import numpy as np
from transformers import Transformer_Decoder

import tensorflow as tf
import tensorflow_addons as tfa
import random
import time
from progress.bar import Bar
from rdkit.Chem import MolFromSmiles, AllChem
from rdkit import DataStructs
"""
class generateTransformer():
  def __init__(self, model, start_token, end_token, smiles_dict, sampling_temp = 1.9, top_k = 3):
    self.model = model
    self.start_str_token = start_token
    self.end_str_token = end_token
    self.sampling_temp = sampling_temp
    self.smiles_dict = smiles_dict
    self.k = top_k

  def __call__(self, num_to_generate, max_smiles_len = 74): 
    print('>>>>>>>>>>>>>>> Generation <<<<<<<<<<<<<<<')   
    
    start_token_embedding = self.smiles_dict[self.start_str_token]
    end_token_embedding = self.smiles_dict[self.end_str_token]
    
    molecules = []
    for num in range(num_to_generate):
        print(f'{num+1}/{num_to_generate}')        
        
        output = tf.TensorArray(tf.int64, size=0, dynamic_size=True, clear_after_read=False)
        output = output.write(0, start_token_embedding)
        
        for i in range(max_smiles_len):
            #print(i)
            #print(output.stack().numpy())
            #print(output.stack().numpy()[:,tf.newaxis])
            
            #print(f'input:{input_seq.shape} e output: {output.stack().shape}')
            pred, _ = self.model(output.stack()[:,tf.newaxis], training = False)
            #out = tf.math.argmax(pred[-1,-1:,:],-1)
            #print(output.stack())
            
            #output = output.write(i+1, pred[0])       
            
            ##logits = pred[-1][-1:] / self.sampling_temp
            ##probs = tf.nn.softmax(logits, axis=-1)
            #print(probs)
        # sample from the distribution or take the most likely
            
            ##out=tf.argmax(np.random.multinomial(1,probs[0], 1),-1)#tf.argmax(tf.random.categorical(probs,1),-1)      
            
            ##output = output.write(i+1, out[0])   
            out = self.sample_from(pred[-1][-1])
            output = output.write(i+1, out)
            if out == end_token_embedding:
              break    
        #print(output.stack().numpy())
        molecules.append(output.stack())
        
    smiles = self.int_to_string_smiles(molecules)
    perc_valid = self.validity(smiles)
    
    return smiles, perc_valid
        
  def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
       # print(logits, indices)
        indices = np.asarray(indices).astype("int32")
        #print(indices)
        preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        #print(preds)
        preds = np.asarray(preds).astype("float32")
        #print(preds)
        return np.random.choice(indices, p=preds)
        
              
  def int_to_string_smiles(self, predictions):   
        inv_dict = {v: k for k, v in self.smiles_dict.items()}   
        
        smiles_pred=[]
        for mol in range(len(predictions)):
            smile = ''
            for token in range(1,len(predictions[mol][:])):
              #print(pred[mol][token].numpy())
              if (predictions[mol][token].numpy() == self.smiles_dict['[SEP]'] or predictions[mol][token].numpy() == self.smiles_dict['[PAD]']):
                 break
              smile += f'{inv_dict[predictions[mol][token].numpy()]}'
            smiles_pred.append(smile)
        return smiles_pred
    
  def validity(self, smiles_list):  
        total = len(smiles_list)
        valid_smiles =[]
        count = 0
        for sm in smiles_list:
            if MolFromSmiles(sm) != None and sm !='':
                valid_smiles.append(sm)
                count = count +1
        perc_valid = count*100/total
        
        return perc_valid
        
  """
  
  
class Generator():
    def __init__(
        self, model, max_tokens, start_tokens,smiles_dict, top_k=10, print_every=1
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.print_every = print_every
        self.k = top_k
        self.smiles_dict = smiles_dict

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def  __call__(self,num_generation,  epoch, logs=None):
        molec = []
        for generation in range(num_generation):
                
            start_tokens = [_ for _ in self.start_tokens]
            if (epoch + 1) % self.print_every != 0:
                return
            num_tokens_generated = 0
            tokens_generated = []
            while num_tokens_generated <= self.max_tokens:
                pad_len = self.max_tokens - len(start_tokens)
                sample_index = len(start_tokens) - 1
                if pad_len < 0:
                    x = start_tokens[:self.max_tokens]
                    sample_index = self.max_tokens - 1
                elif pad_len > 0:
                    x = start_tokens + [0] * pad_len
                else:
                    x = start_tokens
                x = np.array([x])
                y, _ = self.model.predict(x)
                sample_token = self.sample_from(y[0][sample_index])
                tokens_generated.append(sample_token)
                start_tokens.append(sample_token)
                num_tokens_generated = len(tokens_generated)
            molec.append(self.start_tokens + tokens_generated)
        smiles_list = self.int_to_string_smiles(molec)
        #print(f"generated text:\n{txt}\n") 
        valid_smiles, valid = self.validity(smiles_list)
        print(len(valid_smiles), valid_smiles)
        
        if len(valid_smiles) > 0:            
            perc_unique_smiles = self.uniqueness(valid_smiles)
            td = None#self.diversity(valid_smiles, None)
        else:
            perc_unique_smiles = None
            td = None
            
        return smiles_list, valid_smiles, valid, perc_unique_smiles, td
    
    
    def int_to_string_smiles(self, predictions):   
        inv_dict = {v: k for k, v in self.smiles_dict.items()}   
        smiles_pred=[]
        for mol in range(len(predictions)):
            smile = ''
            for token in range(1,len(predictions[mol][:])):
              #print(pred[mol][token].numpy())
              if (predictions[mol][token] == self.smiles_dict['[SEP]'] or predictions[mol][token] == self.smiles_dict['[PAD]']):
                 break
              smile += f'{inv_dict[predictions[mol][token]]}'
            smiles_pred.append(smile)
        return smiles_pred
        
    def validity(self, smiles_list):  
        total = len(smiles_list)
        valid_smiles =[]
        count = 0
        for sm in smiles_list:
            if MolFromSmiles(sm) != None and sm !='':
                valid_smiles.append(sm)
                count = count +1
        perc_valid = count*100/total
        
        return valid_smiles, perc_valid    
    
    
    def uniqueness(self, smiles_list):
        unique_smiles=list(set(smiles_list))
        return (len(unique_smiles)/len(smiles_list))*100
    
    def diversity(self,smiles_A,smiles_B):
        td = 0
        
        fps_A = []
        for i, row in enumerate(smiles_A):
            print(i, row)
            try:
                mol = MolFromSmiles(row)
                fps_A.append(AllChem.GetMorganFingerprint(mol, 3))
            except:
                print('ERROR: Invalid SMILES!') 
            
        if smiles_B == None:
            for ii in range(len(fps_A)):
                for xx in range(len(fps_A)):
                    ts = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
                    td += ts          
          
            td = td/len(fps_A)**2
        else:
            fps_B = []
            for j, row in enumerate(smiles_B):
                try:
                    mol = Chem.MolFromSmiles(row)
                    fps_B.append(AllChem.GetMorganFingerprint(mol, 3))
                except:
                  
                   print('ERROR: Invalid SMILES!') 
            
            for jj in range(len(fps_A)):
                for xx in range(len(fps_B)):
                    ts = 1 - DataStructs.TanimotoSimilarity(fps_A[jj], fps_B[xx]) 
                    td += ts
            
            td = td / (len(fps_A)*len(fps_B))
        print("Tanimoto distance: " + str(td))  
        return td





class SMILES_Generator():
    def __init__(
        self, model, max_tokens, start_str_tokens,end_str_tokens, smiles_dict, top_k=10):
        self.model = model
        self.max_tokens = max_tokens
        self.start_str_tokens = start_str_tokens
        self.end_str_tokens = end_str_tokens
        self.k = top_k
        self.smiles_dict = smiles_dict

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        #print(logits, indices)
        indices = tf.cast(indices, dtype = tf.int32)
        #print(f'indices {indices.shape}')
        logits = logits[tf.newaxis]
        preds = tf.keras.activations.softmax(logits)
        #print( preds)
        preds = np.asarray(tf.squeeze(preds)).astype("float32")
        token = np.random.choice(indices, p=preds)
        #print(f'sample {token}')
        return token

    def  __call__(self,num_to_generate):
        start_tokens = self.smiles_dict[self.start_str_tokens]
        end_tokens = self.smiles_dict[self.end_str_tokens] 
        
        print('>>>>>>>>>>>>>>> Generation <<<<<<<<<<<<<<<')   
        molecules = []
       
        for num in range(num_to_generate):
            index = 0
            #print(f'{num+1}/{num_to_generate}')        
            x = np.zeros([self.max_tokens])
            x[index] = start_tokens
            #print(f' x start {x}, shape {x.shape}')
            while index < self.max_tokens:
                index += 1
                #print(index)
                #print(i)
                #print(output.stack().numpy().shape)
                #print(output.stack().numpy()[:,tf.newaxis])
                
                #print(f'input:{input_seq.shape} e output: {output.stack().shape}')
                #print(x[tf.newaxis].shape)
                inp = tf.cast(x[tf.newaxis], dtype=tf.int32)
                pred = self.model.predict(inp)
                #print(f'pred {pred[0]}')
                #print(f'pred shape {pred.shape}')
                #layer = tf.keras.layers.Softmax()
                preds = tf.nn.softmax(pred, axis = -1)#layer(inp).numpy()
                
                #print(preds, preds.shape)                
                sample_token = self.sample_with_temp(preds[0][index-1]) #self.sample_from(pred[0][index-1])
                #print(f'sample {sample_token}')
                
                if index < self.max_tokens:  
                    x[index] = sample_token
                if index == self.max_tokens:  
                    x = np.append(x,[sample_token],-1)
                
                #print(f'x {x}, shape {x.shape}')
                
                if sample_token == end_tokens:
                  break    
            #print(output.stack().numpy())
            molecules.append(x)

        smiles_list = self.int_to_string_smiles(molecules)
        #print(f"generated text:\n{txt}\n") 
        valid_smiles, valid = self.validity(smiles_list)
      
            
        return smiles_list, valid_smiles, valid
    
    
    def int_to_string_smiles(self, predictions):   
        inv_dict = {v: k for k, v in self.smiles_dict.items()}   
        smiles_pred=[]
        for mol in range(len(predictions)):
            smile = ''
            for token in range(1,len(predictions[mol][:])):
              #print(pred[mol][token].numpy())
              if (predictions[mol][token] == self.smiles_dict['[SEP]'] or predictions[mol][token] == self.smiles_dict['[PAD]']):
                 break
              smile += f'{inv_dict[predictions[mol][token]]}'
            smiles_pred.append(smile)
        return smiles_pred
        
    def validity(self, smiles_list):  
        total = len(smiles_list)
        valid_smiles =[]
        count = 0
        for sm in smiles_list:
            if MolFromSmiles(sm) != None and sm !='':
                valid_smiles.append(sm)
                count = count +1
        perc_valid = count*100/total
        
        return valid_smiles, perc_valid    
     
    def sample_with_temp(self, preds):
        
        """
        #samples an index from a probability array 'preds'
        preds: probabilities of choosing a character
        
        """
       
        preds_ = np.log(preds).astype('float64')/0.8 #self.sampling_temp
        probs= np.exp(preds_)/np.sum(np.exp(preds_))
        #out = np.random.choice(len(preds), p = probs)
        
        out=np.argmax(np.random.multinomial(1,probs, 1))
        return out
        


class SMILES_Generator_():
    def __init__(
        self, model, max_tokens, start_str_tokens,end_str_tokens, smiles_dict, sample_temperature=0.8):
        self.model = model
        self.max_tokens = max_tokens
        self.start_str_tokens = start_str_tokens
        self.end_str_tokens = end_str_tokens
        self.sample_temperature = sample_temperature
        self.smiles_dict = smiles_dict


    def  __call__(self,num_to_generate):
        start_tokens = self.smiles_dict[self.start_str_tokens]
        end_tokens = self.smiles_dict[self.end_str_tokens] 
        
        print('Generating molecules...')   
        molecules = []
       
        for num in range(num_to_generate):
            index = 0
            #print(f'{num+1}/{num_to_generate}')        
            x = np.zeros([self.max_tokens])
            x[index] = start_tokens
            #print(f' x start {x}, shape {x.shape}')
            while index < self.max_tokens:
                index += 1
                #print(index)
                #print(i)
                #print(output.stack().numpy().shape)
                #print(output.stack().numpy()[:,tf.newaxis])
                
                #print(f'input:{input_seq.shape} e output: {output.stack().shape}')
                #print(x[tf.newaxis].shape)
                inp = tf.cast(x[tf.newaxis], dtype=tf.int32)
                pred,_ = self.model.predict(inp)
                #print(f'pred {pred[0]}')
                #print(f'pred shape {pred.shape}')
                #layer = tf.keras.layers.Softmax()
                #print(f'pred {pred.shape}')
                preds = tf.nn.softmax(pred, axis = -1)#layer(inp).numpy()
                
                #print(preds, preds.shape)                
                sample_token = self.sample_with_temp(preds[0][index-1]) #self.sample_from(pred[0][index-1])
                #print(f'sample {sample_token}')
                
                if index < self.max_tokens:  
                    x[index] = sample_token
                if index == self.max_tokens:  
                    x = np.append(x,[sample_token],-1)
                
                #print(f'x {x}, shape {x.shape}')
                
                if sample_token == end_tokens:
                  break    
            #print(output.stack().numpy())
            molecules.append(x)

        smiles_list = self.int_to_string_smiles(molecules)
        #print(f"generated text:\n{txt}\n") 
        valid_smiles, valid = self.validity(smiles_list)
      
            
        return smiles_list, valid_smiles, valid
    
    
    def int_to_string_smiles(self, predictions):   
        inv_dict = {v: k for k, v in self.smiles_dict.items()}   
        smiles_pred=[]
        for mol in range(len(predictions)):
            smile = ''
            for token in range(1,len(predictions[mol][:])):
              #print(pred[mol][token].numpy())
              if (predictions[mol][token] == self.smiles_dict['[SEP]'] or predictions[mol][token] == self.smiles_dict['[PAD]']):
                 break
              smile += f'{inv_dict[predictions[mol][token]]}'
            smiles_pred.append(smile)
        return smiles_pred
        
    def validity(self, smiles_list):  
        total = len(smiles_list)
        valid_smiles =[]
        count = 0
        for sm in smiles_list:
            if MolFromSmiles(sm) != None and sm !='':
                valid_smiles.append(sm)
                count = count +1
        perc_valid = count*100/total
        
        return valid_smiles, perc_valid    
     
    def sample_with_temp(self, preds):
        
        """
        #samples an index from a probability array 'preds'
        preds: probabilities of choosing a character
        
        """
       
        preds_ = np.log(preds).astype('float64')/self.sample_temperature
        probs= np.exp(preds_)/np.sum(np.exp(preds_))
        #out = np.random.choice(len(preds), p = probs)
        
        out=np.argmax(np.random.multinomial(1,probs, 1))
        return out





class Generator_():
    def __init__(
        self, model, max_tokens, start_str_tokens,end_str_tokens, smiles_dict, sample_temperature = 0.9):
        self.model = model
        self.max_tokens = max_tokens
        self.start_str_tokens = start_str_tokens
        self.end_str_tokens = end_str_tokens
        self.sample_temperature = sample_temperature
        self.smiles_dict = smiles_dict


    def  __call__(self,num_to_generate):
        start_tokens = self.smiles_dict[self.start_str_tokens]
        #end_tokens = self.smiles_dict[self.end_str_tokens] 
        
        #print('>>>>>>>>>>>>>>> Generation <<<<<<<<<<<<<<<') 
        print('Generating molecules...')
       
        x = np.zeros([num_to_generate, self.max_tokens])
        index = 0
        x[:, index] = start_tokens
        #print(f' x start {x}, shape {x.shape}') # (num_to_generate,73) 
        while index < self.max_tokens:
            index += 1
                #print(index)
                #print(i)
                #print(output.stack().numpy().shape)
                #print(output.stack().numpy()[:,tf.newaxis])
                
                #print(f'input:{input_seq.shape} e output: {output.stack().shape}')
                #print(x[tf.newaxis].shape)
            inp = tf.cast(x, dtype=tf.int32)
            pred, _ = self.model.predict(inp)
            #(f'pred shape {pred.shape}') #(num_to_generate, 73, 35)
                #print(f'pred {pred[0]}')
                #print(f'pred shape {pred.shape}')
                #layer = tf.keras.layers.Softmax()
            preds = tf.nn.softmax(pred, axis = -1)#layer(inp).numpy()
            #print(f'preds {preds.shape}') #(num_to_generate, 73, 35)    
                #print(preds, preds.shape)                
            #sample_tokens = [self.sample_with_temp(preds[i][index-1]) for i in range (num_to_generate)] #self.sample_from(pred[0][index-1])
            sample_tokens = self.sample_temp(preds[:,index-1], num_to_generate)
            #print(f'sample {sample_token}')
                
            if index < self.max_tokens:  
                #print(x.shape, sample_tokens.shape)  
                x[:,index] = sample_tokens[:,0]
            if index == self.max_tokens:  
                #print(x.shape, sample_tokens)  
                x = np.concatenate((x, np.array(sample_tokens[:,0])[:,np.newaxis]), axis=-1)
                #x = np.append(x,sample_tokens,-1)
                
                #print(f'x {x}, shape {x.shape}')
                
        #print(x.shape, sample_tokens)    
            #print(output.stack().numpy())
        molecules = x.tolist()

        smiles_list = self.int_to_string_smiles(molecules)
        #print(f"generated text:\n{txt}\n") 
        valid_smiles, valid = self.validity(smiles_list)
      
            
        return smiles_list, valid_smiles, valid
    
    
    def int_to_string_smiles(self, predictions):   
        inv_dict = {v: k for k, v in self.smiles_dict.items()}   
        smiles_pred=[]
        for mol in range(len(predictions)):
            smile = ''
            for token in range(1,len(predictions[mol][:])):
              #print(pred[mol][token].numpy())
              if (predictions[mol][token] == self.smiles_dict['[SEP]'] or predictions[mol][token] == self.smiles_dict['[PAD]']):
                 break
              smile += f'{inv_dict[predictions[mol][token]]}'
            smiles_pred.append(smile)
        return smiles_pred
        
    def validity(self, smiles_list):  
        total = len(smiles_list)
        valid_smiles =[]
        count = 0
        for sm in smiles_list:
            if MolFromSmiles(sm) != None and sm !='':
                valid_smiles.append(sm)
                count = count +1
        perc_valid = count*100/total
        
        return valid_smiles, perc_valid    
     
    def sample_temp(self, preds, num_to_generate):
        #print(f'preds sample {preds.shape}') #(35,)
        preds_ = np.log(preds).astype('float64')/self.sample_temperature #self.sampling_temp
        probs= tf.nn.softmax(preds_, axis = -1) #np.exp(preds_)/np.sum(np.exp(preds_))
        #out = np.random.choice(len(preds), p = probs)
        r_p = [np.random.multinomial(1,probs[i,:], 1) for i in range(num_to_generate)]
        out=np.argmax(np.array(r_p), axis = -1)
        return out
