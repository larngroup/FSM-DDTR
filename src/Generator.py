# -*- coding: utf-8 -*-
"""


"""

import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import json
import random
import time
import argparse
import sys
import os

from transformers import Transformer_Decoder
from utils import *
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def cmd_options():
    parser = argparse.ArgumentParser(description='Program that trains the Generator Model and generates molecules')

    parser.add_argument('--train', action='store_true', help='Training')
    parser.add_argument('--generate', action='store_true', help='Generating')


    parser.add_argument('--d_model', type = int,  default=512, metavar='', help='D_model (dimensional embedding vector)')
    parser.add_argument('--dff', type = int, default=1024, metavar='', help='Dff (feed forward  projection size)')
    parser.add_argument('--num_heads', type = int, default=8, metavar='', help='Number of heads')
    parser.add_argument('--num_decoders', type = int,  default=4, metavar='', help='Number of decoders')
    parser.add_argument('--rate', type = float,  default=0.1, metavar='', help='Rate')
    parser.add_argument('--func', type = str,  default='relu', metavar='', help='Activation Function')
    parser.add_argument('--batch', type = int,  default=256, metavar='', help='Batch size')
    parser.add_argument('--epochs', type = int,  default=100, metavar='', help='Number of epochs')
    parser.add_argument('--optimizer', type = str,  default='RAdam', metavar='', help='Optimizer')
    parser.add_argument('--patience', type = int,  default=10, metavar='', help='Patience in EarlyStopping')
    parser.add_argument('--min_delta', type = float,  default=0.001, metavar='', help='EarlyStopping mininum delta')
    parser.add_argument('--max_smiles_len', type = int,  default=72, metavar='', help='Maximum length of SMILES for training, needs to be larger than 0')
    parser.add_argument('--num_smiles', type = int, metavar='', help='Number of SMILES for training, needs to be larger than 0')

    parser.add_argument('--temperature',  type=float, default=0.9, metavar='', help='Sample temperature used to add randomness to generation')
    parser.add_argument('--num_generation', type=int, default=500, metavar='', help='Amount of molecules to generate')
    parser.add_argument('--metrics',  type=bool, default = True, metavar='', help='True if to show diversity, validity, uniqueness metrics')
    parser.add_argument('--mol_path', type=str, default = "../molecules/pre_train/", metavar='', help='Path to store the resulting molecules from the trained generator (include valid and non-valid molecules)')
     
    parser.add_argument('--dataset', type = str, default="../dataset/data_for_generation_notduplicate.csv", metavar='', help='dataset path')
    parser.add_argument('--dict', type = str, default="../dictionary/dictionary_smiles.txt", metavar='', help='dictionary smiles path')
    parser.add_argument('--save_path', type = str, default="../models/generation/", metavar='', help='save model path')
    parser.add_argument('--saved_model', required='--generate' in sys.argv, type = str, metavar='', help='saved model path')
    parser.add_argument('--path_csv', type=str, default = "../results/generation/results.csv", metavar='', help='Path of csv to write the results of generation')
    
    args = parser.parse_args()
    return args




class Generator_Model():
    def __init__(self, d_model, dff, num_heads, num_decoders,smiles_max_tokens,smiles_vocab_size, act_func, rate, loss_object, optimizer, smiles_dictionary, patience, min_delta, path_saved_model = None):        
        if path_saved_model == None:
            self.model = Transformer_Decoder(d_model, dff, num_heads, num_decoders, smiles_max_tokens, smiles_vocab_size, act_func, rate)
        else: 
            self.model = tf.keras.models.load_model(path_saved_model)
        
        self.loss_object = loss_object
        self.patience = patience
        self.min_delta = min_delta
        self.smiles_dictionary = smiles_dictionary
        self.num_heads = num_heads
        self.num_decoders = num_decoders
        self.act_func = act_func
        self.smiles_max_tokens = smiles_max_tokens
        self.optimizer = optimizer


            
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
        y_true = tf.cast(y_true, dtype=tf.int32)
        #print(f'output_real {output_real.shape} and argmax_indices {argmax_indices.shape}')
        accuracy = tf.equal(y_true,pred)
        acc = tf.cast((accuracy), dtype=tf.float32)
        acc *= mask
        return tf.reduce_sum(acc)/tf.reduce_sum(mask)
        


    def train_step(self, x,target):

        with tf.GradientTape() as tape:
            predictions, _ = self.model(x, training = True)
                #print(f'pred {predictions}')
            loss = self.loss_function(target, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        acc = self.accuracy_function(target, predictions)
        return loss, acc

    
   
    def train_model(self, inp_smiles_emb, epochs, batch_size, filename):
        print('Training...')

        smiles_x = tensor_smiles[:,:-1]
        smiles_y = tensor_smiles[:,1:]
           
        inp = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(smiles_x)).batch(batch_size, drop_remainder = False)
        label = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(smiles_y)).batch(batch_size, drop_remainder = False)
        data = tf.data.Dataset.zip((inp, label))


        last_loss = {'epoch':0,'value':1000}

        for epoch in range(epochs):            
            print(f'Epoch {epoch+1}/{epochs}')
            
            start = time.time()           
            
            loss_epoch = []
            acc_epoch = []
            
            for num, (x_train, y_train) in enumerate(data):
                loss_generator, acc_generator = self.train_step(x_train, y_train)
                loss_epoch.append(loss_generator)
                acc_epoch.append(acc_generator)
                
                if num == len(data)-1:
                    print(f'{num+1}/{len(data)} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f} - accuracy: {np.mean(acc_epoch):.4f}')    
                    
            if (last_loss['value'] - np.mean(loss_epoch)) >= self.min_delta: 
                last_loss['value'] = np.mean(loss_epoch)
                last_loss['epoch'] = epoch+1
                print('Saving model...')
                self.model.save(filename)
            if ((epoch+1) - last_loss['epoch']) >= self.patience:
                break 


    def generate_molecules(self, num_to_generate, start_str_tokens,sample_temperature):
        print('Generating molecules...')

        start_tokens = self.smiles_dictionary[start_str_tokens] 
        x = np.zeros([num_to_generate, self.smiles_max_tokens])
        index = 0
        x[:, index] = start_tokens
        #print(f' x start {x}, shape {x.shape}') # (num_to_generate,73) 
        while index < self.smiles_max_tokens-1:
        
            index += 1
            inp = tf.cast(x, dtype=tf.int32)
            pred, _ = self.model.predict(inp)
            preds = tf.nn.softmax(pred, axis = -1)
            sample_tokens = self.sample_temp(preds[:,index-1], num_to_generate, sample_temperature)

            x[:,index] = sample_tokens[:,0]

        molecules = x.tolist()

        smiles_list = int_to_string_smiles(molecules,self.smiles_dictionary)
        valid_smiles, valid = validity(smiles_list)      
            
        return smiles_list, valid_smiles, valid
    


    def sample_temp(self, preds, num_to_generate, sample_temperature):
        preds_ = np.log(preds).astype('float64')/sample_temperature
        probs= tf.nn.softmax(preds_, axis = -1) #np.exp(preds_)/np.sum(np.exp(preds_))
        r_p = [np.random.multinomial(1,probs[i,:], 1) for i in range(num_to_generate)]
        out=np.argmax(np.array(r_p), axis = -1)
        return out



if __name__ == '__main__':

    args = cmd_options()

    smiles_dictionary = json.load(open(args.dict))

    if args.train:           

        df = pd.read_csv(args.dataset)  
        df = df.drop_duplicates()   
        
        if args.num_smiles and (args.num_smiles < len(df)):
            df = df.iloc[0:args.num_smiles]

        smiles = df['smiles'].tolist()

        tokens_mol = [smiles_str_to_tokens(x) for x in smiles]
        inp_smiles_emb, _, _, _  = smiles_tokens_to_idx_and_padding(tokens_mol, args.max_smiles_len, smiles_dictionary)  
        
        tensor_smiles = tf.convert_to_tensor(inp_smiles_emb)
        

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        filename = f"{args.save_path}model_{args.d_model}_{args.dff}_{args.num_heads}_{args.num_decoders}_{args.func}_batch{args.batch}_epoch{args.epochs}_{args.optimizer}_data_{tensor_smiles.shape[0]}.h5py"
        #filename = f"{args.save_path}model_trained.h5py"

        if args.optimizer == 'RAdam':
            optimizer = tfa.optimizers.RectifiedAdam(
                 learning_rate = 1e-3,
                 beta_1 = 0.9,
                 beta_2 = 0.999,
                 weight_decay = 0.1)
        else:
            optimizer = args.optimizer

        model = Generator_Model(args.d_model, args.dff, args.num_heads, args.num_decoders, args.max_smiles_len+1, len(smiles_dictionary),
                                     args.func, args.rate, loss_object, optimizer, smiles_dictionary, args.patience, args.min_delta, path_saved_model = args.saved_model)

        model.train_model(tensor_smiles, args.epochs, args.batch, filename)



    if args.generate:
        
        model = Generator_Model(None, None, None, None, args.max_smiles_len+1, len(smiles_dictionary),
                                     None, None, None, None, smiles_dictionary, None, None, path_saved_model = args.saved_model)

        smiles_list, valid_smiles, valid = model.generate_molecules(args.num_generation, '[CLS]', args.temperature)
        
        if valid_smiles != []:
            unique = uniqueness(valid_smiles)
            div = diversity(valid_smiles)
        else:
            unique = None
            div = None

        if args.metrics:
            print(f'% valid: {valid}')
            print(f'% uniqueness: {unique}')  
            print(f'diversity: {div}')

        save_func(args.path_csv,(args.saved_model, args.temperature,args.num_generation,valid,unique,div),
             ['model_path','temperature','num_generate','%valid', '%uniqueness', 'diversity'])


        with open(f"{args.mol_path}/molecules_{args.temperature}T_{args.num_generation}.txt", "w") as outfile:
            outfile.write("\n".join(smiles_list))

