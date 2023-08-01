# -*- coding: utf-8 -*-
"""

"""


import pandas as pd
import json
import tensorflow as tf
import tensorflow_addons as tfa
from utils import *
import numpy as np
import itertools
import glob
import csv
import argparse
import sys
import os

config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def cmd_options():
    parser = argparse.ArgumentParser(description='Program that trains, validates and tests the predictor A2A recetor Model')

    parser.add_argument('--train', action='store_true', help='Training')
    parser.add_argument('--validation', action='store_true', help='Validating')
    parser.add_argument('--test', action='store_true', help='Testing')

    parser.add_argument('--rate', type = int,  default=0.1, metavar='', help='Dropout rate')
    parser.add_argument('--hidden_func', nargs="+", default=[], type = str, metavar='', help='Activation functions of dense hidden layers (eg. 512 128)')
    parser.add_argument('--hidden_dim', nargs="+", default=[], type = int, metavar='', help='Hidden layers dimensions, must be in sync with --hidden_func (eg. relu linear)')
    
    parser.add_argument('--max_smiles_len', type = int,  default=72, metavar='', help='Maximum length of SMILES for training, needs to be larger than 0')
    parser.add_argument('--batch', type = int,  default=32, metavar='', help='Batch size')
    parser.add_argument('--epochs', type = int,  default=200, metavar='', help='Number of epochs')
    parser.add_argument('--optimizer', type = str,  default='RAdam', metavar='', help='Optimizer')
    parser.add_argument('--patience', type = int,  default=20, metavar='', help='Patience in EarlyStopping')
    parser.add_argument('--min_delta', type = float,  default=0.0001, metavar='', help='EarlyStopping mininum delta')

    parser.add_argument('--path_cluster', type = str, default = "../dataset/clusters_indices/", metavar='', help='Path of clusters used for applying cross validation')
    parser.add_argument('--path_csv', type=str, default = "../results/classification_gridsearch/gridsearch_results.csv", metavar='', help='Path of csv to write the results of cross validation')
     
    parser.add_argument('--metrics', type=bool, default = True, metavar='', help='True if to show the metrics of the model with the test dataset')
    parser.add_argument('--graph',  type=bool, default = True, metavar='', help='True if to show the graph with y_test and y_pred of the model with the test dataset')
    parser.add_argument('--fig_path',  type=str, default = "../results/classification/", metavar='', help='Path to save the figure')
    parser.add_argument('--graph_title',  type=str, default = 'Adenosine A2A', metavar='', help='Graphic title')

    parser.add_argument('--training_dataset', type = str, default="../dataset/data_classification_a2a_train.csv", metavar='', help='Training dataset path')
    parser.add_argument('--testing_dataset', type = str, default="../dataset/data_classification_a2a_test.csv", metavar='', help='Testing dataset path')
    parser.add_argument('--dict', type = str, default='../dictionary/dictionary_smiles.txt', metavar='', help='dictionary smiles path')
    parser.add_argument('--save_path', type = str, default="../models/classification/", metavar='', help='save model path')
    parser.add_argument('--saved_model',required='--test' in sys.argv, type = str, default = None, metavar='', help='saved model path')
    parser.add_argument('--pre_train', type = str, default = "../models/masking_pre_train/model_4heads_4encoders_batch_size256_RAdam.h5py", metavar='', help='Path to the model file with the masked learning model trained')

    args = parser.parse_args()
    return args



    
class Classification_SMILES_Model(tf.keras.Model):
    def __init__(self, smiles_len, rate, optimizer, loss, metrics_list, cls_layers_units, cls_act_func,pre_train_path, **kwargs):
        super().__init__()
        
        self.smiles_input = tf.keras.Input(shape=(smiles_len,), dtype=tf.int32, name='smiles_input') 

        self.model_trained = tf.keras.models.load_model(pre_train_path)
        self.model_trained.summary()
	
        self.ffnn_layer = [tf.keras.layers.Dense(units=cls_layers_units[i],activation=cls_act_func[i], name=f'Classification_Layer_{i}') for i in range(len(cls_layers_units))]
        self.dropout_layer = [tf.keras.layers.Dropout(rate, name=f'Dropout_Layer_{i}') for i in range(len(cls_layers_units))]
        
        self.final_layer = tf.keras.layers.Dense(units=1,activation='linear', name='Final_Classification_Layer')
                                         
        self.optimizer = optimizer
        self.loss = loss
        self.metrics_list = metrics_list

        self.model = self.build_model()
              
        
    def build_model(self):
       
        smiles_in = self.smiles_input 
        smiles_out,_,_,all_layers = self.model_trained(smiles_in)  
       
        smiles_cls_out = tf.gather(smiles_out, 0, axis=1)

        for i in range(len(self.ffnn_layer)):
           smiles_cls_out = self.dropout_layer[i](self.ffnn_layer[i](smiles_cls_out))
          
        out = self.final_layer(smiles_cls_out)
        
        model = tf.keras.Model(inputs=[smiles_in],outputs=[out], name='predictor_model')
        model.compile(optimizer=self.optimizer,loss=self.loss, metrics=self.metrics_list)
        model.summary()

        return model
    
    
    def fit_model(self,dataset, batch, epochs, callback_list = None, val_option = False, val_dataset = None):
        
        smiles_inp, target = dataset
        callback_list = callback_list
    
        if not val_option:
            self.model.fit(x=smiles_inp, y=target,
                      batch_size = batch, epochs = epochs,
                      verbose = 2, callbacks = callback_list)
            
        else:
            smiles_val_inp, target_val_y = val_dataset
            self.model.fit(x=smiles_inp, y=target,         
                      batch_size = batch, epochs = epochs,
                      verbose = 2, callbacks = callback_list, 
                      validation_data=([smiles_val_inp], [target_val_y])) 
       
        return self.model



def grid_search(parameters,data,k_folds,folder,save_file):
    metrics_results=[[],[],[]]
    
    for num_run in range(len(k_folds)):
        print("-------------------//--------------------")
        print("Run: "+str(num_run))
        print("-------------------//--------------------")
        index_train=list(itertools.chain.from_iterable([k_folds[i] for i in range(len(k_folds)) if i!=num_run]))

        index_val=k_folds[num_run]

        
        data_train = [tf.gather(i,index_train) for i in data]
  
        data_val = [tf.gather(i,index_val) for i in data]
        
        es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                         min_delta = parameters['min_delta'], patience = parameters['patience'], mode = 'min', 
                                         restore_best_weights=True)
        
        cls_units= parameters['cls_dim_layers'].copy()
        cls_units.append(1)
        cls_act = parameters['cls_act_func'].copy()
        cls_act.append('linear')
        print(cls_units,cls_act)

        file_name = f"/{parameters['pre_train_path'].split('/')[-1].split('.')[0]}_{'_'.join(cls_act)}_{'_'.join([str(i) for i in cls_units])}_{parameters['batch_size']}_{parameters['epochs']}_{num_run}"
        #file_name = f"/model_trained_{num_run}"

        mc = tf.keras.callbacks.ModelCheckpoint(filepath = folder+file_name, monitor = 'loss',
                                           save_best_only=True, save_weights_only = False, mode = 'min')
        
        callbacks = [es, mc]
        
        optimizer = tfa.optimizers.RectifiedAdam(lr=5e-5)
        pred_model = Classification_SMILES_Model(parameters['max_inp_len'],parameters['Dropout'], optimizer, 'mse', parameters['list_metrics'],
                                                parameters['cls_dim_layers'], parameters['cls_act_func'],parameters['pre_train_path']).fit_model(data_train,
                                        parameters['batch_size'],parameters['epochs'],callbacks,True,data_val)
       
        mse, q2, ccc = pred_model.evaluate([data_val[0]],[data_val[1]])
        
        metrics_results[0].append(mse)
        metrics_results[1].append(q2)
        metrics_results[2].append(ccc)
        
        
        result_values = (parameters['pre_train_path'], cls_act, cls_units, parameters['batch_size'], parameters['epochs'], mse,q2,ccc,num_run)       

        
        save_func(save_file,result_values, ['pre_train_path', 'cls_act', 'cls_units', 'batch_size', 'epochs', 'mse', 'q2', 'ccc', 'num_run'])
        tf.keras.backend.clear_session()
        
    result_values = (parameters['pre_train_path'], cls_act, cls_units, parameters['batch_size'], parameters['epochs'], mse,q2,ccc,'Mean')  
        
        
    #save_func(save_file,result_values)   





if __name__ == '__main__':

    args = cmd_options()

    smiles_dictionary = json.load(open(args.dict))


    if args.train:

        es = tf.keras.callbacks.EarlyStopping(monitor ='loss',
                                          min_delta = args.min_delta, patience = args.patience, mode = 'min', 
                                          restore_best_weights=True)
        
        cls_units= args.hidden_dim.copy()
        cls_units.append(1)
        cls_act = args.hidden_func.copy()
        cls_act.append('linear')

        file_name = f"{args.pre_train.split('/')[-1].split('.')[0]}_{'_'.join(cls_act)}_{'_'.join([str(i) for i in cls_units])}_{args.batch}_{args.epochs}_{args.optimizer}"
 
        mc = tf.keras.callbacks.ModelCheckpoint(filepath = args.save_path+ file_name,
                                                monitor = 'loss',
                                            save_best_only=True, save_weights_only = False, mode = 'min')
       
        callbacks = [es,mc]

        df = pd.read_csv(args.training_dataset, index_col = False)  
        tokens_mol = [smiles_str_to_tokens(x) for x in df['smiles'].tolist()]
        inp_smiles_emb, _,_,_  = smiles_tokens_to_idx_and_padding(tokens_mol, args.max_smiles_len, smiles_dictionary)  


        pic50, q1, q3 = normalization(df['pIC50'].tolist())
        with open(f"../dataset/norm_q1_q3_{args.training_dataset.split('/')[-1].split('.')[0]}.txt", 'w') as f:
            json.dump({"q1":q1, "q3":q3}, f)


        inp = tf.convert_to_tensor(inp_smiles_emb)
        target = tf.convert_to_tensor(pic50)[:,tf.newaxis]

        mse_loss = tf.keras.losses.MeanSquaredError(name='mean_squared_error')
        
        list_metrics = [q2, ccc] 
        
        if args.optimizer == 'RAdam':
            optimizer = tfa.optimizers.RectifiedAdam(learning_rate=5e-5)
        else:
            optimizer = args.optimizer
        
        model = Classification_SMILES_Model(args.max_smiles_len+2, args.rate, optimizer, 'mse', list_metrics, args.hidden_dim, args.hidden_func, args.pre_train)
        model.fit_model((inp, target), args.batch, args.epochs,callbacks,None,None)


    if args.validation:

        df = pd.read_csv(args.training_dataset, index_col = False)  
        tokens_mol = [smiles_str_to_tokens(x) for x in df['smiles'].tolist()]
        smiles_emb, _,_,_  = smiles_tokens_to_idx_and_padding(tokens_mol, args.max_smiles_len, smiles_dictionary)  

        pic50,_,_ = normalization(df['pIC50'].tolist())

        inp = tf.convert_to_tensor(smiles_emb)
        target = tf.convert_to_tensor(pic50)[:,tf.newaxis]
        
        clusters = find_clusters(glob.glob(f'{args.path_cluster}*'))

        for cls_units, cls_act in zip([[], [512,128], [512],[512,128]], [[], ['gelu', 'relu'], ['linear'], ['linear','gelu']]):
            parameters = {'max_inp_len':args.max_smiles_len+2,'patience':args.patience ,'min_delta': args.min_delta,'Dropout': args.rate,
                          'cls_act_func': cls_act, 'cls_dim_layers':cls_units,'list_metrics':[q2, ccc], 'batch_size': args.batch,'epochs':args.epochs,
                          'pre_train_path': args.pre_train}
            
            grid_search(parameters,[inp, target],
                        [list(clusters[i][1].iloc[:,0]) for i in range(len(clusters))],
                        '/'.join(args.path_csv.split('/')[:-1]), args.path_csv)
            tf.keras.backend.clear_session()


    if args.test:

        norm_parameters = json.load(open(f"../dataset/norm_q1_q3_{args.training_dataset.split('/')[-1].split('.')[0]}.txt"))
        q1 = norm_parameters['q1']
        q3 = norm_parameters['q3']
   
        df_test = pd.read_csv(args.testing_dataset, index_col = False) 
        tokens_mol = [smiles_str_to_tokens(x) for x in df_test['smiles'].tolist()]
        smiles_emb_test, _,_,_  = smiles_tokens_to_idx_and_padding(tokens_mol, args.max_smiles_len, smiles_dictionary)      
        pic50_test,_,_ = normalization(np.array(df_test['pIC50']), q1, q3)

        x_test = tf.convert_to_tensor(smiles_emb_test)
        y_test = tf.convert_to_tensor(pic50_test)[:,tf.newaxis]
        
        trained_model = tf.keras.models.load_model(args.saved_model,custom_objects={'q2':q2, 'ccc':ccc}) #('../model/classification/smiles_bert_4_4_RAdam_ffn_512linear_128_gelu_1linear',custom_objects={'q2':q2, 'ccc':ccc}) 
               
        if args.metrics:
            pred_metrics = inference_metrics(trained_model,[x_test,y_test])
            print(f'Metrics:')
            for metric, value in pred_metrics.items():
                print(metric, round(value.numpy(),4))

        save_func(f'{args.fig_path}results.csv',(args.saved_model,pred_metrics['MSE'].numpy(),pred_metrics['R2'].numpy(),pred_metrics['CCC'].numpy()),['model_path','MSE','R2','CCC'])

        if args.graph:
            pred_values = trained_model.predict([x_test]) 
            scatter_plot(y_test, pred_values,args.graph_title,f"{args.fig_path}{args.testing_dataset.split('/')[-1].split('.')[0]}_{args.saved_model.split('/')[-1]}.png")


        


        