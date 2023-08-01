# -*- coding: utf-8 -*-
"""


"""

import numpy as np
import tensorflow as tf
import random


def positional_encoding(position, d_model):    
    pos = np.arange(position)[:, np.newaxis] #a column vector
    i = np.arange(d_model)[np.newaxis, :] #a row vector
    positencoding = np.zeros((position, d_model))
   
    angle = pos/np.power(10000, ( i / d_model))

    positencoding[:,::2] = np.sin(angle[:,::2])
    positencoding[:,1::2] = np.cos(angle[:,1::2])
    return positencoding[np.newaxis, ...] # (1, position, d_model)




def create_mask(seq):
    mask = tf.cast(tf.math.equal(seq, tf.zeros_like(seq)), tf.float32)
    return mask[:, tf.newaxis,:]  # (batch_size, 1, seq_len)




def scaled_dot_product(k,v,q,mask = None):
    """  q_shape  (batch, seq_len_q, dk);  k_shape == (batch, seq_len_k, dk);  v_shape == (batch, seq_len_v, dk_v) """
    
    dk = k.shape[-1]

    matmul = tf.cast(tf.matmul(q,k, transpose_b=True), tf.float32)  #(batch, seq_len_q, seq_len_k)
    scaleddotprod = matmul/tf.sqrt(tf.cast(dk, tf.float32))
    
    if mask is not None:
      scaleddotprod += (mask * (-1e9))   

    attention_weights = tf.nn.softmax(scaleddotprod, axis=-1) # (batch, seq_len_q, seq_len_k) 
    attention = tf.cast(tf.matmul(attention_weights,v), tf.float32) # (batch, seq_len_q, dk_v)
    return attention, attention_weights




class Multi_Head_Attention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        assert d_model % num_heads == 0
      
        self.num_heads = num_heads       
        self.d_model = d_model
        self.dk = d_model // num_heads
        
                 
    def build(self, input_shape):
        self.wq = [tf.keras.layers.Dense(self.dk, name='Wq_%d'%i) for i in range(self.num_heads)]
        self.wk = [tf.keras.layers.Dense(self.dk, name='Wk_%d'%i) for i in range(self.num_heads)]
        self.wv = [tf.keras.layers.Dense(self.dk, name='Wv_%d'%i) for i in range(self.num_heads)]
        self.w0 = tf.keras.layers.Dense(self.d_model,name ='W0')


    def call(self, v, k, q, mask):
       """ v,k, q tensors with shape (batch_size, seq_len, d_model)"""
       
       attention_all = []
       attention_weights_all = []
       for h in range(self.num_heads):
           attention, attention_weights = scaled_dot_product(self.wk[h](k), self.wv[h](v), self.wq[h](q), mask)
           attention_all.append(attention)
           attention_weights_all.append(attention_weights)
      
       attention_matrix = tf.concat(attention_all, axis = -1) # (batch_size, seq_len_q, d_model)   
       output = self.w0(attention_matrix) # (batch_size, seq_len_q, d_model)
       
       return output,attention_weights
    
   
    def get_config(self):
        config = super(Multi_Head_Attention,self).get_config()
        config.update({
            'd_model':self.d_model,
            'num_heads':self.num_heads,
            'dk': self.dk,
            'wq': self.wq,
            'wk':self.wk,
            'wv':self.wv,
            'w0':self.w0})
        return config



    
class Point_Wise_FeedForwardNetwork(tf.keras.layers.Layer):    
    def __init__(self, d_model, dff, act_func, rate = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dff = dff
        self.rate = rate
        self.act_func = act_func

           
    def build(self, input_shape):
        self.ff1 = tf.keras.layers.Dense(self.dff, activation=self.act_func)
        self.ff2 = tf.keras.layers.Dense(self.d_model)
        self.dropout = tf.keras.layers.Dropout(self.rate)

    
    def call(self,x, training):
        return self.ff2(self.dropout(self.ff1(x), training=training))
 

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model':self.d_model,
            'dff':self.dff,
            'act_func':self.act_func,
            'dropout_rate': self.rate,
            'ff1':self.ff1,
            'ff2':self.ff2,
            'dropout':self.dropout})
        return config



  
class Normalization_layer(tf.keras.layers.Layer):
    def __init__(self, rate):
        super().__init__()

        self.rate = rate


    def build(self, input_shape):
        self.normLayer = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop = tf.keras.layers.Dropout(self.rate)

   
    def call(self, input_x, output_sublayer, training):
        output_sublayer = self.drop(output_sublayer, training=training)
        output = self.normLayer(input_x + output_sublayer)
        return output


    def get_config(self):
        config = super().get_config()
        config.update({
            'dropout_rate': self.rate,
            'normLayer':self.normLayer,
            'drop':self.drop})
        return config



  
class Encoder_layer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads, act_func, rate = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.rate = rate
        self.act_func = act_func


    def build(self, input_shape):
        self.attentionLayer = Multi_Head_Attention(self.d_model, self.num_heads)
        self.ffLayer = Point_Wise_FeedForwardNetwork(self.d_model, self.dff, self.act_func, self.rate)
        self.normLayer = [Normalization_layer(self.rate) for i in range (2)]

    
    def call(self, x_v, x_k, x_q, mask, training):
        headAttention, encoder_weight = self.attentionLayer(x_v, x_k, x_q, mask)
        outputSubLayer1 = self.normLayer[0](x_q,headAttention, training)  #(batch, seq_len_q, d_model)
        
        ffn = self.ffLayer(outputSubLayer1, training) #(batch, seq_len_q, d_model)
        outputSubLayer2 = self.normLayer[1](outputSubLayer1, ffn, training)
        return outputSubLayer2, encoder_weight


    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dff': self.dff,
            'num_heads': self.num_heads,
            'act_func': self.act_func,
            'dropout_rate': self.rate,
            'attentionLayer':self.attentionLayer,
            'ffLayer':self.ffLayer,
            'normLayer':self.normLayer})
        return config



  
class Transformer_Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads, num_encoders,smiles_len, act_func, rate):
        super().__init__()

        self.smiles_len = smiles_len
        self.d_model = d_model
        self.dff = dff
        self.act_func = act_func
        self.rate = rate
        self.num_heads = num_heads
        self.num_encoders = num_encoders
        
            
    def build(self, input_shape):
        self.embeddingLayer = tf.keras.layers.Embedding(self.smiles_len+1, self.d_model)
        self.positionalEncodingLayer = positional_encoding(self.smiles_len, self.d_model)
        self.dropout = tf.keras.layers.Dropout(self.rate) 
        self.encoderStack = [Encoder_layer(self.d_model, self.dff, self.num_heads, self.act_func, self.rate) for _ in range (self.num_encoders)]

    
    def call(self, x, training = True):
        mask_smiles = create_mask(x)
        x = self.embeddingLayer(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += tf.convert_to_tensor(self.positionalEncodingLayer, dtype=tf.float32) #PE -> (1,input_seq_len,d_model)
        x = self.dropout(x, training=training)
        
        all_outputs = []
        all_weights = []    
        for i in range(self.num_encoders):
            x,encoder_weights = self.encoderStack[i](x, x, x, mask_smiles, training) # v, k, q
            all_weights.append(encoder_weights)
            all_outputs.append(x)

        return x, all_weights, all_outputs  # (batch_size, input_seq_len, d_model)


    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dff': self.dff,
            'act_func': self.act_func,
            'num_heads': self.num_heads,
            'dropout_rate': self.rate,
            'smiles_len': self.smiles_len,       
            'num_encoders':self.num_encoders,
            'embeddingLayer': self.embeddingLayer,
            'positionalEncodingLayer':self.positionalEncodingLayer,
            'dropout':self.dropout,
            'encoderStack': self.encoderStack})
        return config



  
class Masked_Smiles_Model(tf.keras.Model):#(tf.keras.layers.Layer):#(tf.keras.Model):
    def __init__(self, d_model, dff, num_heads, num_encoders,smiles_len,smiles_vocab_size, act_func, rate = 0.1):
        super().__init__()

        self.smiles_vocab_size = smiles_vocab_size
        self.smiles_len = smiles_len
        self.d_model = d_model
        self.dff = dff
        self.act_func = act_func
        self.rate = rate
        self.num_heads = num_heads
        self.num_encoders = num_encoders                   
          

    def build(self, input_shape):
        self.transformer_encoder = Transformer_Encoder(self.d_model, self.dff, self.num_heads, self.num_encoders,self.smiles_len, self.act_func, self.rate)
        self.ffnn = tf.keras.layers.Dense(self.smiles_vocab_size, activation='softmax')    

    
    def call(self, x, training = True):     
        x, all_weights, all_outputs = self.transformer_encoder(x, training)      
        x_ff = self.ffnn(x)
        return x, x_ff, all_weights, all_outputs
 

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dff': self.dff,
            'act_func': self.act_func,
            'num_heads': self.num_heads,
            'dropout_rate': self.rate,
            'vocab_size': self.smiles_vocab_size,
            'smiles_len': self.smiles_len,            
            'num_encoders':self.num_encoders,
            'transformer_encoder':self.transformer_encoder,
            'ffnn': self.ffnn})
        return config



  
class Decoder_layer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads, act_func, rate = 0.1, multi_head_block_per_decoder = 1):
        super().__init__()

        #assert (multi_head_block_per_decoder <= 2 and multi_head_block_per_decoder > 0)
        
        self.d_model = d_model
        self.dff = dff
        self.act_func = act_func
        self.num_heads = num_heads
        self.rate = rate
        self.multi_head_per_decoder = multi_head_block_per_decoder




    def build(self, input_shape):
        self.attentionLayer = Multi_Head_Attention(self.d_model, self.num_heads) #[Multi_Head_Attention(self.d_model, self.num_heads) for _ in range(self.multi_head_per_decoder)]
        self.ffLayer = Point_Wise_FeedForwardNetwork(self.d_model, self.dff, self.act_func, self.rate)
        self.normLayer = [Normalization_layer(self.rate) for _ in range(2)] #(self.multi_head_per_decoder +1)]

    
    def call(self, x_v, x_k, x_q, look_ahead_mask, training):#, mask = None, output_encoder = None):
        headAttention, decoder_weight = self.attentionLayer(x_v, x_k, x_q, look_ahead_mask)  #self.attentionLayer[0](x_v, x_k, x_q, look_ahead_mask)
        outputSubLayer = self.normLayer[0](x_q,headAttention, training)  #(batch, seq_len_q, d_model)

        #if self.multi_head_per_decoder > 1:
        #     headAttention, decoder_weight = self.attentionLayer[1](output_encoder, output_encoder, outputSubLayer, mask)
        #     outputSubLayer = self.normLayer[1](outputSubLayer, headAttention, training)  
        
        ffn = self.ffLayer(outputSubLayer, training) #(batch, seq_len_q, d_model)
        outputSubLayer = self.normLayer[1](outputSubLayer, ffn, training)#self.normLayer[self.multi_head_per_decoder](outputSubLayer, ffn, training)
        return outputSubLayer, decoder_weight


    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dff': self.dff,
            'act_func': self.act_func,
            'num_heads': self.num_heads,
            'dropout_rate': self.rate,
            'multi_head_per_decoder': self.multi_head_per_decoder,
            'attentionLayer':self.attentionLayer,
            'ffLayer':self.ffLayer,
            'normLayer':self.normLayer})
        return config



  
class Transformer_Decoder(tf.keras.Model):
    def __init__(self, d_model, dff, num_heads, num_decoders, smiles_len, smiles_vocab_size, act_func, rate = 0.1):
        super().__init__()

        self.smiles_vocab_size = smiles_vocab_size
        self.smiles_len = smiles_len
        self.d_model = d_model
        self.dff = dff
        self.act_func = act_func
        self.rate = rate
        self.num_heads = num_heads
        self.num_decoders = num_decoders
        
            
    def build(self, input_shape):
        self.embeddingLayer = tf.keras.layers.Embedding(self.smiles_len+2, self.d_model)
        self.positionalEncodingLayer = positional_encoding(self.smiles_len, self.d_model)
        self.dropout = tf.keras.layers.Dropout(self.rate) 
        self.decoderStack = [Decoder_layer(self.d_model, self.dff, self.num_heads, self.act_func, self.rate) for _ in range (self.num_decoders)] 
        self.ffLayer = tf.keras.layers.Dense(self.smiles_vocab_size)

    
    def call(self, x, training = True):
        padding_mask = create_mask(x)
        looking_ahead_mask = self.create_look_ahead_mask(x, padding_mask)
        x = self.embeddingLayer(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += tf.convert_to_tensor(self.positionalEncodingLayer, dtype=tf.float32) #PE -> (1,input_seq_len,d_model)
        x = self.dropout(x, training=training)
        
        all_weights = []    
        for i in range(self.num_decoders):
            x,decoder_weights = self.decoderStack[i](x, x, x, looking_ahead_mask, training = training) # v, k, q 
            all_weights.append(decoder_weights)

        x = self.ffLayer(x)
        return x, all_weights   # (batch_size, input_seq_len, d_model)

    
    def create_look_ahead_mask(self, seq, padding_mask):
        mask = tf.cast(tf.constant(np.triu(np.ones(seq.shape[1]), k=1)), dtype = tf.float32)
        looking_ahead_mask = tf.maximum(padding_mask, mask)
        return looking_ahead_mask

    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dff': self.dff,
            'act_func':self.act_func,
            'num_heads': self.num_heads,
            'dropout_rate': self.rate,
            'vocab_size': self.smiles_vocab_size,
            'smiles_len': self.smiles_len,       
            'num_decoders':self.num_decoders,
            'embeddingLayer':self.embeddingLayer,
            'positionalEncodingLayer':self.positionalEncodingLayer,
            'dropout':self.dropout,
            'decoderStack':self.decoderStack,
            'ffLayer':self.ffLayer})
        return config
    
