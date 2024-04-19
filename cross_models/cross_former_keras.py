import tensorflow as tf

from tensorflow.keras.layers import Layer, LayerNormalization, Dense, Dropout, LayerNormalization, Dropout

# import tensorflow as tf
from utils.common_settings import *

from crossformerkeras.cross_models.cross_encoder_keras import Encoder
from crossformerkeras.cross_models.cross_decoder_keras import Decoder
from crossformerkeras.cross_models.attn_keras import FullAttention, AttentionLayer, TwoStageAttentionLayer
from crossformerkeras.cross_models.cross_embed_keras import DSW_embedding

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input

from math import ceil

from common_settings import * # here all the important modules are loaded -> Input() also I hope

class CrossformerKeras(keras.Model):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, baseline = False, device='cuda:0', name="base"):
        super(CrossformerKeras, self).__init__(name=name)
        self.clf = network
        

    def call(self, x_seq):

        ### READ DICT IN INPUT ####

        input_X = Input(shape = [self.in_len, self.data_dim], name = "input_X", batch_size=self.batch_size, dtype=tf.float16)
        # for now
        if self.flag_input_dict:
            if (self.flag_y_vector):
                input_y = Input((2,), name="input_y", dtype=tf.float16)
            else:
                input_y = Input((), name="input_y", dtype=tf.float16)

            input_dataset = Input((), name="input_dataset", dtype = tf.int64)
            input_person = Input((), name="input_person", dtype = tf.int64)

        # print("shape input_X", input_X)
        # print("shape input_y",input_y)

        x_seq = input_X
        # print(tf.shape(x_seq))
        ### END READING ###

        if self.baseline:
            base = tf.reduce_mean(x_seq, axis=1, keepdims=True, dtype=tf.float16)
        else:
            base = tf.zeros_like(x_seq[:, :self.out_len, :], dtype=tf.float16)
        batch_size = x_seq.shape[0]
        # Handling input sequence padding
        if self.in_len_add != 0:
            padding = tf.tile(x_seq[:, :1, :], [1, self.in_len_add, 1], dtype=tf.float16)
            x_seq = tf.concat([padding, x_seq], axis=1, dtype=tf.float16)    

        # print("CHECK dtype", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype)

        x_seq = self.enc_value_embedding(x_seq)     
        # print("CHECK dtype", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype)
        x_seq = x_seq + self.enc_pos_embedding        
        # print("CHECK dtype", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype)
        x_seq = self.pre_norm(x_seq)        

        # print("CHECK dtype encoding", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype)
        enc_out = self.encoder(x_seq)              

        # print("CHECK dtype", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype)

        #dec_in = tf.tile(self.dec_pos_embedding, [batch_size, 1, 1, 1])
        dec_in = tf.repeat(self.dec_pos_embedding, repeats=batch_size, axis=0)
        # print("CHECK dtype0", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype)
        predict_y = self.decoder(dec_in, enc_out)        
        # print("CHECK dtype1", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype, predict_y.dtype, base.dtype)


        ### HERE SHOULD BE FINE
                
        before_return = base + predict_y[:, :self.out_len, :]
        print("CHECK dtype2", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype, predict_y.dtype, before_return.dtype)

        just_before = tf.reshape(before_return, [32, 54])
        # before_return = before_return[:,:,:2]
        
        a = just_before[:,:2]
        # print("CHECK dtype3", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype, predict_y.dtype, just_before.dtype)
        # print("CHECK dtype4", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype, predict_y.dtype, a.dtype)

        return a

    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_len, self.data_dim)