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
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.batch_size = 32

        self.baseline = baseline

        self.device = device

        # for handling input in the call()
        self.flag_input_dict=True
        self.flag_y_vector=True


        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len # ceil goes to upper integer
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        print("UOUOUOUOUOUOUOUO", self.pad_in_len, self.pad_out_len, self.in_len_add)

        # Embedding
        # input_shape = (32, data_dim, in_len,)        
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = tf.Variable(tf.random.normal([1, data_dim, (self.pad_in_len // seg_len), d_model], dtype=tf.float64), dtype=tf.float64, trainable=True, name="enc_pos_embedding")
        print("ENC POS EMBEDDING", tf.shape(self.enc_pos_embedding), self.enc_pos_embedding.dtype)
        self.pre_norm = LayerNormalization(epsilon=0.001)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor)
        
        #mandare dentro roba nel self encode
        
        # Decoder
        self.dec_pos_embedding = tf.Variable(tf.random.normal([1, data_dim, (self.pad_out_len // seg_len), d_model], dtype=tf.float64), dtype=tf.float64, trainable=True, name="dec_pos_embedding")
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor)

        print("ASDASDASDASD", tf.shape(self.dec_pos_embedding))
        

    def call(self, x_seq):

        ### READ DICT IN INPUT ####

        input_X = Input(shape = [self.in_len, self.data_dim], name = "input_X", batch_size=self.batch_size, dtype=tf.float64)
        # for now
        if self.flag_input_dict:
            if (self.flag_y_vector):
                input_y = Input((2,), name="input_y", dtype=tf.float64)
            else:
                input_y = Input((), name="input_y", dtype=tf.float64)

            input_dataset = Input((), name="input_dataset", dtype = tf.int64)
            input_person = Input((), name="input_person", dtype = tf.int64)

        print("shape input_X", input_X)
        print("shape input_y",input_y)

        x_seq = input_X
        print(tf.shape(x_seq))
        ### END READING ###

        if self.baseline:
            base = tf.reduce_mean(x_seq, axis=1, keepdims=True, dtype=tf.float64)
        else:
            base = tf.zeros_like(x_seq[:, :self.out_len, :], dtype=tf.float64)
        batch_size = x_seq.shape[0]
        # Handling input sequence padding
        if self.in_len_add != 0:
            padding = tf.tile(x_seq[:, :1, :], [1, self.in_len_add, 1], dtype=tf.float64)
            x_seq = tf.concat([padding, x_seq], axis=1, dtype=tf.float64)    

        print("CHECK dtype", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype)

        x_seq = self.enc_value_embedding(x_seq)     
        print("CHECK dtype", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype)
        x_seq = x_seq + self.enc_pos_embedding        
        print("CHECK dtype", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype)
        x_seq = self.pre_norm(x_seq)        

        print("CHECK dtype encoding", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype)
        enc_out = self.encoder(x_seq)              

        print("CHECK dtype", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype)

        #dec_in = tf.tile(self.dec_pos_embedding, [batch_size, 1, 1, 1])
        dec_in = tf.repeat(self.dec_pos_embedding, repeats=batch_size, axis=0)
        print("CHECK dtype0", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype)
        predict_y = self.decoder(dec_in, enc_out)        
        print("CHECK dtype1", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype, predict_y.dtype, base.dtype)


        ### HERE SHOULD BE FINE
                
        before_return = base + predict_y[:, :self.out_len, :]
        print("CHECK dtype2", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype, predict_y.dtype, before_return.dtype)

        just_before = tf.reshape(before_return, [32, 54])
        # before_return = before_return[:,:,:2]
        
        a = just_before[:,:2]
        print("CHECK dtype3", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype, predict_y.dtype, just_before.dtype)
        print("CHECK dtype4", x_seq.dtype, self.enc_value_embedding.dtype, self.enc_pos_embedding.dtype, self.pre_norm.dtype, predict_y.dtype, a.dtype)

        return a

    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_len, self.data_dim)