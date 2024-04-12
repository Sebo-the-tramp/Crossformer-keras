import keras

from keras import layers

import tensorflow as tf

from cross_models.cross_encoder_keras import Encoder
from cross_models.cross_decoder_keras import Decoder
from cross_models.attn_keras import FullAttention, AttentionLayer, TwoStageAttentionLayer
from cross_models.cross_embed_keras import DSW_embedding

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input

from einops import rearrange, repeat

from math import ceil


class CrossformerKeras(keras.Model):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, baseline = False, device='cuda:0'):
        super(CrossformerKeras, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size

        self.baseline = baseline

        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len # ceil goes to upper integer
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = layers.Embedding(input_dim = (self.pad_in_len // seg_len), output_dim = d_model)
        self.pre_norm = layers.LayerNormalization()

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = tf.Variable(tf.random.normal([1, data_dim, (self.pad_out_len // seg_len), d_model]), trainable=True)
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor)
        

    def call(self, x_seq):
        if self.baseline:
            base = tf.reduce_mean(x_seq, axis=1, keepdims=True)
        else:
            base = tf.zeros_like(x_seq[:, :self.out_len, :])
        batch_size = x_seq.shape[0]
        # Handling input sequence padding
        if self.in_len_add != 0:
            padding = tf.tile(x_seq[:, :1, :], [1, self.in_len_add, 1])
            x_seq = tf.concat([padding, x_seq], axis=1)

        x_seq = self.enc_value_embedding(x_seq)

        # print("##"*10, "0", tf.shape(x_seq))
        
        # Positional embeddings
        positions = tf.range(start=0, limit=self.pad_in_len // self.seg_len, delta=1)
        x_seq += self.enc_pos_embedding(positions)

        # print("##"*20, "1", tf.shape(x_seq))

        x_seq = self.pre_norm(x_seq)

        # print("##"*20, "1.5")

        enc_out = self.encoder(x_seq)
        # print("Done 1.5")

        # print("##"*10, "2")

        #dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        expanded_embeddings = tf.expand_dims(self.dec_pos_embedding, axis=0)  # Shape: [1, input_length, embedding_dim]

        dec_in = tf.tile(self.dec_pos_embedding, [batch_size, 1, 1, 1])
        predict_y = self.decoder(dec_in, enc_out)

        return base + predict_y[:, :self.out_len, :]

    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_len, self.data_dim)