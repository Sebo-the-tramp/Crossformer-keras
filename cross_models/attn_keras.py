from einops import rearrange, repeat
import numpy as np

from math import sqrt

import keras
from keras import layers
# import tensorflow as tf
from utils.common_settings import *


class FullAttention(layers.Layer):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__(name="Full_attention")
        self.scale = scale
        self.dropout = layers.Dropout(attention_dropout)
        
    def call(self, queries, keys, values):
        B, L, H, E = queries.shape
        # _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)
        # print("QUERIES", queries.shape, keys.shape, values.shape)

        scores = tf.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(tf.keras.backend.softmax(scale * scores, axis=-1))
        V = tf.einsum("bhls,bshd->blhd", A, values)
        
        return V
    

class AttentionLayer(layers.Layer):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1, name="general"):
        super(AttentionLayer, self).__init__(name=name)

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = layers.Dense(d_keys * n_heads)
        self.key_projection = layers.Dense(d_keys * n_heads)
        self.value_projection = layers.Dense(d_values * n_heads)
        self.out_projection = layers.Dense(d_model)
        self.n_heads = n_heads

    def call(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        queries = rearrange(queries, 'b l (h d) -> b l h d', h = H)
        
        # print("QUERIES FULL ATTENTION", self.name, queries.shape, tf.shape(queries), keys.shape, values.shape)
        keys = self.key_projection(keys)
        keys = rearrange(keys, 'b s (h d) -> b s h d', h = H)
        values = self.value_projection(values)
        values = rearrange(values, 'b s (h d) -> b s h d', h = H)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = rearrange(out, 'b l h d -> b l (h d)')
        # # print("DONE", out.shape)

        return self.out_projection(out)

class TwoStageAttentionLayer(layers.Layer):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff = None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__(name="TSA_layer")
        d_ff = d_ff or 4*d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout = dropout, name="time_attention")
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout = dropout, name="dim_sender")
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout = dropout, name="dim_receiver")
        
        self.router = self.add_weight(name='router', shape=(seg_num, factor, d_model), initializer='random_normal', trainable=True, dtype=tf.float64)
        
        self.dropout = layers.Dropout(dropout)

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()
        self.norm4 = layers.LayerNormalization()

        self.MLP1 = tf.keras.Sequential([layers.Dense(d_ff), layers.Activation('gelu'), layers.Dense(d_model)], name="sequential_MLP1_2SA")
        self.MLP2 = tf.keras.Sequential([layers.Dense(d_ff), layers.Activation('gelu'), layers.Dense(d_model)], name="sequential_MLP2_2SA")

    def call(self, x):  
        #Cross Time Stage:
        #  Directly apply MSA to each dimension
        _, ts_d, seg_num = x.shape[0], x.shape[1], x.shape[2]
        # if(batch is None):
        #     batch = -1
        batch = tf.shape(x)[0]
        # print("BATCH", batch, ts_d, seg_num)

        # # print("TWO STAGE ATTENTION STARTED")
        # print("TIME ATTENTION", x.shape)
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        
        # # print(time_in.shape)
        # # print(self.dropout(time_enc).shape)

        dim_in = time_in + self.dropout(time_enc)
        # # print(dim_in.shape)
        dim_in = self.norm1(dim_in)
        # # print("POIII",dim_in.shape)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # better to use tf.convert_to_tensor to convert the tensor to a tensor        

        # print("DIMENSIONS ATTENTION", x.shape)

        #Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', ts_d = ts_d, seg_num = seg_num)
        batch_router = tf.repeat(self.router, repeats = batch, axis = 0)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', seg_num=seg_num)

        # print("AFTRE TWO STAGE")

        return final_out
