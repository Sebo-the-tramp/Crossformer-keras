import keras

from keras import layers

from crossformerkeras.cross_models.attn_keras import FullAttention, AttentionLayer, TwoStageAttentionLayer

from math import ceil

import tensorflow as tf

class SegMerging(layers.Layer):
    '''
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    '''
    def __init__(self, d_model, win_size, norm_layer=tf.keras.layers.LayerNormalization,name="SegMerging"):
        super(SegMerging, self).__init__(name=name)
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = layers.Dense(d_model)
        self.norm = norm_layer(axis=-1)

    def call(self, x):
        # print(tf.executing_eagerly())        
        batch_size, ts_d, seg_num, d_model = x.shape

        pad_num = seg_num % self.win_size        

        if pad_num != 0:
            pad_num = self.win_size - pad_num
            paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, pad_num], [0, 0]])
            x = tf.pad(x, paddings, "CONSTANT")        

        #this is correct!!

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = tf.concat(seg_to_merge, axis=-1)                

        x = self.norm(x)
        x = self.linear_trans(x)
        # print(self.win_size, self.d_model)        

        return x
    

class scale_block(layers.Layer):
    '''
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    '''
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, \
                    seg_num = 10, factor=10):
        super(scale_block, self).__init__(name='scale_block')

        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, layers.LayerNormalization)
        else:
            self.merge_layer = None
        
        self.encode_layers = []

        self.depth = depth

        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, \
                                                        d_ff, dropout))
    
    def call(self, x):
        _, ts_dim, _, _ = x.shape        

        if self.merge_layer is not None:               
            x = self.merge_layer(x)                

        for layer in self.encode_layers:
            # print("During")
            x = layer(x)        
    
        return x
    

class Encoder(keras.Model):
    '''
    The Encoder of Crossformer.
    '''
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout,
                in_seg_num = 10, factor=10):
        super(Encoder, self).__init__(name="encoder")
        self.encode_blocks = []        

        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, block_depth, dropout,\
                                            in_seg_num, factor))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(win_size, d_model, n_heads, d_ff, block_depth, dropout,\
                                            ceil(in_seg_num/win_size**i), factor))        

    def call(self, x):
        encode_x = []
        encode_x.append(x)        
        
        for i, block in enumerate(self.encode_blocks):                          
            x = block(x)                                                
            encode_x.append(x)            
            
        return encode_x    
    
    def compute_output_shape(self, input_shape):
        return input_shape