import keras

import tensorflow as tf
from keras import layers

from einops import rearrange, repeat
from crossformerkeras.cross_models.attn_keras import FullAttention, AttentionLayer, TwoStageAttentionLayer

class DecoderLayer(layers.Layer):
    '''
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    '''
    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num = 10, factor = 10, name="change_me"):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, \
                                d_ff, dropout)    
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout = dropout, name="cross_attention" + name)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)
        self.MLP1 = tf.keras.Sequential([layers.Dense(d_model), layers.Activation('gelu'), layers.Dense(d_model)],  name="sequential_MLP1_Decoder" + name)
        self.linear_pred = layers.Dense(seg_len)
    

    def call(self, x, cross):
        '''
        x: the output of last decoder layer
        cross: the output of the corresponding encoder layer
        '''        

        # print("##"*10, "Bath", x.shape)        
        # print("##"*10, "Cross", cross.shape)

        batch_size, ts_d, out_seg_num, d_model = x.shape
        # print("DECODER stuff", batch_size, ts_d, out_seg_num, d_model)
        
        # # print("##"*10, "Bath", x.shape)
        x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')

        
        tmp = self.cross_attention(
            x, cross, cross,
        )
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x+y)
        
        # d_batch_ts_d, d_in_seg_num, d_d_model = dec_output.shape
        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', ts_d=ts_d)
        # dec_output = tf.reshape(dec_output, [batch_size, d_batch_ts_d//batch_size, d_in_seg_num, d_d_model])
        # # print(dec_output.shape)
        layer_predict = self.linear_pred(dec_output)
        # # print(layer_predict.shape)
        l_b, l_out_d, l_seg_num, l_seg_len = layer_predict.shape
        # layer_predict = tf.reshape(layer_predict, [l_b, l_out_d*l_seg_num, l_seg_len])
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')
        

        return dec_output, layer_predict
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    
class Decoder(layers.Layer):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout,\
                out_seg_num, factor, router=False):
        super(Decoder, self).__init__(name="decoder_base")

        self.router = router
        self.decode_layers = []
        for i in range(d_layers):            
            self.decode_layers.append(DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, \
                                        out_seg_num, factor, name="decoder_layer_keras_"+str(i)))

    def build(self, input_shape):
        # Explicitly building the decoder layers with an expected input shape
        # This line ensures that all nested layers have their weights created, assuming input_shape is correctly provided
        # # print("###"*10, "AM I CALLED?")
        for layer in self.decode_layers.layers:
            layer.build(input_shape)
        super(Decoder, self).build(input_shape)  # Mark the layer as built

    def call(self, x, cross):
        final_predict = None
        i = 0
        
        ts_d = x.shape[1]        
        for layer in self.decode_layers:
            # print("mhmm", i)
            cross_enc = cross[i]
            # # print("##"*10, "Cross Enc", cross_enc.shape)

            x, layer_predict = layer(x,  cross_enc)

            # # print(x.shape,layer_predict.shape)

            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
        

        # print("DONE")

        f_b, f_out_d_f_seg_num, f_seg_len = tf.shape(final_predict)[0], tf.shape(final_predict)[1], tf.shape(final_predict)[2]
        f_out_d = ts_d
        f_seg_num = f_out_d_f_seg_num//f_out_d
        final_predict = tf.reshape(final_predict, [f_b, f_out_d, f_seg_num, f_seg_len])
        final_predict = tf.transpose(final_predict, [0, 2, 3, 1])
        final_predict = tf.reshape(final_predict, [f_b, f_seg_num * f_seg_len, f_out_d]) 

        # print("FINAL", final_predict.shape)               

        return final_predict
    
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], input_shape[1], input_shape[2])