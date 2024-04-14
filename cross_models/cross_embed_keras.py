import keras
from keras import layers
from einops import rearrange

class DSW_embedding(layers.Layer):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__(name="DSW_embedding")
        self.seg_len = seg_len

        self.linear = layers.Dense(d_model)

    def call(self, x):
        batch, ts_len, ts_dim = x.shape
        # print(x.shape)
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        # print(x_embed.shape)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        
        return x_embed

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)

