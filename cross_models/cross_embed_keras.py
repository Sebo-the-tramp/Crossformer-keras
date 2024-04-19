import keras
from keras import layers
from einops import rearrange

class DSW_embedding(layers.Layer):
    def __init__(self, seg_len, d_model, seg_num):
        super(DSW_embedding, self).__init__(name="DSW_embedding")
        self.seg_len = seg_len
        self.seg_num = seg_num

        self.linear = layers.Dense(d_model)

    def call(self, x, enc_embed):
        batch, ts_len, ts_dim = x.shape
        # if(batch is None):
        #     batch = -1

        # print("BAAATCH", batch, ts_len, ts_dim)

        # print("EMBEDIGIIININ", x.shape)        
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        # print(x_embed.shape)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', seg_num=self.seg_num, d = ts_dim)
        # print(x_embed.shape)
        return x_embed + enc_embed

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)

