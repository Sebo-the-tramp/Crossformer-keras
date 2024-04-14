import numpy as np

# import keras
# from keras import layers
import tensorflow as tf

# from utils.tools import EarlyStopping, adjust_learning_rate

from crossformerkeras.cross_models.cross_former_keras import CrossformerKeras

from crossformerkeras.cross_exp.exp_basic_keras import Exp_Basic

# from data.data_loader import Dataset_MTS
# from torch.utils.data import DataLoader

class Exp_crossformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_crossformer, self).__init__(args)

    def _build_model(self):
        model = CrossformerKeras(
            self.args.data_dim,
            self.args.in_len,
            self.args.out_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model,
            self.args.d_ff,
            self.args.n_heads,
            self.args.e_layers,
            self.args.dropout,
            self.args.baseline,
            self.device
        )

        # this fixes the batch_size problem        
        # dummy_input = tf.zeros((self.args.batch_size, self.args.in_len, self.args.data_dim))
        # Create a dummy input array matching the input specifications of your model
        # Note: Adjust the shape according to your model's expected input
        dummy_input = np.random.random((32, model.in_len, model.data_dim)).astype('float32')  # batch_size, in_len, data_dim

        # Pass the dummy data through the model to build it
        out = model(tf.constant(dummy_input))
        print("outououou", out.shape)        

        print(model.summary())

        return model