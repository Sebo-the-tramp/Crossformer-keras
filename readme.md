🚧 🏗 UNOFFICIAL CROSSFORMER implementation in KERAS - Still a work in progress!! 🚧 🏗

# CrossFormer-Keras

## Introduction

Welcome, brave souls, to the humble abode of CrossFormer-Keras, where PyTorch tensors turn into Keras tensors as if by magic (but actually through a lot of blood, sweat, and desperate debugging sessions). This repository is my journey of rewriting the CrossFormer, a state-of-the-art transformer model that excels in handling long-range dependencies, from PyTorch to Keras. Why, you ask? Because sometimes in life, we just want to watch the world burn. Or maybe because I enjoy pain. Either way, here we are.

## Why Keras?
You might wonder, "Why Keras?" Well, every once in a while, a person decides to climb Everest because it's there. Rewriting this in Keras seemed equally challenging and unreasonable, so it had to be done. Plus, TensorFlow said I couldn't, and I'm nothing if not petty.

## Joke aside
It was written for a project, I hope it can be helpful to someone when it will start working!

-> refer to readme_OLD.md for any other thing (technical)

## PROGRESS

### 12.04.2024


I was able to make it work. 

Now the number of parameters coincide with torch.summary() from the original implementation:

PYTORCH IMPLEMENTATION:

```bash
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
├─DSW_embedding: 1-1                               [-1, 7, 28, 256]          --
|    └─Linear: 2-1                                 [-1, 256]                 1,792
├─LayerNorm: 1-2                                   [-1, 7, 28, 256]          512
├─Encoder: 1-3                                     [-1, 7, 28, 256]          --
|    └─ModuleList: 2                               []                        --
|    |    └─scale_block: 3-1                       [-1, 7, 28, 256]          1,389,056
|    |    └─scale_block: 3-2                       [-1, 7, 14, 256]          1,485,568
|    |    └─scale_block: 3-3                       [-1, 7, 7, 256]           1,467,648
├─Decoder: 1-4                                     [-1, 24, 7]               --
|    └─ModuleList: 2                               []                        --
|    |    └─DecoderLayer: 3-4                      [-1, 7, 4, 256]           1,724,934
|    |    └─DecoderLayer: 3-5                      [-1, 7, 4, 256]           1,724,934
|    |    └─DecoderLayer: 3-6                      [-1, 7, 4, 256]           1,724,934
|    |    └─DecoderLayer: 3-7                      [-1, 7, 4, 256]           1,724,934
====================================================================================================
```

KERAS IMPLEMENTATION:

```bash
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 DSW_embedding (DSW_embeddin  multiple                 1792      
 g)                                                              
                                                                 
 layer_normalization (LayerN  multiple                 512       
 ormalization)                                                   
                                                                 
 encoder (Encoder)           multiple                  4342272   
                                                                 
 decoder_base (Decoder)      multiple                  6899736   
                                                                 
=================================================================
Total params: 11,301,656
Trainable params: 11,301,656
Non-trainable params: 0
_________________________________________________________________
```

What I found out is that with my current environment, the model is not running on the GPU, this is why is very slow and that's why I will need to change some packages.

Also there is some error in the number of steps calculated by the keras ```.fit()``` operation.

See you tomorrow!

### 13.04.2024
