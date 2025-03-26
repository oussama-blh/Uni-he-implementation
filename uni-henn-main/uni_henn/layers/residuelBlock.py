from seal import *
import math
import numpy as np
import torch.nn as nn

from .activation import approximated_ReLU_converter
from .convolutional import conv2d_layer_converter_

from uni_henn.utils.context import Context
from uni_henn.utils.structure import Output, Cuboid

    
def residuelBlock(context: Context, In: Output, Img: Cuboid, layer, data_size):
    
    identity = In
    Out = In 

    # Loop through the submodules inside the ResidualBlock
    for sub_layer_name, sub_layer in layer.named_children():
        
        if isinstance(sub_layer, nn.Conv2d):
            Out = conv2d_layer_converter_(context, Out, Img, sub_layer, data_size)

        if sub_layer.__class__.__name__ == "ApproxReLU":
            Out = approximated_ReLU_converter(context, Out)

        if sub_layer_name == 'downsample' :
            for sub_sub_layer in sub_layer:
                if isinstance(sub_sub_layer, nn.Conv2d):
                    identity = conv2d_layer_converter_(context, In, Img,sub_sub_layer, data_size)

                if sub_layer.__class__.__name__ == "ApproxReLU":
                    identity = approximated_ReLU_converter(context, Out)
    
            for i in range(layer.out_channels):
                
                context.evaluator.mod_switch_to_inplace(identity.ciphertexts[i], Out.ciphertexts[i].parms_id())

                identity.ciphertexts[i].scale(Out.ciphertexts[i].scale())

                Out.ciphertexts[i] = context.evaluator.add(identity.ciphertexts[i], Out.ciphertexts[i])

    return Out