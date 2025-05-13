from seal import *
import math
import numpy as np


from uni_henn.utils.context import Context
from uni_henn.utils.structure import Output, Cuboid

def batchNormalisation(context: Context, In: Output, Img: Cuboid, layer, data_size) : 
    
    Gamma = layer.weight.detach().tolist()
    Beta = layer.bias.detach().tolist()
    Mean = layer.running_mean.detach().tolist()
    Variance = layer.running_var.detach().tolist()

    featureNumber = layer.num_features

    Out = Output(
        ciphertexts = [], 
        size = Cuboid(
            length = layer.num_features,
            height = In.size.h,
            width = In.size.w
        ), 
        interval = In.interval, 
        const = 1
    )

    for f in range(featureNumber):
        Outputs = In.ciphertexts[f]

        varianceVector = [Variance[f]]*context.number_of_slots 
        plaintextVariance = context.encoder.encode(varianceVector, Outputs.scale())
        context.evaluator.mod_switch_to_inplace(plaintextVariance, Outputs.parms_id())
        Outputs = context.evaluator.sub_plain(Outputs, plaintextVariance)
        
        gammaMeanVector = [Gamma[f]/Mean[f]]*context.number_of_slots
        plaintextGammaMean= context.encoder.encode(gammaMeanVector, Outputs.scale())
        context.evaluator.mod_switch_to_inplace(plaintextGammaMean, Outputs.parms_id())
        Outputs = context.evaluator.multiply_plain(Outputs, plaintextGammaMean)
        context.evaluator.relinearize_inplace(Outputs, context.relin_keys)
        context.evaluator.rescale_to_next_inplace(Outputs)
        
        betaVector = [Beta[f]]*context.number_of_slots 
        plaintextBeta = context.encoder.encode(betaVector, Outputs.scale())
        context.evaluator.mod_switch_to_inplace(plaintextBeta, Outputs.parms_id())
        Outputs = context.evaluator.add_plain(Outputs, plaintextBeta)

        Out.ciphertexts.append(Outputs)

    return Out
        



