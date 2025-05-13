from uni_henn import *
from models.model_structures import M8, Net

from seal import *
from torchvision import datasets
import numpy as np
import torch
import onnxruntime as ort
import os

import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
import onnx

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # m5_model = M8()
    # m5_model = torch.load(current_dir + '/models/pathologie_model.pth', map_location=torch.device('cpu'))

    # Load the ONNX model
    onnx_model = onnx.load(current_dir + '/models/ONNXModels/breast_resNet2.onnx')


    # Create an ONNX Runtime session
    session = ort.InferenceSession(current_dir + '/models/ONNXModels/breast_resNet2.onnx')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    CIFAR10_Img = Cuboid(1, 28, 28)

    context = sys.argv[1]

    HE_m5 = HE_CNN(onnx_model, CIFAR10_Img, context)

    num_of_data = int(context.number_of_slots // HE_m5.data_size)

    
    data_flag = 'breastmnist'

    download = False

    info = INFO[data_flag]
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # Preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Load datasets (already split)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    # Encapsulate datasets into DataLoader
    test_loader = DataLoader(dataset=test_dataset, batch_size=num_of_data, shuffle=True)

    print ("*"*30)

    data, _label = next(iter(test_loader))
    _label = _label.tolist()

    ppData = preprocessing(np.array(data), CIFAR10_Img, num_of_data, HE_m5.data_size)

    ciphertext_list = HE_m5.encrypt(ppData) 
    
    result_ciphertext = HE_m5(ciphertext_list, _time=True)

    result_plaintext = HE_m5.decrypt(result_ciphertext)

    # Convert to numpy for ONNX Runtime
    data_numpy = data.numpy()
        
    # Run inference with ONNX Runtime
    origin_outputs = session.run([output_name], {input_name: data_numpy})[0]


    for i in range(num_of_data):
        """Model result without homomorphic encryption"""
        origin_results = origin_outputs[i].flatten().tolist()
        origin_result = origin_results.index(max(origin_results))
        
        """Model result with homomorphic encryption"""
        he_result = -1
        MIN_VALUE = -1e10
        sum_error = 0
        
        for idx in range(n_classes):
            he_output = result_plaintext[idx + HE_m5.data_size*i]
            sum_error = sum_error + np.abs(origin_results[idx] - he_output)
            
            if MIN_VALUE < he_output:
                MIN_VALUE = he_output
                he_result = idx
        
        """ After calculating the sum of errors between the results of the original model 
        and the model with homomorphic encryption applied, 
        Outputting whether it matches the original results. """
        print('%sth result Error: %.8f\t| Result is %s' % (str(i+1), sum_error, "Correct" if origin_result == he_result else "Wrong"))