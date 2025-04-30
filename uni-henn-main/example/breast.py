from uni_henn import *
from seal import *
import numpy as np
import onnxruntime as ort
import os

import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
import onnx

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load ONNX model
onnx_model = onnx.load(current_dir + '/models/ONNXModels/breast_resNet.onnx')
session = ort.InferenceSession(current_dir + '/models/ONNXModels/breast_resNet.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

CIFAR10_Img = Cuboid(1, 28, 28)
context = sys.argv[1]
HE_m5 = HE_CNN(onnx_model, CIFAR10_Img, context)
max_data_capacity = int(context.number_of_slots // HE_m5.data_size)

data_flag = 'breastmnist'
info = INFO[data_flag]
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])
data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
test_dataset = DataClass(split='test', transform=data_transform, download=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=max_data_capacity, shuffle=False)

print("*" * 30)

# Total counters
total_samples = 0
correct_onnx = 0
correct_he = 0
i_batch  = 0


for data, _label in test_loader:

        i_batch = i_batch + 1
        print("batch number :",i_batch)

        _label = _label.tolist()
        total_samples += len(_label)
        
        num_of_data = min(max_data_capacity, int(data.size(0)))

        ppData = preprocessing(np.array(data), CIFAR10_Img, num_of_data, HE_m5.data_size)
        ciphertext_list = HE_m5.encrypt(ppData)
        result_ciphertext = HE_m5(ciphertext_list, _time=True)
        result_plaintext = HE_m5.decrypt(result_ciphertext)

        data_numpy = data.numpy()
        origin_outputs = session.run([output_name], {input_name: data_numpy})[0]

        for i in range(len(_label)):
            true_label = _label[i][0] if isinstance(_label[i], list) else _label[i]

            origin_results = origin_outputs[i].flatten().tolist()
            origin_result = origin_results.index(max(origin_results))
            if origin_result == true_label:
                correct_onnx += 1

            he_result = -1
            MIN_VALUE = -1e10
            for idx in range(n_classes):
                he_output = result_plaintext[idx + HE_m5.data_size * i]
                if MIN_VALUE < he_output:
                    MIN_VALUE = he_output
                    he_result = idx

            if he_result == true_label:
                correct_he += 1

# Accuracy display
print("\n" + "*" * 30)
print(f"Total Samples: {total_samples}")
print(f"ONNX Model Accuracy: {100 * correct_onnx / total_samples:.2f}%")
print(f"HE Model Accuracy: {100 * correct_he / total_samples:.2f}%")