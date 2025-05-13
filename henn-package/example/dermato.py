from uni_henn import *
from models.model_structures import M8, Lenet5

from seal import *
import numpy as np
import torch

import os

import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader


if __name__ == "__main__":

    class Flatten(torch.nn.Module):
        def forward(self, x):
            return torch.flatten(x, 1)

    def _approximated_ReLU(x):
        return 0.117071 * x**2 + 0.5 * x + 0.375373

    class ApproxReLU(torch.nn.Module):
        def forward(self, x):
            return _approximated_ReLU(x)
    
    class ResidualBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            
            # Store attributes
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.stride = stride
            
            # Main path
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False)
            # self.bn1 = nn.BatchNorm2d(out_channels)  # Normalize activations
            self.relu1 = ApproxReLU()

            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                stride=1, padding=1, bias=False)
            # self.bn2 = nn.BatchNorm2d(out_channels)

            # Skip connection with downsampling if needed
            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                            stride=stride, bias=False),  # Kernel=1 to preserve spatial dimensions
                    # nn.BatchNorm2d(out_channels)
                )

            self.relu2 = ApproxReLU()

        def forward(self, x):
            identity = x

            # Main path
            out = self.conv1(x)
            # out = self.bn1(out)
            out = self.relu1(out)

            out = self.conv2(out)
            # out = self.bn2(out)

            # Adjust identity shape if needed
            if self.downsample is not None:
                identity = self.downsample(x)
            
            # Add skip connection
            out += identity
            out = self.relu2(out)
            
            return out


    current_dir = os.path.dirname(os.path.abspath(__file__))

    m5_model = Lenet5()
    m5_model = torch.load(current_dir + '/models/dermato_model.pth', map_location=torch.device('cpu'))

    CIFAR10_Img = Cuboid(3, 28, 28)

    context = sys.argv[1]

    HE_m5 = HE_CNN(m5_model, CIFAR10_Img, context)
    
    print("the data size :", HE_m5.data_size)
    num_of_data = int(context.number_of_slots // HE_m5.data_size)

    
    data_flag = 'dermamnist'

    download = False


    info = INFO[data_flag]

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

    print("the batch size :", num_of_data)
    print('*'*30)

    data, _label = next(iter(test_loader))
    _label = _label.tolist()

    ppData = preprocessing(np.array(data), CIFAR10_Img, num_of_data, HE_m5.data_size)

    ciphertext_list = HE_m5.encrypt(ppData)
    
    result_ciphertext = HE_m5(ciphertext_list, _time=True)

    result_plaintext = HE_m5.decrypt(result_ciphertext)

    
    for i in range(num_of_data):
        """Model result without homomorphic encryption"""
        origin_results = m5_model(data)[i].flatten().tolist()
        origin_result = origin_results.index(max(origin_results))
        """Model result with homomorphic encryption"""
        he_result = -1
        MIN_VALUE = -1e10
        sum = 0
        for idx in range(7):
            he_output = result_plaintext[idx + HE_m5.data_size*i]

            sum = sum + np.abs(origin_results[idx] - he_output)

            if(MIN_VALUE < he_output):
                MIN_VALUE = he_output
                he_result = idx

        """
        After calculating the sum of errors between the results of the original model and the model with homomorphic encryption applied, Outputting whether it matches the original results.
        """        
        print('%sth result Error: %.8f\t| Result is %s' %(str(i+1), sum, "Correct" if origin_result == he_result else "Wrong"))