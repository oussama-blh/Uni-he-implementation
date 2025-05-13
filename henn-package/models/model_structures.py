import torch

def _approximated_ReLU(x):
    return 0.117071 * x**2 + 0.5 * x + 0.375373

class Square(torch.nn.Module):
    def forward(self, x):
        return x**2

class ApproxReLU(torch.nn.Module):
    def forward(self, x):
        return _approximated_ReLU(x)

class Flatten(torch.nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)

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
    
    def forward(self, x):
        identity = x

        # Main path
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        # Adjust identity shape if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection
        out += identity
        out = self.relu(out)
        
        return out
    

class M1(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(M1, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=3, padding=0)
        self.Square1 = Square()
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(9 * 9 * 8, 64)
        self.Square2 = Square()
        self.FC2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Square1(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        out = self.Square2(out)
        out = self.FC2(out)
        return out

class M2(torch.nn.Module):
    def __init__(self):
        super(M2, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0)
        self.Square1 = Square()
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size=2)
        self.Conv2 = torch.nn.Conv2d(in_channels=4, out_channels=12, kernel_size=5, stride=1, padding=0)
        self.Square2 = Square()
        self.AvgPool2 = torch.nn.AvgPool2d(kernel_size=2)
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(192, 10)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Square1(out)
        out = self.AvgPool1(out)
        out = self.Conv2(out)
        out = self.Square2(out)
        out = self.AvgPool2(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        return out

class M3(torch.nn.Module):
    def __init__(self):
        super(M3, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.ApproxReLU1 = ApproxReLU()
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size=2)
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(1014, 120)
        self.ApproxReLU2 = ApproxReLU()
        self.FC2 = torch.nn.Linear(120, 10)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.ApproxReLU1(out)
        out = self.AvgPool1(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        out = self.ApproxReLU2(out)
        out = self.FC2(out)
        return out

class M4(torch.nn.Module):
    def __init__(self, hidden=84, output=10):
        super(M4, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.Square1 = Square()
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size = 2)
        self.Conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.Square2 = Square()
        self.AvgPool2 = torch.nn.AvgPool2d(kernel_size = 2)
        self.Conv3 = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.Square3 = Square()
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(120, hidden)
        self.Square4 = Square()
        self.FC2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Square1(out)
        out = self.AvgPool1(out)
        out = self.Conv2(out)
        out = self.Square2(out)
        out = self.AvgPool2(out)
        out = self.Conv3(out)
        out = self.Square3(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        out = self.Square4(out)
        out = self.FC2(out)
        return out

class M5(torch.nn.Module):
    def __init__(self, output=10):
        super(M5, self).__init__()
        # L1 Image shape=(?, 32, 32, 1)
        #    Conv     -> (?, 30, 30, 16)
        #    Pool     -> (?, 15, 15, 16)
        self.Conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.Square1 = Square()
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size = 2)
        # L2 Image shape=(?, 15, 15, 16)
        #    Conv     -> (?, 12, 12, 64)
        #    Pool     -> (?, 6, 6, 64)
        self.Conv2 = torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=4, stride=1, padding=0)
        self.Square2 = Square()
        self.AvgPool2 = torch.nn.AvgPool2d(kernel_size = 2)
        # L2 Image shape=(?, 6, 6, 64)
        #    Conv     -> (?, 4, 4, 128)
        #    Pool     -> (?, 1, 1, 128)
        self.Conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.Square3 = Square()
        self.AvgPool3 = torch.nn.AvgPool2d(kernel_size = 4)
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(1*1*128, output)


    def forward(self, x):
        out = self.Conv1(x)
        out = self.Square1(out)
        out = self.AvgPool1(out)
        out = self.Conv2(out)
        out = self.Square2(out)
        out = self.AvgPool2(out)
        out = self.Conv3(out)
        out = self.Square3(out)
        out = self.AvgPool3(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        return out


class M6(torch.nn.Module):
    def __init__(self):
        super(M6, self).__init__()
        # L1 Image shape=(?, 16, 16, 1)
        #    Conv     -> (?, 14, 14, 6)
        #    Pool     -> (?, 7, 7, 6)
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=4, stride=2, padding=0)
        self.Square1 = Square()
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(294, 64)
        self.Square2 = Square()
        self.FC2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Square1(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        out = self.Square2(out)
        out = self.FC2(out)
        return out
    
class M7(torch.nn.Module):
    def __init__(self):
        super(M7, self).__init__()
        self.Conv1 = torch.nn.Conv1d(in_channels=1, out_channels=2, kernel_size=2, stride=2, padding=0)
        self.Square1 = Square()        
        self.Conv2 = torch.nn.Conv1d(in_channels=2, out_channels=4, kernel_size=2, stride=2, padding=0)
        self.Flatten = Flatten()
        self.FC1 = torch.nn.Linear(128, 32)
        self.Square2 = Square()
        self.FC2 = torch.nn.Linear(32, 5)
        
    def forward(self, x):
        out = self.Conv1(x)
        out = self.Square1(out)
        out = self.Conv2(out)
        out = self.Flatten(out)
        out = self.FC1(out)
        out = self.Square2(out)
        out = self.FC2(out)
        return out
    

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3),
            torch.nn.ReLU())

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 64, kernel_size=3),
            torch.nn.ReLU())
        
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.ReLU())

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 9))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class M8(torch.nn.Module):
    def __init__(self, outputs=2):
        super(M8, self).__init__()

        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
    
        # ReLU activation function
        self.relu1 = torch.nn.ReLU()

        # Max pooling layer to reduce image size by factor 2
        self.pool = torch.nn.AvgPool2d(kernel_size=2)

        # Second convolutional layer
        self.conv2 = torch.nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # ReLU activation function
        self.relu2 = torch.nn.ReLU()

        # Third convolutional layer
        self.conv3 = torch.nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # ReLU activation function
        self.relu3 = torch.nn.ReLU()

        self.Flatten = Flatten()

        # Fully connected layer to make predictions
        self.fc = torch.nn.Linear(in_features=6272, out_features=outputs)

    # Forward propagation function
    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.relu3(output)

        # Flatten the output from the convolutional layers
        output = self.Flatten(output)

        # Fully connected layer to make predictions
        output = self.fc(output)

        return output
    

# class Lenet5(torch.nn.Module):
#     def __init__(self, outputs=2):
#         super(Lenet5, self).__init__()

#         # First convolutional layer
#         self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
#         self.relu1 = torch.torch.nn.ReLU()
#         self.pool = torch.torch.nn.AvgPool2d(kernel_size=2, stride=2)  # Reduce feature map size

#         # Second convolutional layer
#         self.conv2 = torch.torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.relu2 = torch.torch.nn.ReLU()

#         # Second pooling layer to further reduce size
#         self.pool2 = torch.torch.nn.AvgPool2d(kernel_size = 2, stride = 2)  # ADDED: Extra pooling layer

#         self.Flatten = Flatten()

#         # Fully connected layer
#         self.fc1 = torch.nn.Linear(in_features=576, out_features=120)  # Reduced size
#         self.fc2 = torch.nn.Linear(in_features=120, out_features=84)  # Reduced size
#         self.fc3 = torch.nn.Linear(in_features=84, out_features=outputs)  # Reduced size

#     def forward(self, input):
#         output = self.conv1(input)
#         output = self.relu1(output)
#         output = self.pool(output)

#         output = self.conv2(output)
#         output = self.relu2(output)
#         output = self.pool2(output)  # Apply extra pooling layer

#         output = self.Flatten(output)

#         output = self.fc1(output)
#         output = self.fc2(output)
#         output = self.fc3(output)
#         return output
    
class Lenet5(torch.nn.Module):
    def __init__(self, outputs=2):
        super(Lenet5, self).__init__()

        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.relu1 = torch.nn.ReLU()
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)  # Reduce feature map size

        # Second convolutional layer
        # self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.relu2 = nn.ReLU()
        self.res1 = ResidualBlock(in_channels=6, out_channels=16, stride=1)
        # Second pooling layer to further reduce size
        self.pool2 = torch.nn.AvgPool2d(kernel_size = 2, stride = 2)  # ADDED: Extra pooling layer

        self.Flatten = Flatten()

        # Fully connected layer
        self.fc1 = torch.nn.Linear(in_features=576, out_features=120)  # Reduced size
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)  # Reduced size
        self.fc3 = torch.nn.Linear(in_features=84, out_features=outputs)  # Reduced size

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.pool(output)

        # output = self.conv2(output)
        # output = self.relu2(output)
        output = self.res1(output)
        output = self.pool2(output)  # Apply extra pooling layer

        output = self.Flatten(output)

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output