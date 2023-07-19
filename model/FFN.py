import torch
import torch.nn as nn
import torch.nn.functional as F

class FCwithLeaky(nn.Module):
    
    def __init__(self, input_feature, output_feature, use_relu : bool = False):
        super().__init__()
        self.use_relu = use_relu
        self.fc = nn.Linear(input_feature, output_feature)
    
    def forward(self, x):
        out = self.fc(x)
        if self.use_relu:
            out = F.leaky_relu(out)
        return out

class FFN(nn.Module):
    def __init__(self, inputFeatures, layers: list):
        super().__init__()
        modules = []
        layers.reverse()
        layers.append(inputFeatures)
        layers.reverse()
        for i in range(len(layers) - 1):
            submodule = FCwithLeaky(layers[i], layers[i + 1], True if i < len(layers) - 2 else False)
            modules.append(submodule)
        
        self.model_0 = nn.Sequential(*modules[:-1])
        self.model_1 = nn.Sequential(modules[-1])
        
    def forward(self, x, use_softmax = False):
        coverage = self.model_0(x)
        res = self.model_1(coverage)
        if use_softmax:
            res = F.log_softmax(res, dim=1)
        return res, coverage
    

if __name__=='__main__':
    model = FFN(25, [16, 8, 4, 1])
    x = torch.zeros((32, 25)) # Batch 32, features 25
    output, _ = model(x)
    print(output.size())
