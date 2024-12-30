import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features = 9, hl1 = 5, hl2 = 4, classification = 3):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hl1)
        self.fc2 = nn.Linear(hl1, hl2)
        self.fc2 = nn.Linear(hl2, classification)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x