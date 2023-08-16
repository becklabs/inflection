import torch.nn as nn
import torch.nn.functional as F

class EmbeddingPredictor(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(EmbeddingPredictor, self).__init__()
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(embedding_size, embedding_size * 2)
        self.fc2 = nn.Linear(embedding_size * 2, embedding_size * 2)
        self.fc3 = nn.Linear(embedding_size * 2, embedding_size * 2)
        self.fc4 = nn.Linear(embedding_size * 2, embedding_size * 2)
        self.fc5 = nn.Linear(embedding_size * 2, embedding_size * 2)
        self.fc6 = nn.Linear(embedding_size * 2, embedding_size)
        self.leaky_relu = nn.LeakyReLU()
  
    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.leaky_relu(self.fc4(x))
        x = self.leaky_relu(self.fc5(x))
        x = F.normalize(self.fc6(x), p=2, dim=1)
        return x