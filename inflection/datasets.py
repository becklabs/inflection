from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PairEmbeddingDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, device: str = 'cpu'):

        X = np.array([np.array(e) for e in dataframe['embeddings_x'].values])
        self.X = torch.FloatTensor(X).to(device)

        y = np.array([np.array(e) for e in dataframe['embeddings_y'].values])
        self.y = torch.FloatTensor(y).to(device)
        self.y_int = dataframe['product_int_y'].values
    
    def __getitem__(self, index) -> Any:
        return self.X[index], self.y[index], self.y_int[index]
    
    def __len__(self) -> int:
        return len(self.X) 

def parse_db_result(result: Dict):
    metadatas = result.pop('metadatas')
    result_df = pd.DataFrame(result)
    metadatas_df = pd.DataFrame(metadatas)
    df = pd.concat([result_df, metadatas_df], axis=1)
    return df