
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        # 计算预测评分，这里使用点积作为评分
        scores = torch.sum(user_embedding * item_embedding, dim=1)
        return scores

    def predict(self, user_ids):
        # 将用户ID转换为Tensor，并移到GPU上（如果有）
        user_ids_tensor = torch.tensor(user_ids, dtype=torch.long).to(device)

        # 获取用户的嵌入向量
        user_embedding = self.user_embedding(user_ids_tensor)

        # 计算用户与所有物品的预测评分
        item_ids = torch.arange(self.item_embedding.num_embeddings).to(device)
        item_embedding = self.item_embedding(item_ids)
        scores = torch.matmul(user_embedding, item_embedding.transpose(0, 1))

        # 取每个用户最高分数对应的物品ID作为预测结果
        predictions = torch.argmax(scores, dim=1)

        return predictions.cpu().numpy()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)