import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from MF import MatrixFactorization as mf

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

df = pd.read_csv('../data/train_dataset.csv')



X = df['user_id']
Y= df['item_id']

x_train, x_valid, y_train, y_valid = train_test_split(X,Y,train_size=0.9,test_size=0.1,random_state=42,shuffle=False)


# 假设 num_users 和 num_items 分别为用户和图书的总数
num_users = max(df['user_id']) + 1
num_items = max(df['item_id']) + 1

# 初始化模型
model = mf(num_users, num_items).to(device)

# 假设使用随机梯度下降优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 假设训练过程
def train_model(model, optimizer, x_train, y_train, epochs=10, batch_size=64):
    model.train()
    for epoch in range(epochs):
        for idx in tqdm(range(0, len(x_train), batch_size)):
            batch_x = x_train[idx:idx + batch_size].to(device)
            batch_y = y_train[idx:idx + batch_size].to(device)

            optimizer.zero_grad()
            scores = model(batch_x, batch_y)

            # 假设损失函数为平方误差
            loss = torch.nn.functional.mse_loss(scores, torch.zeros_like(scores))

            loss.backward()
            optimizer.step()


# 训练模型
# train_model(model, optimizer, torch.tensor(x_train.values), torch.tensor(y_train.values), epochs=10, batch_size=64)


# 假设评估模型的函数
def evaluate_model(model, x_valid, y_valid):
    model.eval()
    with torch.no_grad():
        x_valid_tensor = torch.tensor(x_valid.values).to(device)
        y_valid_tensor = torch.tensor(y_valid.values).to(device)
        predict = model.predict(x_valid_tensor)
        # 假设使用 F1 值作为评估指标
        f1 = f1_score(y_valid.values, predict,average='weighted')
    return f1

#torch.save(model, '../models/MF.pth')
model = torch.load('../models/MF.pth')


# 评估模型
f1 = evaluate_model(model, x_valid, y_valid)
print(f'Validation F1 score: {f1}')


# 生成test的最终结果
target = pd.read_csv('../data/test_dataset.csv')
target_tensor=torch.tensor(target.values).to(device)
res = model.predict(target_tensor)
# 将结果数据转换为DataFrame
final_df = pd.DataFrame({'user_id': [row for row in target], 'item_id': [row for row in res]})
# 将DataFrame保存到final.csv
final_df.to_csv('../data/final.csv', index=False)
