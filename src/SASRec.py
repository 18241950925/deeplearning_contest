import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# 判断是否可以使用GPU加速
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 加载数据集
df = pd.read_csv('../data/train_dataset.csv')
print('共{}个用户，{}本图书，{}条记录'.format(max(df['user_id']) + 1, max(df['item_id']) + 1, len(df)))


# 假设进行一些简单的数据预处理
def preprocess_data(df):
    # 可以根据需要进行数据预处理，比如处理缺失值、去除异常值等
    return df


df = preprocess_data(df)

# 划分训练集和验证集
X = df['user_id']
Y = df['item_id']

x_train, x_valid, y_train, y_valid = train_test_split(X, Y, train_size=0.85, random_state=42)


# 定义 SASRec 模型
class SASRec(torch.nn.Module):
    def __init__(self, num_items, embed_dim=32, num_heads=1, num_blocks=1):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # Item embedding layer
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, padding_idx=0)

        # Positional encoding
        self.pos_embedding = torch.nn.Embedding(100, embed_dim)

        # Self-attention layers
        self.self_attentions = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(embed_dim, num_heads) for _ in range(num_blocks)
        ])

        # Layer normalization
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        # Final output layer
        self.output_layer = torch.nn.Linear(embed_dim, num_items)

    def forward(self, input_seq):
        # Embedding input sequence
        item_embed = self.item_embedding(input_seq)

        # Adding positional encoding
        seq_len = input_seq.size(1)
        pos_seq = torch.arange(seq_len).unsqueeze(0).repeat(input_seq.size(0), 1).to(device)
        pos_embed = self.pos_embedding(pos_seq)
        seq_embed = item_embed + pos_embed

        # Self-attention blocks
        for self_attention in self.self_attentions:
            seq_embed, _ = self_attention(seq_embed, seq_embed, seq_embed)

        # Layer normalization and final output
        seq_embed = self.layer_norm(seq_embed)
        logits = self.output_layer(seq_embed)

        return logits


# 初始化 SASRec 模型
num_items = max(df['item_id']) + 1  # 假设 item_id 从 0 开始
model = SASRec(num_items).to(device)

# 假设使用随机梯度下降优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 假设训练过程
def train_model(model, optimizer, x_train, y_train, epochs=10, batch_size=64):
    model.train()
    for epoch in range(epochs):
        # 使用 tqdm 来显示进度条
        for idx in tqdm(range(0, len(x_train), batch_size)):
            batch_x = x_train[idx:idx + batch_size].to(device)
            batch_y = y_train[idx:idx + batch_size].to(device)

            optimizer.zero_grad()
            logits = model(batch_x)

            # 假设损失函数为交叉熵损失
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, num_items), batch_y.view(-1))

            loss.backward()
            optimizer.step()


# 训练模型
train_model(model, optimizer, torch.tensor(x_train.values).unsqueeze(2), torch.tensor(y_train.values), epochs=10,
            batch_size=64)


# 假设评估模型的函数
def evaluate_model(model, x_valid, y_valid):
    model.eval()
    with torch.no_grad():
        x_valid_tensor = torch.tensor(x_valid.values).unsqueeze(2).to(device)
        y_valid_tensor = torch.tensor(y_valid.values).to(device)
        logits = model(x_valid_tensor)
        predictions = torch.argmax(logits, dim=2).cpu().numpy()
        f1 = f1_score(y_valid.values.flatten(), predictions.flatten(), average='micro')
    return f1


# 评估模型
f1 = evaluate_model(model, x_valid, y_valid)
print(f'Validation F1 score: {f1}')
