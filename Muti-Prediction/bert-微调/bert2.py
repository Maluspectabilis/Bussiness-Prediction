import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

torch.cuda.empty_cache()

# 读取 CSV 文件
df = pd.read_csv("E:\\pycharm-workspace\BERT\data\_final_all_data_use.csv", encoding='GBK')

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 加载预训练的 BERT 模型及其分词器
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=50)

# 对文本进行分词和转换
def tokenize_text(text):
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return tokens["input_ids"], tokens["attention_mask"]

# 自动建立关键字和类别的映射关系
def build_keyword_mapping(data_df, keyword_column):
    keyword_mapping = {}
    for _, row in data_df.iterrows():
        keyword = row[keyword_column]
        if keyword not in keyword_mapping:
            keyword_mapping[keyword] = len(keyword_mapping)

    return keyword_mapping

# 使用自动建立的关键字和类别的映射关系进行数据预处理
def preprocess_data(data_df, keyword_mapping, use_text2=False):
    input_ids = []
    attention_masks = []
    labels = []

    for _, row in data_df.iterrows():
        keyword = row["keyword"]
        text = row["annotation_dec"] if use_text2 else row["docstring"]

        # 处理空值
        if pd.isna(text):
            continue

        label = keyword_mapping.get(keyword, -1)

        if label != -1:
            input_id, attention_mask = tokenize_text(text)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

# 调用函数建立关键字和类别的映射关系
keyword_mapping = build_keyword_mapping(df, "keyword")

# 处理训练数据和测试数据
train_inputs, train_masks, train_labels = preprocess_data(train_df, keyword_mapping, use_text2=False)
test_inputs, test_masks, test_labels = preprocess_data(test_df, keyword_mapping, use_text2=False)

train_inputs2, train_masks2, train_labels2 = preprocess_data(train_df, keyword_mapping, use_text2=True)
test_inputs2, test_masks2, test_labels2 = preprocess_data(test_df, keyword_mapping, use_text2=True)

# 设置训练参数
batch_size = 4
learning_rate = 2e-5
epochs = 5

# 创建数据加载器
train_data = torch.utils.data.TensorDataset(train_inputs, train_masks, train_labels)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_data = torch.utils.data.TensorDataset(test_inputs, test_masks, test_labels)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# 设置优化器和损失函数
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for inputs, masks, labels in train_loader:
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()

        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)

    # 在测试集上评估模型性能
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, masks, labels in test_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(inputs, attention_mask=masks)
            _, predicted_labels = torch.max(outputs.logits, dim=1)

            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples

    print(f"Epoch {epoch+1}/{epochs}, Avg. Train Loss: {avg_train_loss:.8f}, Accuracy: {accuracy:.8f}")

model_path = "bert_2.pt"
torch.save(model.state_dict(), model_path)