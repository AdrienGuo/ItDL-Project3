# -*- coding: utf-8 -*-
""" Time Series Forecasting.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1--v45biFX-zB-N32NcfUSnaPiq3-PNc7

# **Project 3 - Time Series Forecasting**

Kaggle: https://www.kaggle.com/c/store-sales-time-series-forecasting

Each team will be graded by the following materials:
1. The name of your team in Kaggle and the Kaggle ranking (i.e., accuracy) of your 
submission.
2. Your source code and the script(s) (if any) for data pre-processing.
3. A list of instructions how your code is executed for model training and inference.
4. A short report (up to 2 pages) – describing the key concepts of data pre-processing, 
model, the code, and model training. (Be concise in your report!) At the end of the 
report, please specify the jobs each teammate are in charge of.
Please zip all these items into a zip file and then submit via eeclass before the midnight 
of 12/25/2021.
**Important Note: Please make sure your model is based on RNN.**
 """

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd
import numpy as np
import csv

import time

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ===========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""## **File**"""

directory_path = "data/"

train_df = pd.read_csv(directory_path+"train.csv")
test_df = pd.read_csv(directory_path+"test.csv")


# 只留最後 5個月的資料
train_data = train_df

# +--------------------------------------+
# |         Analyze the Data             |
# +--------------------------------------+

# One day contains 1782 rows.<br>
# If we want to use past 15 days data to predict 16th day, then we need to input 15*1782 data!!

# Train data contains 1684 days.
DAY_ROWS = 1782     # Number of row in one day
NUM_DAYS = 1684     # Number of total day


# +--------------------------------------+
# |         Normalization                |
# +--------------------------------------+

minmax_store = preprocessing.MinMaxScaler()
minmax_sales = preprocessing.MinMaxScaler()

train_data[["store_nbr"]] = minmax_store.fit_transform(train_data[["store_nbr"]])
train_data[["sales"]] = minmax_sales.fit_transform(train_data[["sales"]])


# +--------------------------------------+
# |         Label Encoding               |
# +--------------------------------------+

# labelencoder = LabelEncoder()

# print("Here is the new family")
# print("Here is the shape: {}".format(train_data[['family']].to_numpy().ravel().shape))
# print(labelencoder.fit_transform(train_data[['family']]))
#train_data[['family']] = labelencoder.fit_transform(train_data[['family']].to_numpy().ravel())

#test_data = labelencoder.fit_transform(test_df[['family']])

train_label = train_data[["sales"]]         # 兩個中括號是為了讓輸出格式變成 dataframe
train_data = train_data[['sales']]

test_data = test_df
test_data = test_data[[]]

# +--------------------------------------+
# |      Turn into the shape I want      |
# |                                      |
# | From (3000888, 3) to (3000888/1782, 3*1782) = (1684, 5346)
# |                                      |
# |     也就是把每一天的資訊
# |     都拉成在同一列裡面
# |     而不是分成好幾列
# +--------------------------------------+

def create_data(train_data, train_label):
    new_train_data = []
    new_train_label = []
    for i in range(0, train_data.shape[0]//1782, 1):
        new_data = train_data[i*1782: (i+1)*1782].to_numpy().ravel()
        new_label = train_label[i*1782: (i+1)*1782].to_numpy().ravel()
        # 把同一天的 feature 都拉成一條長長的 sequence

        new_train_data.append(new_data)
        new_train_label.append(new_label)

    return new_train_data, new_train_label

new_train_data, new_train_label = create_data(train_data, train_label)
# new_train_data (list)
# new_train_label (list)

train_data = new_train_data       # (list)
train_label = new_train_label     # (list)

# +--------------------------------------+
# |         資料分割                      |
# +--------------------------------------+

#train_data, val_data, train_label, val_label = train_test_split(new_train_data, new_train_label, train_size=0.8, shuffle=False)

# +--------------------------------------+
# |     DataSet and DataLoader           |
# +--------------------------------------+

SEQ_LEN = 30
BATCH_SIZE = 32

class RNNDataset(Dataset):
    def __init__(self, data, label, seq_len):     # seq_len is the length of each input
        # in this case, if seq_len=3, then it will input 3-day info. in each input
        self.data = torch.FloatTensor(data)
        self.label = torch.FloatTensor(label)
        #self.input = torch.from_numpy(input).float()
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - self.seq_len)
        # 先算出總共有幾天 再去減掉 一次input的天數

    def __getitem__(self, index):
        return self.data[index: index+self.seq_len], self.label[index+self.seq_len]

train_dataset = RNNDataset(train_data, train_label, SEQ_LEN)
#val_dataset = RNNDataset(val_data, val_label, SEQ_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# +--------------------------------------+
# |             RNNModel                 |
# +--------------------------------------+

# **bidirectional**: make this RNN bidirectional this is very useful in many applications where the next sequences can help previous sequences in learning

# <a href="https://imgur.com/BnntzYd"><img src="https://i.imgur.com/BnntzYd.png" title="source: imgur.com" /></a>

# N time steps (horizontally)
# In our case, N is the period of Date.

INPUT_DIM = len(train_data[0])
N_NEURONS = 1000                       # This will be the output dimension of LSTM
NUM_LAYERS = 5
OUTPUT_DIM = len(train_label[0])      # This is the output dimension we want

class RNNModel(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNModel, self).__init__()
        # self.batch_size = batch_size      好像是不需要 batch_size
        self.input_dim =  input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        #self.rnn = nn.RNNCell(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers = num_layers,
            batch_first=True,
            dropout=0.2,
            #bidirectional=True,
        )
        # **加一個 bidirection
        # if batch_first: (Batch Size, Sequence Length, Input Dimension)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # bidirectional: make this RNN bidirectional this is very useful in many applications/
        #                where the next sequences can help previous sequences in learning
        # batch_first=True: make shape to (Batch Size, Sequence Length, Input Dimension)
    
    # Initialize hidden weight
    def init_hidden(self, x):
        hidden0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        cell0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        # hidden = torch.zeros(self.num_layers, self.batch_size, self.input_dim)
        # 不能寫成上面這行是因為 batch_size 可能會不固定 (當資料長度不能被 batch_size 整除的時候)
        return hidden0, cell0

    def forward(self, x):
        #print(f"Here is the x: {x.size()}")
        hidden0, cell0 = self.init_hidden(x)
        output, (hidden, cell) = self.lstm(x)
        # output dim: (Batch Size, Sequence Length, Hidden Dimension)
        # hidden dim: (Num Layers, Batch Size, Hidden Dimension)
        # cell dim: (Num Layers, Batch Size, Hidden Dimension)
        output = output[:, -1, :]    # Just want the last sequence of LSTM
        output = self.fc(output)

        return output

# +--------------------------------------+
# |             Train                    |
# +--------------------------------------+

# -------- Optimization (class) ----------
# Try a new way to build the train block. <br>
# Reference: https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

class Optimization():
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        self.model.train()
        y_hat = self.model(x)
        #print(f"This is pred: {y_hat.size()}")
        #print(f"This is actual: {y.size()}")
        loss = self.loss_fn(y_hat, y)       # Compute loss
        self.optimizer.zero_grad()
        loss.backward()                     # Compute gradient
        self.optimizer.step()               # Update parameters

        return loss.item()
    
    def train(self, train_loader, n_epochs):
        model_path = "./model.pth"

        for epoch in range(n_epochs):
            start = time.time()
            train_loss = []     # initialize batch_losses
            self.model.train()
            for data, labels in tqdm(train_loader):
                data = data.to(DEVICE)
                labels = labels.to(DEVICE)
                #print(f"This is data: {data}")
                #print(f"This is labels: {labels}")
                loss = self.train_step(data, labels)
                train_loss.append(loss)
            end = time.time()

            # val_loss = []
            # self.model.eval()

            # for data, labels in val_loader:
            #     data = data.to(DEVICE)
            #     labels = labels.to(DEVICE)
            #     y_hat = self.model(data)
            #     loss = self.loss_fn(y_hat, labels).item()
            #     val_loss.append(loss)

            training_loss = np.mean(train_loss)     # calculate the mean of all the batches in this epoch
            self.train_losses.append(training_loss)
            # valing_loss = np.mean(val_loss)
            # self.val_losses.append(valing_loss)
        
            print(f"[{epoch+1:3}/{n_epochs}] Training loss: {training_loss:.4f}")
            # print(f"[{epoch+1:3}/{n_epochs}] Training loss: {valing_loss:.4f}")
            # print(f"Total Time: {end-start}s")

        torch.save(self.model.state_dict(), model_path)
    
    def evaluate(self, test_loader):
        # type(data) is Tensor
        with torch.no_grad():
            for data in test_loader:
                data = data.to(DEVICE)
                self.model.eval()
                pred_hat = self.model(data)

            return pred_hat


# Criterion
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
#        print(f"This is pred: {pred}")
#        print(f"This is actual: {actual}")
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

# +--------------------------------------+
# |             Training                 |
# +--------------------------------------+
N_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6

model = RNNModel(batch_size=BATCH_SIZE, input_dim=INPUT_DIM, hidden_dim=N_NEURONS, 
                 num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM).to(DEVICE)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)

opt.train(train_dataloader, n_epochs=N_EPOCHS)


# +--------------------------------------+
# |                 Test                 |
# +--------------------------------------+
pred_list = []
future_days = 16

def test_reshape(data, i):
    data = pd.DataFrame(data)
    test_data_tmp = test_data[i*1782: (i+1)*1782]
    test_data_tmp.reset_index(drop=True, inplace=True)
        # drop=True 就是不會把 index 也當作一個欄位
        # inplace 就是會直接取代這個 dataframe
    data = pd.concat([test_data_tmp, data], axis=1, ignore_index=True)
    data = data.to_numpy().ravel()        # test_data 也要拉成一維的
    
    return data

input_list = new_train_data[-30: ]
pred_df_all = pd.DataFrame(columns=["sales"])

for i in range(future_days):
    print(i)
    pred_list = []
    input_list = input_list[-30: ]            # 都只取最後30個
    input_dataset = torch.FloatTensor(input_list)

    input_dataset = input_dataset.unsqueeze(0)
    # 2d -> 3d

    input_dataloader = DataLoader(input_dataset, batch_size=1, shuffle=False, num_workers=0)

    pred = opt.evaluate(input_dataloader)
    
    # 格式轉換
    for ele in pred[0]:
        pred = ele.item()
        if pred < 0:
            pred = 0
        pred_list.append(pred)

    pred_array = test_reshape(pred_list, i)        # i 用來取得要 test_data 的第幾天
    # pred_array 已經轉成 array
    # 因為要 input_list 裡面都是儲存 array type

    input_list.append(pred_array)

    pred_df = pd.DataFrame(pred_list, columns=["sales"])
    pred_df[["sales"]] = minmax_sales.inverse_transform(pred_df[["sales"]])
    
    pred_df_all = pd.concat([pred_df_all, pred_df], axis=0, ignore_index=True)
    
    i+=1

print(pred_df_all[:])


# +--------------------------------------+
# |                 Save                 |
# +--------------------------------------+
def save_file(pred_path, predicts):
    print("Saving result to {}".format(pred_path))
    with open(pred_path, 'w', encoding='utf8', newline='') as f:
        write = csv.writer(f)
        write.writerow(['id', 'sales'])
        for i, pred in enumerate(predicts):
            print(pred)
            write.writerow([i+3000888, pred])

pred_path = directory_path+"prediction.csv"
save_file(pred_path, pred_df_all["sales"])
