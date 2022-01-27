# ItDL-Project3
Introduction to Deep Learning - Project 3

# 1. Data Analysis

在 train.csv 這裡面，有 date, store_nbr, family, sales, onpromotion 共 5 種特徵，date 就是日期、store_nbr 是店家的編號、family 是商品分類、sales 是銷售量、onpromotion 代表是否在促銷。我們發現每一天總共有 1782 筆資料、總共有 1684 天。


# 2. Data Pre-processing

我們除了使用 train.csv 的資料外，我們還考慮了 oil.csv 這裡面的 dcoilwtico 特徵，裡面放了很多天的油價資料，但我們發現，這個特徵除了有缺失值之外，還有某幾天是沒有的，因此我們先將缺失值補上平均，然後使用 merge 將油價和 train.csv 合併，再將 train.csv 裡面沒有油價資訊的那幾天，補上平均。<br>
　在經過多次測試之後我們只考慮 sales 和 dcoilwtico 這兩個特徵，其他特徵對我們的模型並沒有幫助，以及我們有分別對它們做 Normalization。<br>
　因為下面我們使用的模型(LSTM)有輸入的維度限制，因此我們要將這裡的資料，同一天的要放在一起，簡單來說，現在一天有 1782 筆資料，每一筆資料有 sales 和 dcoilwtico 的特徵，那這就是一個 1782 x 2 的 array，我們要將他拉長一個 3564 的 sequence，才能輸入進 LSTM 模型裡面。

# 3. Model and Model Training
## 3.1 LSTM model
我們使用 LSTM 當作我們的模型架構，會使用 LSTM 是因為這次的資料是跟時間序列相關，而 LSTM 就是拿來處理與序列相關的模型之一，效果也比最一般 RNN 模型還要好，因為 LSTM 可以擁有長時間的記憶 ，但 RNN 是沒有辦法的。LSTM 的架構圖如圖片</br>
　橫軸方向代表 Input 的數量，圖片在時間序列的資料裡面代表的就是時間長度(總共幾天)；縱軸的方向代表 LSTM 的 Layer 數量，這兩個都是可以自行設定的；最上面的 Output 是 LSTM 的輸出，因此 LSTM 模型可以是多對多或是多對一的模型，這也是由使用者自行設定。

![](https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs13638-019-1511-4/MediaObjects/13638_2019_1511_Fig9_HTML.png)

## 3.2 Our Model Architecture
我們考慮前 30 天的資料，因此橫軸長度設定為 30、Layer 數量設為 3、Output 我們只取最後一個輸出，因為是考慮前 30 天的資訊後，運算出第 31 天的資訊，因此是多對一的模型。<br>
　另外，每個 LSTM 還可以設置它的神經元個數(hidden_dim)、dropout、bidirectional，最後一個 bidirectional 的意思是讓 LSTM 可以雙向的來訓練資料，而不會因為前面的資料無法參考的後面的資料而讓訓練效果不好。我們的設定：<br>
```python
self.lstm = nn.LSTM(
    input_size=input_dim(3564),
    hidden_size=hidden_dim(100),
    num_layers = num_layers(3),
    batch_first=True,
    dropout=0.2,
    bidirectional=True
)
```

### 3.2.1 Hyperparameter
Optimizer: Adam<br>
Loss function: Mean squared error<br>
Batch size: 16<br>
Epoch: 15<br>


