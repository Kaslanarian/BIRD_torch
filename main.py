from utils.data_loader import Data_Loader
from model.BIRD import BIRD
from model.Network import Network

import torch
from torch import nn
import torch.nn.functional as F

import json
from tqdm import tqdm

data_loader = Data_Loader()

lr = 0.002
batch_size = 32
epochs = 5
embedding_dim = 32
rnn_dim = 16
dense_dim = 32
dropout_rate = 0.75

data_name = "PPDD"
train_size = 2887628
train_size_True = 2887628
black_size = 114250
white_size = 2068683
train_black_size = 79975
real_train_black_size = 1439550
train_white_size = 1448078
val_black_size = 17137
val_white_size = 310302
test_black_size = 17138
test_white_size = 310303
user_index = 908540
word_index = 33000
query_word_index = 33000
nb_classes = 2

train_set_path = data_loader.data_root + data_name + '/train_list.txt'
val_set_path = data_loader.data_root + data_name + '/val_list.txt'
test_set_path = data_loader.data_root + data_name + '/test_list.txt'
online_test_set_path = data_loader.data_root + data_name + '/online_test_1.txt'
online_test_set_path2 = data_loader.data_root + data_name + '/online_test_2.txt'

word_index_path = data_loader.data_root + 'dictionary/' + data_name + '/word_index.json'
with open(word_index_path, 'r', encoding='UTF-8') as fp:
    word_index = json.load(fp)

model = BIRD(dropout_rate=dropout_rate)
model.nb_words = len(word_index) + 1
model.nb_query_words = 0
model.nb_users = 0


def train(
    model,
    data_generator: Data_Loader.session_group_user_query_item_generator_hiera,
    load_feature_label,
    train_data,
    val_data,
    test_data,
    word_index,
    query_word_index,
    user_index,
    black_user_set,
    nb_classes,
    epochs,
    train_size,
    batch_size,
    *args,
    **kwargs,
):
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        print("training")
        model.train()
        for Item, X, X_Q, User_Y, X_Y, Y, Length in tqdm(
                data_generator(
                    train_data,
                    word_index,
                    query_word_index,
                    user_index,
                    black_user_set,
                    nb_classes,
                    batch_size,
                )):
            Item = torch.LongTensor(Item)
            X = torch.LongTensor(X)
            X_Q = torch.LongTensor(X_Q)
            Y = torch.FloatTensor(Y)
            output = model(Item, X, X_Q)
            loss = F.cross_entropy(output, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("evaluating")
        model.eval()
        n_right, n_sample = 0, 0
        for Item, X, X_Q, User_Y, X_Y, Y, Length in tqdm(
                data_generator(
                    test_data,
                    word_index,
                    query_word_index,
                    user_index,
                    black_user_set,
                    nb_classes,
                    batch_size,
                )):
            Item = torch.LongTensor(Item)
            X = torch.LongTensor(X)
            X_Q = torch.LongTensor(X_Q)
            Y = torch.FloatTensor(Y)
            output = model(Item, X, X_Q)

            n_sample += X.shape[0]
            y_pred = output.argmax(1).numpy()
            y_true = Y.argmax(1).numpy()
            n_right += (y_pred == y_true).sum()

        print(n_right / n_sample)


train(
    model,
    data_loader.session_group_user_query_item_generator_hiera,
    data_loader.load_session_group_user_query_item_label_hiera,
    train_set_path,
    val_set_path,
    test_set_path,
    word_index,
    None,
    None,
    None,
    nb_classes,
    epochs,
    train_size,
    batch_size,
)
