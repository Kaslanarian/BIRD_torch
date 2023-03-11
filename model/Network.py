import torch
from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(
        self,
        maxlen=150,
        nb_classes=2,
        nb_words=200000,
        embedding_dim=200,
        dense_dim=200,
        rnn_dim=100,
        cnn_filters=200,
        dropout_rate=0.5,
    ) -> None:
        super().__init__()

        self.maxlen = maxlen
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.nb_users = 1000000
        self.nb_query_words = 1000000

        self.embedding_dim = embedding_dim
        self.dense_dim = dense_dim
        self.rnn_dim = rnn_dim
        self.cnn_filters = cnn_filters
        self.dropout_rate = dropout_rate

        self.word_embeddings = nn.Embedding(
            self.nb_words,
            self.embedding_dim,
            padding_idx=0,
        )
        self.transfer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
        )
        self.transfer_i = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
        )
        self.transfer_q = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.dense_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dense_dim, nb_classes),
        )

    def forward(self, input_x, input_x_i, input_x_q):
        embedded = self.word_embeddings(input_x)
        embedded_i = self.word_embeddings(input_x_i)
        embedded_q = self.word_embeddings(input_x_q)
        embedded_transfer = self.transfer(embedded)
        embedded_i_transfer = self.transfer_i(embedded_i)
        embedded_q_transfer = self.transfer_q(embedded_q)

        sample_size, query_size, title_size, title_len, _ = embedded_i_transfer.shape
        print("size: {}, {}, {}, {}".format(
            sample_size,
            query_size,
            title_size,
            title_len,
        ))  # (32, 20, 21, 28)
        word_level_title_input = embedded_i_transfer.reshape(
            sample_size * query_size * title_size,
            title_len,
            self.embedding_dim,
        )  # 295 (13440, 28, 32)
        word_level_title_output = F.dropout(
            word_level_title_input.mean(1),
            self.dropout_rate,
        )  # 299 (13440, 32)

        title_level_title_input = word_level_title_output.reshape(
            sample_size * query_size,
            title_size,
            self.embedding_dim,
        )  # 308 (640, 21, 32)
        title_level_title_output = F.dropout(
            title_level_title_input.mean(1),
            self.dropout_rate,
        )  # 317 (640, 32)

        query_level_title_input = title_level_title_output.reshape(
            sample_size,
            query_size,
            self.embedding_dim,
        )  # (32, 20, 32)
        query_level_title_output = F.dropout(
            query_level_title_input.mean(1),
            self.dropout_rate,
        )  # (32, 32)

        sample_size, query_size, query_len, _ = embedded_q_transfer.shape
        word_level_query_input = embedded_q_transfer.reshape(
            sample_size * query_size,
            query_len,
            self.embedding_dim,
        )
        word_level_query_output = F.dropout(
            word_level_query_input.mean(1),
            self.dropout_rate,
        )

        sent_level_query_input = word_level_query_output.reshape(
            sample_size,
            query_size,
            self.embedding_dim,
        )
        sent_level_query_output = F.dropout(
            sent_level_query_input.mean(1),
            self.dropout_rate,
        )

        pooling = torch.concat(
            [query_level_title_output, sent_level_query_output],
            dim=-1,
        )
        return self.mlp(pooling)
