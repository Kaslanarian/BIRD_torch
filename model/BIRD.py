import torch
from torch import nn
import torch.nn.functional as F

from model.Network import Network
from model.BPTRU import BPTRUCell


class BIRD(Network):
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
        super().__init__(
            maxlen,
            nb_classes,
            nb_words,
            embedding_dim,
            dense_dim,
            rnn_dim,
            cnn_filters,
            dropout_rate,
        )

        self.word_embeddings = nn.Embedding(
            self.nb_words,
            self.embedding_dim,
        )  # 没有padding
        self.combine_gate = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.Sigmoid(),
        )
        # 一个正向一个反向
        self.rnn1 = BPTRUCell(self.embedding_dim,
                              self.rnn_dim,
                              batch_first=True)
        self.rnn2 = BPTRUCell(self.embedding_dim,
                              self.rnn_dim,
                              batch_first=True)

    def forward(self, input_x, input_x_i, input_x_q):
        embedded = self.word_embeddings(input_x)
        embedded_i = self.word_embeddings(input_x_i)
        embedded_q = self.word_embeddings(input_x_q)
        embedded_transfer = self.transfer(embedded)
        embedded_i_transfer = self.transfer_i(embedded_i)
        embedded_q_transfer = self.transfer_q(embedded_q)

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

        sent_level_query = word_level_query_output.reshape(
            sample_size,
            query_size,
            self.embedding_dim,
        )

        sample_size, query_size, title_size, title_len, _ = embedded_i_transfer.shape
        sent_level_query_rely = sent_level_query.reshape(
            *sent_level_query.shape[:2], 1, 1,
            -1).repeat(1, 1, title_size, title_len, 1).reshape(
                sample_size * query_size * title_size * title_len,
                self.embedding_dim,
            )

        word_level_title_input = embedded_i_transfer.reshape(
            sample_size * query_size * title_size * title_len,
            self.embedding_dim,
        )

        title_word_atten = torch.sum(
            sent_level_query_rely * word_level_title_input, -1)
        title_word_atten = F.softmax(
            title_word_atten.reshape(
                sample_size * query_size * title_size,
                title_len,
            ),
            dim=-1,
        ).reshape(sample_size * query_size * title_size * title_len, 1)

        word_level_title_input_atten = word_level_title_input * title_word_atten

        word_level_title_output = word_level_title_input_atten.reshape(
            sample_size * query_size * title_size,
            title_len,
            self.embedding_dim,
        ).sum(1)

        sent_level_query_rely_2 = sent_level_query.unsqueeze(2).repeat(
            1, 1, title_size, 1).reshape(
                sample_size * query_size * title_size,
                self.embedding_dim,
            )

        word_level_title_output_atten = F.softmax(
            torch.sum(word_level_title_output * sent_level_query_rely_2,
                      -1).reshape(
                          sample_size * query_size,
                          title_size,
                      ),
            dim=-1).reshape(sample_size * query_size * title_size, 1)

        title_level_title_input_atten = word_level_title_output * word_level_title_output_atten
        title_level_title_input = title_level_title_input_atten.reshape(
            sample_size * query_size,
            title_size,
            self.embedding_dim,
        )

        title_level_title_output = F.dropout(
            torch.sum(title_level_title_input, 1),
            self.dropout_rate,
        )

        query_level_title = torch.reshape(
            title_level_title_output,
            [sample_size, query_size, self.embedding_dim],
        )

        combine_gate = self.combine_gate(
            torch.concat(
                [sent_level_query, query_level_title],
                dim=-1,
            ))

        query_level = (1 - combine_gate) * query_level_title + \
                        combine_gate * sent_level_query

        output1, state1 = self.rnn1(query_level)
        output2, state2 = self.rnn2(query_level)

        c1, c2 = state1[1], state2[1]
        cell_memory = torch.concat([c1, c2], -1)
        rnn_outputs = torch.concat([output1, output2], -1)
        last_hidden = rnn_outputs[:, -2:-1]
        last_hidden_reply = last_hidden.repeat(1, query_size, 1)

        numerator = torch.sum(last_hidden_reply * rnn_outputs, axis=-1)
        denominator = torch.multiply(
            last_hidden_reply.square().sum(-1).sqrt(),
            rnn_outputs.square().sum(-1).sqrt(),
        )

        last_hidden_atten = F.softmax(
            torch.sigmoid(numerator / denominator),
            dim=-1,
        ).unsqueeze(-1)  # ?
        output_attentive = torch.sum(last_hidden_atten * rnn_outputs, 1)
        pooling = torch.concat(
            [output_attentive, cell_memory.squeeze()],
            dim=-1,
        )

        return self.mlp(pooling)