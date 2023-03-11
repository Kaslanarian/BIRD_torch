import torch
from torch import nn


class BPTRUCell(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 batch_first=False) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.W_u = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.W_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_f = nn.parameter.Parameter(torch.randn(hidden_size))
        self.v_f = nn.parameter.Parameter(torch.randn(hidden_size))
        self.b_r = nn.parameter.Parameter(torch.randn(hidden_size))
        self.v_r = nn.parameter.Parameter(torch.randn(hidden_size))
        self.b_o = nn.parameter.Parameter(torch.randn(hidden_size))

    def forward_once(self, x, cur_state):
        h, c = cur_state
        xh = self.W_u(x)
        x1, x2, x3, x4 = torch.split(xh, self.hidden_size, dim=-1)
        f = torch.sigmoid(x1 + self.v_f * c + self.b_f)
        r = torch.sigmoid(x2 + self.v_r * c + self.b_r)
        o = torch.sigmoid(self.W_c(c) + self.W_h(h) + self.b_o)

        new_c = torch.tanh(f * c + (1 - f) * x3 + o * h)
        new_h = torch.tanh(r * h + (1 - r) * x4 + o * c)

        return new_h, (new_h, new_c)

    def init_hidden(self, batch_size):
        return (
            torch.zeros((batch_size, 1, self.hidden_size),
                        device=self.b_f.device),
            torch.zeros((batch_size, 1, self.hidden_size),
                        device=self.b_f.device),
        )

    def forward(self, x):  # no hidden state init
        if not self.batch_first:
            x = x.swapaxes(0, 1)
        b, l, _ = x.shape
        hidden_state = self.init_hidden(b)

        output_list = []
        for t in range(l):
            output, hidden_state = self.forward_once(
                x[:, t:t + 1, :],
                hidden_state,
            )
            output_list.append(output)
        
        return torch.concat(output_list, dim=1), hidden_state
