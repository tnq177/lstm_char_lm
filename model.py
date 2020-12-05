import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
else:
    torch.manual_seed(42)


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class CharLM(nn.Module):
    """LSTM LM with feed-input"""
    def __init__(self, args):
        super(CharLM, self).__init__()
        self.args = args
        self.num_layers = args.num_layers
        self.embed_dim = args.embed_dim
        self.dropout = args.dropout
        self.vocab_size = args.vocab_size
        self.embedding = Parameter(torch.Tensor(self.vocab_size, self.embed_dim))
        self.scale = ScaleNorm(self.embed_dim ** 0.5)
        self.lstm = nn.LSTM(
            input_size=self.embed_dim * 2,
            hidden_size=self.embed_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True)

        nn.init.uniform_(self.embedding, a=-0.01, b=0.01)
        for p in self.lstm.parameters():
            nn.init.uniform_(p, a=-0.01, b=0.01)

    def replace_with_unk(self, toks):
        if self.training:
            mask = torch.rand(toks.size()) <= 0.1
            toks[mask] = 0 # UNK_ID = 0

    def forward(self, inputs, targets):
        embedding = F.normalize(self.embedding, dim=-1)

        # unk replacement during training
        self.replace_with_unk(inputs)
        # [bsz, seq_len] --> [bsz, seq_len, dim]
        inputs = F.embedding(inputs, embedding)
        inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        bsz, seq_len, embed_dim = inputs.size()
        outputs = []
        prev_input = torch.zeros((bsz, 1, embed_dim)).type(inputs.type())
        h = torch.zeros((self.num_layers, bsz, self.embed_dim)).type(inputs.type())
        c = torch.zeros((self.num_layers, bsz, self.embed_dim)).type(inputs.type())
        for i in range(seq_len):
            # [bsz, 1, embed_dim]
            x = inputs[:, i:i+1, :]
            x = torch.cat((x, prev_input), dim=-1)
            x, (h, c) = self.lstm(x, (h, c))
            x = F.dropout(x, p=self.dropout, training=self.training)
            outputs.append(x)
            prev_input = x

        # [bsz, tgt_len, dim]
        outputs = torch.cat(outputs, dim=1)
        outputs = self.scale(outputs)
        logits = F.linear(outputs, embedding)
        logits = logits.reshape(-1, logits.size(-1))
        log_probs = F.log_softmax(logits, -1)

        # calculate loss
        targets = targets.reshape(-1, 1)
        nll_loss = log_probs.gather(dim=-1, index=targets)
        smooth_loss = log_probs.sum(dim=-1, keepdim=True)
        nll_loss = -(nll_loss.sum())
        smooth_loss = -(smooth_loss.sum())

        # label smoothing: https://arxiv.org/pdf/1701.06548.pdf
        label_smoothing = self.args.label_smoothing
        if label_smoothing > 0:
            loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * smooth_loss / self.vocab_size
        else:
            loss = nll_loss

        num_words = bsz * seq_len
        opt_loss = loss / num_words
        return {
            'opt_loss': opt_loss,
            'loss': loss,
            'nll_loss': nll_loss,
            'num_words': num_words,
            'logits': logits
        }

    def _sample(self, output, embedding):
        output = output.reshape(1, -1)
        # [1, dim] --> [1, V]
        logits = F.linear(output, embedding)
        # [1, dim] --> [1, V]
        probs = F.softmax(logits, dim=-1).reshape(-1)
        return torch.multinomial(probs, 1)

    def sample(self, seed, max_length=1000):
        # seed = [1, seq_len]
        embedding = F.normalize(self.embedding, dim=-1)
        inputs = F.embedding(seed, embedding)
        bsz, seq_len, embed_dim = inputs.size()

        prev_input = torch.zeros((bsz, 1, embed_dim)).type(inputs.type())
        h = torch.zeros((self.num_layers, bsz, self.embed_dim)).type(inputs.type())
        c = torch.zeros((self.num_layers, bsz, self.embed_dim)).type(inputs.type())
        for i in range(seq_len):
            # [bsz, 1, embed_dim]
            x = inputs[:, i:i+1, :]
            x = torch.cat((x, prev_input), dim=-1)
            x, (h, c) = self.lstm(x, (h, c))
            prev_input = x

        idx = self._sample(self.scale(prev_input), embedding)
        outputs = [idx.item()]
        for _ in range(max_length):
            # [1, 1, dim]
            x = F.embedding(idx.reshape(1, 1), embedding)
            # [1, 1, dim * 2]
            x = torch.cat((x, prev_input), dim=-1)
            x, (h, c) = self.lstm(x, (h, c))
            prev_input = x
            idx = self._sample(self.scale(x), embedding)
            outputs.append(idx.item())

        return outputs






