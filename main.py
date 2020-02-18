import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

import torchtext
from torchtext.data.utils import get_tokenizer


class TransformerModel(nn.Module):

    def __init__(self, num_token, num_input, num_head, num_hidden, num_layers, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(num_input, dropout)
        encoder_layers = TransformerEncoderLayer(num_input, num_head, num_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(num_token, num_input)
        self.num_input = num_input
        self.decoder = nn.Linear(num_input, num_token)

        self.init_weights()

    @staticmethod
    def _generate_square_subsequent_mask(mask_size):
        mask = torch.triu(torch.ones(mask_size, mask_size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.num_input)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, num_input, pe_dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=pe_dropout)

        pe = torch.zeros(max_len, num_input)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, num_input, 2).float() * (-math.log(10000.0) / num_input))
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


TEXT = torchtext.data.Field(
    tokenize=get_tokenizer("basic_english"),
    init_token="<sos>",
    eos_token="<eos>",
    lower=True,
)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(data, batch_size):
    data = TEXT.numericalize([data.examples[0].text])
    num_batch = data.size(0) // batch_size
    data = data.narrow(0, 0, num_batch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


train_batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, train_batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

bptt = 35


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:(i + seq_len)]
    target = source[(i + 1):(i + 1 + seq_len)].view(-1)
    return data, target


ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout_rate = 0.2  # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout_rate).to(device)


criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    num_token = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, num_token), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            print(f"| epoch {epoch:3d} | {batch:5d}/{len(train_data) // bptt:5d} batches | loss {cur_loss:5.2f}")
            total_loss = 0


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    num_token = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, num_token)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


best_val_loss = float("inf")
epochs = 2
best_model = None

for epoch in range(1, epochs + 1):
    train()
    val_loss = evaluate(model, val_data)
    print("-" * 46)
    print(f"| end of epoch {epoch:3d} | valid loss {val_loss:5.2f}")
    print("-" * 46)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step(epoch)

test_loss = evaluate(best_model, test_data)
print("=" * 46)
print(f"| End of training | test loss {test_loss:5.2f}")
print("=" * 46)
