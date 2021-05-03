import re, os
from vocab import Vocab
from glove import GloVeDataset, GloVeModel
import torch.optim as optim
from torch.utils.data import DataLoader

def tokenizer(text):
    word_pattern = re.compile(r'''([0-9]+|[-/,\[\]{}`~!@#$%\^&*()_\+=:;"'?])|(\.) |(\.$)|([a-z]'[a-z])| ''')
    tokens = [token for token in word_pattern.split(text) if token]
    return tokens

FOLDER_PATH = "/content/data2" 
VOCAB_FILE_PATH = "/content/vocab2.json"

VOCAB_SIZE = 20000
EMBEDDING_DIM = 128

BATCH_SIZE = 32768
NUM_WORKERS = 2

device = "cuda"

## Building vocabulary from a set of .txt files
vocab = Vocab()
vocab.load(VOCAB_FILE_PATH)
print(vocab.total_vocab_size, vocab.size)

dataset = GloVeDataset(vocab, tokenizer=tokenizer)
dataset.generate(FOLDER_PATH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

model = GloVeModel(vocab.size+4, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
lr = 0.01
for param_group in optimizer.param_groups:
    param_group['lr'] = lr

## Train the GloVe model
model.train(
    train_loader=train_loader,
    optimizer=optimizer,
    num_epochs=5,
    save_dir="/content/",
    device=device
)