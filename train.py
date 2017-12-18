import logging
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import dataset
from embeddings import EmbeddingsDictionary
from dataset import SifDataset
from model import BowModel

logger = logging.getLogger()
# on met le niveau du logger à DEBUG, comme ça il écrit tout
logger.setLevel(logging.INFO)

fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
console.setLevel(logging.INFO)
logger.addHandler(console)

# Here we load only a small chunk of the embeddings (100k most common words)
# You can change it if you want
all_words = set(line.strip() for line in open('all_sst_words.txt'))
emb_dict = EmbeddingsDictionary(word_whitelist=all_words)

data = SifDataset()
train_exs, train_labels = dataset.preprocess_dataset(data.train, emb_dict.dictionary)
logging.info('Loaded train, size={}, npos={}'.format(len(train_exs), sum(train_labels).sum()))
dev_exs, dev_labels = dataset.preprocess_dataset(data.dev, emb_dict.dictionary)
logging.info('Loaded dev, size={}, npos={}'.format(len(dev_exs), sum(dev_labels).sum()))


model = BowModel(emb_dict.emb)
loss_fn = nn.NLLLoss()
optimized_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(optimized_params, lr=0.003)


def train_epoch():
    model.train()
    n_in_loss = 0
    total_loss = 0
    n_correct = 0
    for ex, label in zip(train_exs, train_labels):
        model.zero_grad()
        log_probs = model(Variable(ex.unsqueeze(0), requires_grad=False))
        loss = loss_fn(log_probs, Variable(label, requires_grad=False))
        total_loss += loss.data.sum()
        n_correct += (log_probs.data.max(dim=1)[1] == label).sum()
        n_in_loss += label.numel()
        if (n_in_loss >= 10000):
            logging.info('Train Loss: {:.3f}, accuracy: {:.1f}%'.format(
                total_loss / n_in_loss, 100.0 * n_correct / n_in_loss))
            total_loss = 0
            n_in_loss = 0
            n_correct = 0
        loss.backward()
        optimizer.step()


def test_epoch(epoch):
    model.eval()
    total_loss = 0
    n_correct = 0
    for ex, label in zip(dev_exs, dev_labels):
        log_probs = model(Variable(ex.unsqueeze(0), volatile=True))
        loss = loss_fn(log_probs, Variable(label, volatile=True))
        n_correct += (log_probs.data.max(dim=1)[1] == label).sum()
        total_loss += loss.data.sum()
    accuracy = 100.0 * n_correct / len(dev_exs)
    avg_loss = total_loss / len(dev_exs)
    logging.info('Epoch {}, test loss: {:.3f}, accuracy: {:.1f}%'.format(
        epoch, avg_loss, accuracy))
    return accuracy


for epoch in range(0, 50):
    logging.info('Starting epoch {}'.format(epoch))
    train_epoch()
    accuracy = test_epoch(epoch)
    torch.save({'net': model, 'dict': emb_dict.dictionary, 'score': accuracy}, 'model.pth')
