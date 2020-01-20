import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import SentimentLSTM
from saa import MovieDataLoader
from tqdm import tqdm

dataset = MovieDataLoader()
train_loader, val_loader, test_loader = dataset.tr_dl, dataset.val_dl, dataset.test_dl

vocab_size = len(dataset.TEXT.vocab) + 1
output_size = 1
embedding_dim = 128
hidden_dim = 256
n_layers = 1
batch_size = 64
device = 'cpu'
model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_size, n_layers)
model.to(device)

lr=0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 2
counter = 0
print_every = 1000
clip = 5
valid_loss_min = np.Inf

model.load_state_dict(torch.load('./state_dict.pt'))

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)

model.eval()
for inputs, labels in tqdm(test_loader):
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc * 100))