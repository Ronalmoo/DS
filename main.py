import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
from model import SentimentLSTM
from data_loader import MovieDataLoader
from tqdm import tqdm

dataset = MovieDataLoader()
tr_dl, val_dl, test_dl = dataset.tr_dl, dataset.val_dl, dataset.test_dl
vocab_size = len(dataset.TEXT.vocab) + 1
hidden_dim = 128
layer_dim = 2  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 1
batch_size = 64
model = SentimentLSTM(vocab_size, 256, 512, output_dim, layer_dim)
# model = model.cuda()
criterion = nn.BCELoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(loss_fn, model, data_loader, device):
    if model.training:
        model.eval()
    total_loss = 0
    correct_count = 0
    val_h = model.init_hidden(batch_size)
    for step, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        val_h = tuple([each.data for each in val_h])
        data, target = data.to(device), target.to(device)
        # data, target = data.cuda(), target.cuda()
        output, val_h = model(data, val_h)
        with torch.no_grad():
            loss = loss_fn(output.squeeze(), target.float())
        total_loss += loss.item() * data.size(0)
        output = output >= 0.5
        correct_count += (output.max(dim=-1).indices == target).sum().item()
    else:
        acc = correct_count / len(data_loader.dataset)
        loss = total_loss / len(data_loader.dataset)
    return acc, loss


def train(loss_fn, model, tr_dl, val_dl, optimizer, device, epochs):
    tr_acc_history, tr_loss_history = [], []
    val_acc_history, val_loss_history = [], []
    if not model.training:
        model.train()
    for epoch in tqdm(range(epochs), total=epochs):
        tr_loss = 0
        correct_count = 0
        h = model.init_hidden(batch_size)

        for step, (data, target) in tqdm(enumerate(tr_dl), total=len(tr_dl)):
            h = tuple([e.data for e in h])
            data, target = data.to(device), target.to(device)

            # data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output, h = model(data, h)
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            output = output >= 0.5
            correct_count += (output.max(dim=-1).indices == target).sum().item()
            tr_loss += loss.item()
        else:
            tr_acc = correct_count / len(tr_dl.dataset)
            tr_loss /= (step + 1)
            val_acc, val_loss = test(loss_fn, model, val_dl, device)
            tr_acc_history.append(tr_acc)
            tr_loss_history.append(tr_loss)
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            tqdm.write('epoch: {:3}, tr_acc: {:.2%}, tr_loss: {:.3f}, val_acc: {:.2%}, val_loss: {:.3f}'.format(epoch,
                                                                                                                tr_acc,
                                                                                                                tr_loss,
                                                                                                                val_acc,
                                                                                                                val_loss))
    print("Finished!")
    plt.figure(figsize=(8, 4))
    plt.title("tr/val Loss")
    plt.ylabel("loss")
    plt.subplot(1, 2, 1)
    plt.plot(tr_loss_history, 'g-', label="training_loss")
    plt.plot(val_loss_history, 'c:', label="val_loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("")
    plt.ylabel("tr/val Accuracy")
    plt.plot(tr_acc_history, 'b-', label="training_acc")
    plt.plot(val_acc_history, 'r:', label="val_acc")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    train(criterion, model, tr_dl, val_dl, optimizer, 'cpu', 10)


if __name__ == '__main__':
    main()