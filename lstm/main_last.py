import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import SentimentLSTM, BiLSTM
from saa import MovieDataLoader
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--emb_size', default=128, type=int, help="Embedding size")
parser.add_argument('--n_layer', default=1, type=int, help="number of layer")
parser.add_argument('--batch_size', default=64, type=int, help="batch size")
parser.add_argument('--hidden_size', default=256, type=int, help="size of hidden layer")
parser.add_argument('--output_size', default=1, type=int, help="size of output layer")
parser.add_argument('--lr', default=0.01, type=float, help="learning rate")
parser.add_argument('--epochs', default=5, type=int, help="epoch number of training")
parser.add_argument('--models', default='lstm', type=str)
parser.add_argument('--clip', default=0.25, type=float, help="learning rate")

args = parser.parse_args()

dataset = MovieDataLoader()
train_loader, val_loader, test_dl = dataset.tr_dl, dataset.val_dl, dataset.test_dl

vocab_size = len(dataset.TEXT.vocab) + 1
n_layers = args.n_layer
batch_size = args.batch_size
embedding_dim = args.emb_size
hidden_dim = args.hidden_size
output_size = args.output_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_size, n_layers)
model.to(device)
# if args.models == 'bi-lstm':
    # model = BiLSTM(vocab_size, embedding_dim, hidden_dim, output_size, 2 * n_layers)
    # model.to(device)

lr = args.lr
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = args.epochs
counter = 0
print_every = 200
clip = args.clip


def evaluate(model, val_iter):
    """evaluate model"""
    val_h = model.init_hidden(batch_size)
    val_losses = []
    val_correct = 0

    model.eval()
    with torch.no_grad():
        for inp, lab in tqdm(val_loader):
            val_h = tuple([each.data for each in val_h])
            inp, lab = inp.to(device), lab.to(device)
            out, val_h = model(inp, val_h)

            val_loss = criterion(out.squeeze(), lab.float())
            pred = torch.round(out.squeeze())  # Rounds the output to 0/1
            correct_tensor = pred.eq(lab.float().view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            val_correct += np.sum(correct)
            val_losses.append(val_loss.item())

    val_loss = np.mean(val_losses)
    val_acc = val_correct / len(val_loader.dataset)
    return val_acc, val_loss


def train(model, optimizer, tr_dl):
    model.train()
    valid_loss_min = np.Inf
    count = 0            
    for i in tqdm(range(epochs)):
        train_losses, train_correct = [], 0 
        h = model.init_hidden(batch_size)

        for inputs, labels in tqdm(train_loader):
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output, h = model(inputs, h)

            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            pred = torch.round(output.squeeze())
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            train_correct += np.sum(correct)
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        train_acc = train_correct / len(train_loader.dataset)
        # return train_acc, train_loss
        val_acc, val_loss = evaluate(model, val_loader)
        print("Epoch: {}/{}...".format(i + 1, epochs),
              "Step: {}...".format(counter),
              "Loss: {:.6f}...".format(train_loss),
              "Val Loss: {:.6f}".format(val_loss),
              "Train accuracy: {:.3f}%".format(train_acc * 100),
              "Val accuracy: {:.3f}%".format(val_acc * 100))
        
        if np.mean(val_loss) <= valid_loss_min:

            torch.save(model.state_dict(), './state_dict.pt')
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_loss)))
            valid_loss_min = np.mean(val_loss)
        

def main():
    train(model, optimizer, train_loader)


if __name__ == "__main__":
    main()