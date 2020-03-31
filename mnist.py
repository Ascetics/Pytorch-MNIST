import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(8, 8, 3, padding=1),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(16, 16, 3, padding=1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 64),
                                nn.ReLU(inplace=True))
        self.cls = nn.Linear(64, 10)
        pass

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.cls(x)
        return x

    pass


if __name__ == '__main__':

    ROOT = 'C:/torch_datasets'
    BATCH_SIZE = 200
    EPOCHS = 3

    train_data = MNIST(root=ROOT, transform=Compose([ToTensor()]),
                       train=True)
    valid_data = MNIST(root=ROOT, transform=Compose([ToTensor()]),
                       train=False)
    train_data = DataLoader(train_data, batch_size=BATCH_SIZE)
    valid_data = DataLoader(valid_data, batch_size=BATCH_SIZE)

    model = MNISTModel()
    lossfn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    for e in range(1, EPOCHS + 1):
        train_loss = 0.0
        train_acc = 0.0
        for im, lb in tqdm(train_data):
            optim.zero_grad()

            output = model(im)
            loss = lossfn(output, lb)
            loss.backward()
            train_loss += loss.cpu().detach().item()
            optim.step()

            pred = torch.argmax(output, dim=1)
            train_acc += (pred == lb).sum(dim=0).numpy() / float(train_data.batch_size)
            # print((pred == lb).sum(dim=0))
            pass
        train_loss /= len(train_data)
        train_acc /= len(train_data)

        with torch.no_grad():
            valid_loss = 0.0
            valid_acc = 0.0
            for im, lb in tqdm(valid_data):
                output = model(im)
                loss = lossfn(output, lb)
                valid_loss += loss.cpu().detach().item()

                pred = torch.argmax(output, dim=1)
                valid_acc += (pred == lb).sum(dim=0).numpy() / float(valid_data.batch_size)
                # print((pred == lb).sum(dim=0))
                pass
            valid_loss /= len(valid_data)
            valid_acc /= len(valid_data)

        print(('Epoch:{:d}|'
               'TrainLoss:{:.4f}|TrainAcc:{:.4f}|'
               'ValidLoss:{:.4f}|ValidAcc:{:.4f}').format(e, train_loss, train_acc, valid_loss, valid_acc))
        pass

    pass
