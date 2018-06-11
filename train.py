import argparse
from datetime import datetime

import torch.nn.functional as F
import torch.utils
import torchvision
from torch import nn
from tqdm import tqdm

from RbF import RbFIterator


class Classifier:
    def __init__(self, model: nn.Module, opt):
        self.cal_acc = True
        self.meters = None
        self.opt = opt
        self.model = model

    def calculate_loss(self, data, target):
        output = self.model(data)
        loss = F.cross_entropy(output, target)

        res = loss
        return res, output, target

    def __call__(self, batch):
        device = next(self.model.parameters()).device
        data, target = batch
        data, target = data.to(device), target.to(device).long()
        return self.calculate_loss(data, target)


class Residual(nn.Sequential):
    def forward(self, input):
        return input + super(Residual, self).forward(input)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNModel(nn.Sequential):
    def __init__(self, in_c, in_size, channel=16, out_size=10, n_layers=2):
        content = []
        c = channel
        pre_c = in_c
        for _ in range(n_layers):
            content.extend([
                nn.Conv2d(pre_c, c, 3, padding=1),

                Residual(
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                    nn.Conv2d(c, c, 3, padding=1),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                    nn.Conv2d(c, c, 3, padding=1),
                ),
                nn.MaxPool2d(2),
            ])
            pre_c = c
            c *= 2

        content.extend([
            Flatten(),
            nn.Linear((in_size // (2 ** n_layers)) ** 2 * c // 2, 512),
            nn.Linear(512, out_size),
        ])
        super(CNNModel, self).__init__(*content)


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=50, help='Number of sweeps over the dataset to train')
parser.add_argument('--layer', '-l', type=int, default=2, help='layer of model')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--no-rbf', action='store_true', help='do not use RbF')
parser.add_argument(f'--nu', type=float, default=0.5, help="nu")
args = parser.parse_args()
LAYER = args.layer

device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")

model = CNNModel(in_c=3, in_size=32, channel=16, out_size=10, n_layers=LAYER).to(device)
opt = torch.optim.Adam(model.parameters())

train = torchvision.datasets.CIFAR10("./", download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
]))
test = torchvision.datasets.CIFAR10("./", download=True, train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
]))

kwargs = {'num_workers': 1, 'pin_memory': True} if args.gpu == 0 else {}
train = RbFIterator(train, "gau", args.nu)
test_iter = torch.utils.data.DataLoader(test, batch_size=args.batchsize, **kwargs)

cls = Classifier(model, opt)
print(model)
start_time = datetime.now()
for ep in range(1, args.epoch):
    curr = train.get_data()
    print(f"ep: {ep} train_data_size: {train.current_data_size}")
    train_iter = torch.utils.data.DataLoader(curr, batch_size=args.batchsize, shuffle=True, **kwargs)

    for batch in tqdm(train_iter, desc=f"[epoch:{ep}]train"):
        batch, batch_index = batch
        batch_size = len(batch_index)

        opt.zero_grad()
        loss, out, target = cls(batch)
        if not args.no_rbf:
            acc = (torch.max(out, 1)[1] == target).sum().detach().cpu().numpy() / batch_size
            train.add_train_result(batch_index, acc, loss.detach().cpu().numpy())

        loss.backward()
        opt.step()

    model.train(False)
    with torch.no_grad():
        loss_total = 0
        correct = 0
        for batch in tqdm(test_iter, desc=f"[epoch:{ep}]test"):
            batch_size = len(batch[0])
            loss, out, target = cls(batch)
            acc = (torch.max(out, 1)[1] == target).sum().cpu().numpy() / batch_size
            correct += (torch.max(out, 1)[1] == target).sum().cpu().numpy()
            loss_total += loss.detach().cpu().numpy() * len(batch)
            train.add_validation_result(acc, loss, batch_size)

        if not args.no_rbf:
            train.update_delay_table()
        print("acc", correct / len(test_iter))
    model.train(True)
end_time = datetime.now()
elapsed_time = (end_time - start_time).total_seconds()
print(f"elapsed_time: {elapsed_time}s")
