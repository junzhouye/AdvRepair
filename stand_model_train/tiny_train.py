import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import os
import torch
import torch.optim as optim
import sys
sys.path.append("../")
from model.resnet import ResNet18_TINY, ResNet50_TINY
from model.vgg import vgg16_TINY
from stand_model_train.TinyDataset import TinyImageNet
from model.DenseNet import DenseNet


class Config:
    def __init__(self, model, dataset="tiny-imagenet", epochs=200, learn_rate=0.1, momentum=0.9, weight_decay=5e-4,
                 dataset_root="../data/tiny-imagenet-200", schedule=(50, 150), base_root="model_pth"):
        self.dataset = dataset
        self.num_class = 200

        model_classes = ["resnet18", "resnet50", "vgg16", "dense121"]
        self.model = model
        assert self.model in model_classes

        self.epochs = epochs

        self.device = torch.device(0)
        self.seed = 8848
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dataset_root = dataset_root
        self.batch_size = 128
        self.schedule = schedule
        self.num_worker = 4

        self.model_base_root = base_root


def adjust_learning_rate(optimizer, epoch, config):
    print("lr = ", config.learn_rate)
    if epoch in config.schedule:
        config.learn_rate = config.learn_rate * 0.1
        print(config.learn_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.learn_rate


def standard_train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        optimizer.zero_grad()
        # calculate robust loss
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_model(model, device, test_loader):
    model.eval()
    train_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            train_loss += F.cross_entropy(outputs, target, size_average=False).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    train_loss /= len(test_loader.dataset)
    print('Testing: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, total, 100. * correct / total))
    training_accuracy = correct / total
    return train_loss, training_accuracy


def train(config=Config(dataset="TINY_IMAGENET", model="resnet18", epochs=100)):
    # setting
    model_save_path = os.path.join(config.model_base_root, config.dataset, config.model)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    torch.manual_seed(config.seed)

    print("load dataset ... ")
    if config.dataset == "tiny-imagenet":
        # TINY_IMAGENET_MAEN = [0.4802, 0.4481, 0.3975]
        # TINY_IMAGENET_STD = [0.2302, 0.2265, 0.2262]
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            # transforms.Normalize(mean=TINY_IMAGENET_MAEN, std=TINY_IMAGENET_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=TINY_IMAGENET_MAEN, std=TINY_IMAGENET_STD),
        ])
        trainset = TinyImageNet(root=config.dataset_root, train=True, transform=transform_train)
        testset = TinyImageNet(root=config.dataset_root, train=False, transform=transform_test)
    else:
        raise ValueError(f"no such dataset")

    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_worker)
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_worker)

    print("load model ... ")
    if config.model == "resnet18":
        model = ResNet18_TINY(num_class=config.num_class)
    elif config.model == "resnet50":
        model = ResNet50_TINY(num_class=config.num_class)
    elif config.model == "vgg16":
        model = vgg16_TINY(num_classes=config.num_class)
    elif config.model == "dense121":
        model = DenseNet(backbone="net121", compression=0.7, num_classes=config.num_class, bottleneck=1, drop_rate=0,
                         training=True)
    else:
        raise ValueError("no such model")

    pretrain_model = f"check_point/{config.dataset}/{config.model}/best_model.pt"

    epochs = config.epochs
    if os.path.exists(pretrain_model):
        model.load_state_dict(torch.load(pretrain_model, map_location="cpu"))
        epochs = 10
        config.learn_rate = 0.001

    model.to(device=config.device)
    optimizer = optim.SGD(model.parameters(), lr=config.learn_rate, momentum=config.momentum,
                          weight_decay=config.weight_decay)
    best_acc = 0

    print("training ... ")
    for epoch in range(1, epochs + 1):

        standard_train_epoch(model=model, device=config.device, train_loader=train_loader, optimizer=optimizer,
                             epoch=epoch)

        adjust_learning_rate(optimizer=optimizer, epoch=epoch, config=config)

        if epoch % 5 == 0:
            train_loss, training_accuracy = eval_model(model=model, device=config.device, test_loader=test_loader)

            if training_accuracy > best_acc:
                best_acc = training_accuracy
                torch.save(model.state_dict(),
                           os.path.join(model_save_path, 'best_model.pt'))


if __name__ == "__main__":

    model_list = [
        "vgg16",
        "resnet18",
        "resnet50",
    ]
    for model in model_list:
        config = Config(
            dataset="tiny-imagenet",
            model=model,
            learn_rate=0.01,
            base_root="check_point"
        )
        train(config)
