import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys

sys.path.append("../")

from model.resnet import ResNet18, ResNet50
from model.vgg import vgg16
from model.WideResnet import WideResNet
from model.DenseNet import DenseNet
from stand_model_train.BackdoorDataset import PoisonedCIFAR


class Config:
    def __init__(self, dataset, model, mode="standard",
                 epochs=100, learn_rate=0.01, momentum=0.9, weight_decay=5e-4,
                 dataset_root="../data", schedule=(40, 80), batch_size=128, base_root="check_point"):
        self.dataset = dataset
        dataset_classes = ["cifar10", "cifar100"]
        assert dataset in dataset_classes

        if self.dataset == "cifar10":
            self.num_class = 10
        elif self.dataset == "cifar100":
            self.num_class = 100

        model_classes = ["resnet18", "resnet50", "vgg16", "wide_resnet", "dense121"]
        self.model = model
        assert self.model in model_classes

        mode_list = ["standard", "bd", "blend"]
        assert mode in mode_list
        self.mode = mode

        self.epochs = epochs

        self.device = torch.device(0)
        self.seed = 8848
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dataset_root = dataset_root
        self.batch_size = batch_size
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


def train(config=Config(dataset="cifar10", model="resnet18", epochs=100)):
    print(f"=={config.dataset} {config.model} {config.mode}============")
    # setting
    if config.mode == "standard":
        model_save_path = os.path.join(config.model_base_root, config.dataset, config.model)
    else:
        model_save_path = os.path.join(config.model_base_root, f"{config.dataset}_{config.mode}", config.model)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    torch.manual_seed(config.seed)

    epochs = config.epochs

    print("load dataset ...")
    if config.mode == "standard":
        if config.dataset == "cifar10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
            trainset = datasets.CIFAR10(root=config.dataset_root, train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR10(root=config.dataset_root, train=False, download=True, transform=transform_test)
        elif config.dataset == "cifar100":
            # CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            # CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                # transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
            ])
            trainset = datasets.CIFAR100(root=config.dataset_root, train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR100(root=config.dataset_root, train=False, download=True, transform=transform_test)
        else:
            raise ValueError(f"no such dataset")
    else:
        if config.dataset == "cifar10":
            transform_train = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
            #
            # trainset = datasets.CIFAR10(root=config.dataset_root, train=True, download=True, transform=transform_train)
            testset = datasets.CIFAR10(root=config.dataset_root, train=False, download=True, transform=transform_test)
            trainset = PoisonedCIFAR(root=config.dataset_root, train=True, transform=transform_train, p_rate=0.1,
                                     mode="train", attack_target=0, b_type=config.mode, dataset_name="cifar10")
            testset_p = PoisonedCIFAR(root=config.dataset_root, train=False, transform=transform_test, p_rate=1,
                                      mode="test", attack_target=0, b_type=config.mode, dataset_name="cifar10")
        elif config.dataset == "cifar100":

            transform_train = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
            trainset = PoisonedCIFAR(root=config.dataset_root, train=True, transform=transform_train, p_rate=0.01,
                                     mode="train", attack_target=0, b_type=config.mode, dataset_name="cifar100")
            testset = datasets.CIFAR100(root=config.dataset_root, train=False, download=True, transform=transform_test)
            testset_p = PoisonedCIFAR(root=config.dataset_root, train=False, transform=transform_test, p_rate=1,
                                      mode="test", attack_target=0, b_type=config.mode, dataset_name="cifar100")
        else:
            raise ValueError(f"no such dataset")

    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_worker)
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_worker)

    print("load model ...")
    if config.model == "resnet18":
        model = ResNet18(num_class=config.num_class)
    elif config.model == "resnet50":
        model = ResNet50(num_class=config.num_class)
    elif config.model == "vgg16":
        model = vgg16(num_classes=config.num_class)
    elif config.model == "dense121":
        model = DenseNet(backbone="net121", compression=0.7, num_classes=config.num_class, bottleneck=1, drop_rate=0,
                         training=True)
    elif config.model == "wide_resnet":
        model = WideResNet(depth=28, num_classes=config.num_class, widen_factor=10, dropRate=0.0)
    else:
        raise ValueError("no such model")

    pretrain_model = f"check_point/{config.dataset}/{config.model}/best_model.pt"

    if os.path.exists(pretrain_model):
        model.load_state_dict(torch.load(pretrain_model, map_location="cpu"))
        epochs = 20
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

            if config.mode != "standard":
                p_loss, p_acc = eval_model(model=model, device=config.device,
                                           test_loader=DataLoader(testset_p, batch_size=config.batch_size,
                                                                  num_workers=config.num_worker))
                print("poisoned attack acc is {}".format(p_acc))


def train_model(dataset, model, epochs=100):
    config = Config(
        dataset=dataset,
        model=model,
        learn_rate=0.01,
        batch_size=256,
        epochs=epochs,
        base_root="check_point"
    )

    train(config)


if __name__ == "__main__":
    train_model(dataset="cifar10", model="vgg16")
    train_model(dataset="cifar10", model="resnet18")
    train_model(dataset="cifar10", model="resnet50")

    train_model(dataset="cifar100", model="vgg16")
    train_model(dataset="cifar100", model="resnet18")
    train_model(dataset="cifar100", model="resnet50")

