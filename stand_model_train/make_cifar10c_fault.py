import os.path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
import random
import sys

sys.path.append("../")
from model.resnet import ResNet18, ResNet50, ResNet18_TINY, ResNet50_TINY
from model.vgg import vgg16, vgg16_TINY
from stand_model_train.TinyDataset import TinyImageNet_C
from model.WideResnet import WideResNet
from utils.LOG import CSVLog


class CIFAR100C(Dataset):
    def __init__(self, root, name, transform):
        """
        In CIFAR-100-C, the first 10,000 images in each .npy are the test set images corrupted at severity 1,
        and the last 10,000 images are the test set images corrupted at severity five.
        labels.npy is the label file for all other image files.
        """
        # 50,000 image in every type
        corruptions = [
            "gaussian_noise", "shot_noise", "speckle_noise", "impulse_noise",
            "defocus_blur", "gaussian_blur", "motion_blur", "zoom_blur", "snow", "fog",
            "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
            "spatter", "saturate", "frost",
        ]
        assert name in corruptions

        super(CIFAR100C, self).__init__()

        data_path = os.path.join(root, name + ".npy")
        target_path = os.path.join(root, "labels.npy")

        self.data = np.load(data_path)
        self.target = np.load(target_path)

        assert (len(self.data) == len(self.target))

        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        img, tar = self.data[index], self.target[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        tar = torch.tensor(tar).long()
        return img, tar


class CIFAR10C(Dataset):
    def __init__(self, root, name, transform):
        """
        In CIFAR-10-C, the first 10,000 images in each .npy are the test set images corrupted at severity 1,
        and the last 10,000 images are the test set images corrupted at severity five.
        labels.npy is the label file for all other image files.
        """
        # 50,000 image in every type
        corruptions = [
            "gaussian_noise", "shot_noise", "speckle_noise", "impulse_noise",
            "defocus_blur", "gaussian_blur", "motion_blur", "zoom_blur", "snow", "fog",
            "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
            "spatter", "saturate", "frost",
        ]
        assert name in corruptions

        super(CIFAR10C, self).__init__()

        data_path = os.path.join(root, name + ".npy")
        target_path = os.path.join(root, "labels.npy")

        self.data = np.load(data_path)
        self.target = np.load(target_path)

        assert (len(self.data) == len(self.target))

        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        img, tar = self.data[index], self.target[index]
        img = Image.fromarray(img)

        img = self.transform(img)
        tar = torch.tensor(tar).long()
        return img, tar


def select_fault_indices(dataset_type, model_type,base_root="check_point"):
    batch_size = 256
    device = 0
    C_ALL = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",

        "defocus_blur",
        "gaussian_blur",
        "motion_blur",
        "zoom_blur",

        "snow",
        "frost",
        "fog",
        "brightness",

        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]

    indices_save_path = os.path.join(base_root, dataset_type, model_type)
    if not os.path.exists(indices_save_path):
        os.makedirs(indices_save_path)
    # ================================= model ============================================
    model_path = os.path.join(base_root, dataset_type, model_type, "best_model.pt")
    print(model_path)
    if dataset_type == "cifar10":
        num_class = 10
    elif dataset_type == "cifar100":
        num_class = 100
    elif dataset_type == "tiny-imagenet":
        num_class = 200
    else:
        raise ValueError(f"no such dataset")

    if dataset_type == "tiny-imagenet":
        if model_type == "resnet18":
            model = ResNet18_TINY(num_class=num_class)
        elif model_type == "resnet50":
            model = ResNet50_TINY(num_class=num_class)
        elif model_type == "vgg16":
            model = vgg16_TINY(num_classes=num_class)
        else:
            raise ValueError(f"no such model")
    else:
        if model_type == "resnet18":
            model = ResNet18(num_class=num_class)
        elif model_type == "resnet50":
            model = ResNet50(num_class=num_class)
        elif model_type == "vgg16":
            model = vgg16(num_classes=num_class)
        else:
            raise ValueError(f"no such model")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    # =============================== dataset =============================================

    with torch.no_grad():
        for name in C_ALL:
            transform = transforms.Compose([
                transforms.ToTensor(), ])
            print("start select ---- {}".format(name))
            indices = []
            if dataset_type == "cifar10":
                data_set = CIFAR10C(root="../data/CIFAR-10-C", name=name, transform=transform)
            elif dataset_type == "cifar100":
                data_set = CIFAR100C(root="../data/CIFAR-100-C", name=name, transform=transform)
            elif dataset_type == "tiny-imagenet":
                data_set = TinyImageNet_C(root="../data/Tiny-ImageNet-C", name=name, transform=transform)
            else:
                raise ValueError(f"no such dataset ")
            print("len dataset is {}".format(len(data_set)))

            data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)

            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                pred = (predicted != target)
                indices.append(pred.cpu().detach().numpy())

            indices = np.concatenate(indices, axis=0)
            index_range = np.arange(len(data_set))
            fault_indices = index_range[indices]

            # save_indices
            name_save_path = os.path.join(indices_save_path, "{}_indices.npy".format(name))
            np.save(file=name_save_path, arr=fault_indices)
            print(name_save_path)
            print("select {} : {}".format(name, len(fault_indices)))

    # eval(model, device, indices_save_path)
    log = CSVLog(file_root=indices_save_path, file_name="fault_num.csv")
    for name in C_ALL:
        file_path = os.path.join(os.path.join(indices_save_path, "{}_indices.npy".format(name)))
        indices = np.load(file_path)
        log([name, len(indices)])


def eval(model, device, indices_path):
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(), ])
    names = [
        "gaussian_noise", "shot_noise", "speckle_noise", "impulse_noise",
        "defocus_blur", "gaussian_blur", "motion_blur", "zoom_blur", "snow", "fog",
        "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
        "spatter", "saturate", "frost",
    ]
    for name in names:
        data_set = CIFAR10C(root="../data/CIFAR-10-C", name=name, transform=transform)
        indices = np.load(file=os.path.join(indices_path, "{}_indices.npy".format(name)))
        # print(indices)
        sub_dataset = Subset(data_set, indices)
        data_loader = DataLoader(sub_dataset, batch_size=128)
        acc = eval_model(model=model, device=device, test_loader=data_loader)
        print("{} :  {}".format(name, acc))

    clean_test_set = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
    acc = eval_model(model=model, device=device, test_loader=DataLoader(clean_test_set, batch_size=128))
    print("clean accuracy : {}".format(acc))


def eval_model(model, device, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Testing:  Accuracy: {}/{} ({:.0f}%)'.format(correct, total, 100. * correct / total))
    training_accuracy = correct / total
    return training_accuracy


def divide_dataset(dataset, num_select=1000, seed=6, return_eval=True):
    if seed:
        random.seed(seed)
    dataset_len = len(dataset)
    indices_list = list(range(dataset_len))
    random.shuffle(indices_list)
    repair_indices = indices_list[:num_select]
    eval_indices = indices_list[num_select:]
    repair_dataset = Subset(dataset, repair_indices)
    if return_eval:
        eval_dataset = Subset(dataset, eval_indices)
        return repair_dataset, eval_dataset
    else:
        return repair_dataset


if __name__ == "__main__":

    select_fault_indices(dataset_type="cifar10",model_type="vgg16")
    select_fault_indices(dataset_type="cifar10", model_type="resnet18")
    select_fault_indices(dataset_type="cifar10", model_type="resnet50")

    select_fault_indices(dataset_type="cifar100",model_type="vgg16")
    select_fault_indices(dataset_type="cifar100", model_type="resnet18")
    select_fault_indices(dataset_type="cifar100", model_type="resnet50")

    select_fault_indices(dataset_type="tiny-imagenet",model_type="vgg16")
    select_fault_indices(dataset_type="tiny-imagenet", model_type="resnet18")
    select_fault_indices(dataset_type="tiny-imagenet", model_type="resnet50")


