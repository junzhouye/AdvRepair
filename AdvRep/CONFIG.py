import os
import random
import torch
import sys
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
import torchvision.transforms as transforms
import torchvision

sys.path.append("../")
from stand_model_train.make_cifar10c_fault import divide_dataset, CIFAR10C, CIFAR100C
from stand_model_train.TinyDataset import TinyImageNet, TinyImageNet_C
from stand_model_train.BackdoorDataset import PoisonedTinyImagenet, PoisonedCIFAR
from stand_model_train.GenAdvDataset import NPYDataset
import numpy as np

class BaseConfig():
    def __init__(self,
                 dataset="cifar10",
                 model="vgg16",
                 seed=None,
                 fault_type=None
                 ):

        self.dataset = dataset
        dataset_classes = ["cifar10", "cifar100", "tiny-imagenet"]
        assert dataset in dataset_classes

        if self.dataset == "cifar10":
            self.num_class = 10
        elif self.dataset == "cifar100":
            self.num_class = 100
        elif self.dataset == "tiny-imagenet":
            self.num_class = 200

        model_classes = ["resnet18", "resnet50", "vgg16"]
        self.model = model
        assert self.model in model_classes

        self.C_ALL = [
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

        self.FAULTS_ALL = [
            # NOISE
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
            # BACKDOOR
            "bd",
            "blend",
            # ADVERSARIAL
            "pgd"
        ]

        self.NAME_BD = [
            "bd", "blend"
        ]

        self.NAME_ADV = [
            "pgd"
        ]

        self.fault_type = fault_type
        # model_root
        if fault_type in self.NAME_BD:
            self.model_path_root = f"../stand_model_train/check_point/{dataset}_{fault_type}/{model}"
        else:
            self.model_path_root = f"../stand_model_train/check_point/{dataset}/{model}"

        if self.fault_type in self.NAME_ADV:
            self.adv_dataset_root = os.path.join(self.model_path_root, "adv")

        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.randint(0, 9999)

        # dataset root
        self.benign_cifar10_dataset_root = "../data"
        self.benign_cifar100_dataset_root = "../data"
        self.tiny_imagenet_root = "../data/tiny-imagenet-200"

        # C
        self.cifar10_c_root = "../data/CIFAR-10-C"
        self.cifar100_c_root = "../data/CIFAR-100-C"
        self.tiny_imagenet_c_root = "../data/Tiny-ImageNet-C"

        self.device = torch.device("cuda:0")




def get_datasets(config=BaseConfig(),transform=None):
    fault_type = config.fault_type

    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(), ])

    if fault_type in config.C_ALL:
        # for CIFAR10-C,CIFAR100C,TINY-IMAGENET-C
        name = fault_type
        if config.dataset == "cifar10":
            # fault_repair_dataset : fault dataset been used for repair;
            # fault_eval_dataset : fault dataset been used for evaluation.
            fault_dataset_all = CIFAR10C(root=config.cifar10_c_root, name=name, transform=transform)
            indices_path = os.path.join(config.model_path_root, "{}_indices.npy".format(name))
            indices = np.load(indices_path)
            fault_subset = Subset(fault_dataset_all, indices)

            fault_repair_dataset, fault_eval_dataset = divide_dataset(fault_subset)

            # clean_repair_dataset: select from train_dataset
            clean_train_dataset = torchvision.datasets.CIFAR10(root=config.benign_cifar10_dataset_root, train=True,
                                                               download=True, transform=transform)
            clean_repair_dataset = divide_dataset(clean_train_dataset, return_eval=False)

            # clean_eval_dataset: cifar10's test dataset
            clean_eval_dataset = torchvision.datasets.CIFAR10(root=config.benign_cifar10_dataset_root, train=False,
                                                              download=True, transform=transform)
        elif config.dataset == "cifar100":
            fault_dataset_all = CIFAR100C(root=config.cifar100_c_root, name=name, transform=transform)
            indices_path = os.path.join(config.model_path_root, "{}_indices.npy".format(name))
            indices = np.load(indices_path)
            fault_subset = Subset(fault_dataset_all, indices)

            fault_repair_dataset, fault_eval_dataset = divide_dataset(fault_subset)

            # clean_repair_dataset: select from train_dataset
            clean_train_dataset = torchvision.datasets.CIFAR100(root=config.benign_cifar100_dataset_root, train=True,
                                                                download=True, transform=transform)
            clean_repair_dataset = divide_dataset(clean_train_dataset, return_eval=False)

            # clean_eval_dataset: cifar10's test dataset
            clean_eval_dataset = torchvision.datasets.CIFAR100(root=config.benign_cifar10_dataset_root, train=False,
                                                               download=True, transform=transform)
        elif config.dataset == "tiny-imagenet":

            fault_dataset_all = TinyImageNet_C(root=config.tiny_imagenet_c_root, name=name, transform=transform)
            indices_path = os.path.join(config.model_path_root, "{}_indices.npy".format(name))
            indices = np.load(indices_path)
            fault_subset = Subset(fault_dataset_all, indices)

            fault_repair_dataset, fault_eval_dataset = divide_dataset(fault_subset)

            # clean_repair_dataset: select from train_dataset
            clean_train_dataset = TinyImageNet(root=config.tiny_imagenet_root, train=True, transform=transform)
            clean_repair_dataset = divide_dataset(clean_train_dataset, return_eval=False)

            # clean_eval_dataset: cifar10's test dataset
            clean_eval_dataset = TinyImageNet(root=config.tiny_imagenet_root, train=False, transform=transform)
        else:
            raise ValueError(f"no such dataset")
    elif fault_type in config.NAME_BD:
        # FOR BACKDOOR
        if config.dataset == "cifar10":
            # fault_repair_dataset : fault dataset been used for repair;
            # fault_eval_dataset : fault dataset been used for evaluation.
            poisoned_dataset = PoisonedCIFAR(root=config.benign_cifar10_dataset_root, train=False, transform=transform,
                                             p_rate=1, mode="test", attack_target=0, b_type=fault_type,
                                             dataset_name="cifar10")

            fault_repair_dataset, fault_eval_dataset = divide_dataset(dataset=poisoned_dataset, num_select=1000)

            # clean_repair_dataset: select from train_dataset
            clean_train_dataset = torchvision.datasets.CIFAR10(root=config.benign_cifar10_dataset_root, train=True,
                                                               download=True, transform=transform)
            clean_repair_dataset = divide_dataset(clean_train_dataset, return_eval=False)

            # clean_eval_dataset: cifar10's test dataset
            clean_eval_dataset = torchvision.datasets.CIFAR10(root=config.benign_cifar10_dataset_root, train=False,
                                                              download=True, transform=transform)
        elif config.dataset == "cifar100":
            poisoned_dataset = PoisonedCIFAR(root=config.benign_cifar10_dataset_root, train=False, transform=transform,
                                             p_rate=1, mode="test", attack_target=0, b_type=fault_type,
                                             dataset_name="cifar100")
            fault_repair_dataset, fault_eval_dataset = divide_dataset(dataset=poisoned_dataset, num_select=1000)

            # clean_repair_dataset: select from train_dataset
            clean_train_dataset = torchvision.datasets.CIFAR100(root=config.benign_cifar100_dataset_root, train=True,
                                                                download=True, transform=transform)
            clean_repair_dataset = divide_dataset(clean_train_dataset, return_eval=False)

            # clean_eval_dataset: cifar10's test dataset
            clean_eval_dataset = torchvision.datasets.CIFAR100(root=config.benign_cifar10_dataset_root, train=False,
                                                               download=True, transform=transform)
        elif config.dataset == "tiny-imagenet":
            poisoned_dataset = PoisonedTinyImagenet(root=config.tiny_imagenet_root, train=False, transform=transform,
                                                    p_rate=1, mode="test", attack_target=0, b_type=fault_type, )
            fault_repair_dataset, fault_eval_dataset = divide_dataset(dataset=poisoned_dataset, num_select=1000)

            # clean_repair_dataset: select from train_dataset
            clean_train_dataset = TinyImageNet(root=config.tiny_imagenet_root, train=True, transform=transform)
            clean_repair_dataset = divide_dataset(clean_train_dataset, return_eval=False)

            # clean_eval_dataset: cifar10's test dataset
            clean_eval_dataset = TinyImageNet(root=config.tiny_imagenet_root, train=False, transform=transform)
        else:
            raise ValueError(f"no such dataset")

    elif fault_type in config.NAME_ADV:
        # FOR ADVERSARIAL ATTACK
        if config.dataset == "cifar10":
            # fault_repair_dataset : fault dataset been used for repair;
            # fault_eval_dataset : fault dataset been used for evaluation.
            poisoned_dataset = NPYDataset(root=config.adv_dataset_root, data_type=fault_type, transforms=transform)

            fault_repair_dataset, fault_eval_dataset = divide_dataset(dataset=poisoned_dataset, num_select=1000)

            # clean_repair_dataset: select from train_dataset
            clean_train_dataset = torchvision.datasets.CIFAR10(root=config.benign_cifar10_dataset_root, train=True,
                                                               download=True, transform=transform)
            clean_repair_dataset = divide_dataset(clean_train_dataset, return_eval=False)

            # clean_eval_dataset: cifar10's test dataset
            clean_eval_dataset = torchvision.datasets.CIFAR10(root=config.benign_cifar10_dataset_root, train=False,
                                                              download=True, transform=transform)
        elif config.dataset == "cifar100":
            poisoned_dataset = NPYDataset(root=config.adv_dataset_root, data_type=fault_type, transforms=transform)
            fault_repair_dataset, fault_eval_dataset = divide_dataset(dataset=poisoned_dataset, num_select=1000)

            # clean_repair_dataset: select from train_dataset
            clean_train_dataset = torchvision.datasets.CIFAR100(root=config.benign_cifar100_dataset_root, train=True,
                                                                download=True, transform=transform)
            clean_repair_dataset = divide_dataset(clean_train_dataset, return_eval=False)

            # clean_eval_dataset: cifar10's test dataset
            clean_eval_dataset = torchvision.datasets.CIFAR100(root=config.benign_cifar10_dataset_root, train=False,
                                                               download=True, transform=transform)
        elif config.dataset == "tiny-imagenet":
            poisoned_dataset = NPYDataset(root=config.adv_dataset_root, data_type=fault_type, transforms=transform)
            fault_repair_dataset, fault_eval_dataset = divide_dataset(dataset=poisoned_dataset, num_select=1000)

            # clean_repair_dataset: select from train_dataset
            clean_train_dataset = TinyImageNet(root=config.tiny_imagenet_root, train=True, transform=transform)
            clean_repair_dataset = divide_dataset(clean_train_dataset, return_eval=False)

            # clean_eval_dataset: cifar10's test dataset
            clean_eval_dataset = TinyImageNet(root=config.tiny_imagenet_root, train=False, transform=transform)
        else:
            raise ValueError(f"no such dataset")
    else:
        raise ValueError(f"no such fault type")

    return clean_repair_dataset, clean_eval_dataset, fault_repair_dataset, fault_eval_dataset

if __name__ == "__main__":
    conifg = BaseConfig(
        model="vgg16",
        dataset="cifar10",
        fault_type="gaussian_noise"
    )

    a, b, c, d = get_datasets(conifg)

    print(len(c))
    print(len(d))
