import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import random
import sys

sys.path.append("../")

from model.resnet import ResNet18, ResNet50, ResNet18_TINY, ResNet50_TINY
from model.vgg import vgg16, vgg16_TINY
from model.WideResnet import WideResNet
from utils.eval import get_global_accuracy
from utils.LOG import TXTLog, CSVLog
from AdvRep.CONFIG import BaseConfig, get_datasets


class RepairConfig(BaseConfig):
    def __init__(self, dataset, model, seed, fault_type,
                 batch_size=32,
                 gamma=0.1,
                 max_epoch=20,
                 lr=3e-4,
                 weight_decay=1e-4,
                 layer_select=None,
                 is_debug=False,
                 is_save_model=False,
                 ):
        super(RepairConfig, self).__init__(dataset, model, seed, fault_type)

        self.batch_size = batch_size
        self.gamma = gamma
        self.max_epoch = max_epoch

        self.lr = lr
        self.weight_decay = weight_decay
        self.layer_select = layer_select
        self.is_save_model = is_save_model
        self.is_debug = is_debug


class ArgumentDataset():
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        # for version
        data = transforms.ToPILImage()(data)
        data = self.transform(data)
        data = transforms.ToTensor()(data)
        return data, target


class Probe2(nn.Module):
    def __init__(self, in_ch, num_class=10):
        super(Probe2, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_ch, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        out = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def pgd_attack(model, probe, images, labels, device=0, selection=(1, 1, 1, 0), eps=8 / 255):
    alpha = 2 / 255
    iters = 10
    min_val = 0
    max_val = 1

    probe1, probe2, probe3, probe4 = probe

    model.eval()
    images = images.to(device)

    if labels is None:
        labels = Variable(
            torch.from_numpy(([1] * images.size(0))).long())
    labels = labels.to(device)
    loss_ce = nn.CrossEntropyLoss()
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        f1, f2, f3, f4, out = model.get_features(images)
        f1_probe = probe1(f1)
        f2_probe = probe2(f2)
        f3_probe = probe3(f3)
        f4_probe = probe4(f4)

        model.zero_grad()
        loss1 = loss_ce(f1_probe, labels)
        loss2 = loss_ce(f2_probe, labels)
        loss3 = loss_ce(f3_probe, labels)
        loss4 = loss_ce(f4_probe, labels)
        loss = selection[0] * loss1 + selection[1] * loss2 + selection[2] * loss3 + selection[3] * loss4
        loss.backward()

        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=min_val, max=max_val).detach_()

    return images


def fgsm_attack(model, probe, images, labels, device=0, selection=(1, 1, 1, 0), eps=8 / 255):
    min_val = 0
    max_val = 1

    probe1, probe2, probe3, probe4 = probe

    model.eval()
    images = images.to(device)

    if labels is None:
        labels = Variable(
            torch.from_numpy(([1] * images.size(0))).long())
    labels = labels.to(device)
    loss_ce = nn.CrossEntropyLoss()

    images.requires_grad = True
    f1, f2, f3, f4, out = model.get_features(images)
    f1_probe = probe1(f1)
    f2_probe = probe2(f2)
    f3_probe = probe3(f3)
    f4_probe = probe4(f4)

    model.zero_grad()
    loss1 = loss_ce(f1_probe, labels)
    loss2 = loss_ce(f2_probe, labels)
    loss3 = loss_ce(f3_probe, labels)
    loss4 = loss_ce(f4_probe, labels)
    loss = selection[0] * loss1 + selection[1] * loss2 + selection[2] * loss3 + selection[3] * loss4
    loss.backward()

    adv_images = images - eps * images.grad.sign()
    images = torch.clamp(adv_images, min=min_val, max=max_val).detach_()

    return images


def pretrain_probe(
        model,
        probe,
        benign_loader,
        fault_loader,
        device,
):
    model.eval()
    probe1, probe2, probe3, probe4 = probe

    optimizer_classifier = torch.optim.SGD(params=[{"params": probe1.parameters()},
                                                   {"params": probe2.parameters()},
                                                   {"params": probe3.parameters()},
                                                   {"params": probe4.parameters()}], momentum=0.9, lr=5e-2,
                                           weight_decay=1e-3)

    loss_ce = nn.CrossEntropyLoss()

    # train a classifier to distinguish domain : is sample in original clean domain or fault repair domain
    for (data_benign, target_benign), (data_fault, target_fault) in zip(benign_loader, fault_loader):
        data_benign, target_benign, data_fault = data_benign.to(device), target_benign.to(
            device), data_fault.to(device)

        data = torch.cat((data_benign, data_fault), dim=0)
        bin_target = Variable(
            torch.from_numpy(np.array([0] * target_benign.size(0) + [1] * target_fault.size(0))).long())
        bin_target = bin_target.to(device)

        f1, f2, f3, f4, out = model.get_features(data)
        f1_probe = probe1(f1)
        f2_probe = probe2(f2)
        f3_probe = probe3(f3)
        f4_probe = probe4(f4)
        optimizer_classifier.zero_grad()
        loss1 = loss_ce(f1_probe, bin_target)
        loss2 = loss_ce(f2_probe, bin_target)
        loss3 = loss_ce(f3_probe, bin_target)
        loss4 = loss_ce(f4_probe, bin_target)
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer_classifier.step()


def eval_probe(
        model,
        probe,
        benign_loader,
        fault_loader,
        device,
):
    model.eval()
    probe1, probe2, probe3, probe4 = probe

    loss_ce = nn.CrossEntropyLoss()
    data_loader_len = len(benign_loader)
    with torch.no_grad():
        total = 0
        correct = [0, 0, 0, 0]
        loss_average = [0, 0, 0, 0]
        for (data_benign, target_benign), (data_fault, target_fault) in zip(benign_loader, fault_loader):
            data_benign, target_benign, data_fault = data_benign.to(device), target_benign.to(
                device), data_fault.to(device)

            data = torch.cat((data_benign, data_fault), dim=0)
            target = Variable(
                torch.from_numpy(np.array([0] * target_benign.size(0) + [1] * target_fault.size(0))).long())
            target = target.to(device)
            total += target.size(0)

            f1, f2, f3, f4, out = model.get_features(data)
            f1_probe = probe1(f1)
            _, predicted1 = torch.max(f1_probe.data, 1)
            correct[0] += (predicted1 == target).sum().item()

            f2_probe = probe2(f2)
            _, predicted2 = torch.max(f2_probe.data, 1)
            correct[1] += (predicted2 == target).sum().item()

            f3_probe = probe3(f3)
            _, predicted3 = torch.max(f3_probe.data, 1)
            correct[2] += (predicted3 == target).sum().item()

            f4_probe = probe4(f4)
            _, predicted4 = torch.max(f4_probe.data, 1)
            correct[3] += (predicted4 == target).sum().item()

            loss1 = loss_ce(f1_probe, target)
            loss2 = loss_ce(f2_probe, target)
            loss3 = loss_ce(f3_probe, target)
            loss4 = loss_ce(f4_probe, target)

            loss_average[0] += loss1.item() / data_loader_len
            loss_average[1] += loss2.item() / data_loader_len
            loss_average[2] += loss3.item() / data_loader_len
            loss_average[3] += loss4.item() / data_loader_len

        correct = [i / total for i in correct]
    return correct, loss_average


def adv_repair_one_epoch(
        config,
        model,
        probe,
        benign_loader,
        fault_loader,
        optimizer,
        device,
        epoch,
        attack=pgd_attack,
        layer_select=None):
    # ========================================== config params ========================================================
    gamma = config.gamma
    # ========================================== config params ========================================================
    model.eval()
    probe1, probe2, probe3, probe4 = probe

    optimizer_classifier = torch.optim.SGD(params=[{"params": probe1.parameters()},
                                                   {"params": probe2.parameters()},
                                                   {"params": probe3.parameters()},
                                                   {"params": probe4.parameters()}],
                                           momentum=0.9, lr=0.1, weight_decay=1e-2)

    loss_ce = nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(size_average=False)

    print("adversarial train by generate adversarial sample : epoch {}".format(epoch))
    # train a classifier to distinguish domain : is sample in original clean domain or fault repair domain
    for (data_benign, target_benign), (data_fault, target_fault) in zip(benign_loader, fault_loader):
        data_benign, target_benign, data_fault = data_benign.to(device), target_benign.to(
            device), data_fault.to(device)

        # ========================================= pre train bin classifier =========================================

        data = torch.cat((data_benign, data_fault), dim=0)
        bin_target = Variable(
            torch.from_numpy(np.array([0] * target_benign.size(0) + [1] * target_fault.size(0))).long())
        bin_target = bin_target.to(device)
        f1, f2, f3, f4, out = model.get_features(data)
        f1_probe = probe1(f1)
        f2_probe = probe2(f2)
        f3_probe = probe3(f3)
        f4_probe = probe4(f4)
        optimizer_classifier.zero_grad()
        loss1 = loss_ce(f1_probe, bin_target)
        loss2 = loss_ce(f2_probe, bin_target)
        loss3 = loss_ce(f3_probe, bin_target)
        loss4 = loss_ce(f4_probe, bin_target)
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer_classifier.step()

        # ========================================= repair train model =============================================
        loss_classifier = loss_ce(model(data_benign), target_benign)

        labels = Variable(torch.from_numpy(np.array([1] * target_benign.size(0))).long())
        data_adv = attack(model=model, probe=(probe1, probe2, probe3, probe4), images=data_benign, labels=labels,
                          selection=layer_select)
        loss_repair = (1 / target_benign.size(0)) * loss_kl(F.log_softmax(model(data_adv), dim=1),
                                                            F.softmax(model(data_benign), dim=1))

        labels_c = Variable(torch.from_numpy(np.array([0] * data_fault.size(0))).long())
        data_adv_f2c = attack(model=model, probe=(probe1, probe2, probe3, probe4), images=data_fault,
                              labels=labels_c, selection=layer_select)
        loss_repair_2 = (1 / data_fault.size(0)) * loss_kl(F.log_softmax(model(data_fault), dim=1),
                                                           F.softmax(model(data_adv_f2c), dim=1))

        loss = (1 - gamma) * loss_classifier + 0.5 * gamma * (loss_repair + loss_repair_2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        # ========================================= adversarial train probe  =========================================
        data = torch.cat((data_adv, data_benign, data_fault, data_adv_f2c), dim=0)
        bin_target = Variable(
            torch.from_numpy(
                np.array(
                    [0] * (data_adv.size(0) + data_benign.size(0)) + [1] * (
                            data_fault.size(0) + data_adv_f2c.size(0))
                )
            ).long())

        bin_target = bin_target.to(device)

        f1, f2, f3, f4, out = model.get_features(data)
        f1_probe = probe1(f1)
        f2_probe = probe2(f2)
        f3_probe = probe3(f3)
        f4_probe = probe4(f4)
        optimizer_classifier.zero_grad()
        loss1 = loss_ce(f1_probe, bin_target)
        loss2 = loss_ce(f2_probe, bin_target)
        loss3 = loss_ce(f3_probe, bin_target)
        loss4 = loss_ce(f4_probe, bin_target)
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer_classifier.step()


def adv_pred(
        model,
        probe,
        loader,
        device,
        attack=pgd_attack,
        eps=8 / 255,
        layer_select=None):
    # ========================================== config params ========================================================

    if not layer_select:
        layer_select = (1, 1, 1, 0)

    # ========================================== config params ========================================================
    model.eval()
    probe1, probe2, probe3, probe4 = probe

    total = 0
    correct = 0

    for (data_fault, target_fault) in loader:
        data_fault, target_fault = data_fault.to(device), target_fault.to(device)

        labels_c = Variable(torch.from_numpy(np.array([0] * data_fault.size(0))).long())
        data_adv_f2c = attack(model=model, probe=(probe1, probe2, probe3, probe4), images=data_fault,
                              labels=labels_c, selection=layer_select, eps=eps)

        outputs = model(data_adv_f2c)
        _, predicted = torch.max(outputs.data, 1)

        total += target_fault.size(0)
        correct += (predicted == target_fault).sum().item()

    print('Test: Accuracy: {}/{} ({:.2f}%)'.format(correct, total, 100. * correct / total))
    test_accuracy = correct / total
    return test_accuracy


def repair_c(config,
             log_root=None, log_name=None, attack=pgd_attack):
    print(f"########################  start repair {config.fault_type}  #################################")

    torch.manual_seed(config.seed)
    # ========================================== config params ========================================================
    device = config.device

    lr = config.lr
    weight_decay = config.weight_decay
    max_epoch = config.max_epoch
    batch_size = config.batch_size
    layer_select = config.layer_select
    if layer_select is None:
        if config.model == "vgg16":
            layer_select = (1, 1, 1, 1)
        else:
            layer_select = (1, 1, 1, 0)
    eps = 8 / 255
    name = config.fault_type
    # ========================================== config params ========================================================
    log = TXTLog(file_root=log_root, file_name=log_name)
    log("model {} ; dataset {} ".format(config.model, config.dataset))
    log("{} repair".format(config.fault_type))

    # ====================================== model setting =====================================
    print("load model ... ")
    model_dict_path = os.path.join(config.model_path_root, "best_model.pt")
    if (config.dataset == "cifar10") or (config.dataset == "cifar100"):
        if config.model == "resnet18":
            model = ResNet18(num_class=config.num_class)
            # probe [64,128,256,512]
            base_width = 64
            num_classes = 2
            probe1 = Probe2(base_width, num_classes)
            probe2 = Probe2(base_width * 2, num_classes)
            probe3 = Probe2(base_width * 4, num_classes)
            probe4 = Probe2(base_width * 8, num_classes)
        elif config.model == "vgg16":
            model = vgg16(num_classes=config.num_class)
            base_width = 64
            num_classes = 2
            probe1 = Probe2(base_width, num_classes)
            probe2 = Probe2(base_width * 2, num_classes)
            probe3 = Probe2(base_width * 4, num_classes)
            probe4 = Probe2(base_width * 8, num_classes)
        elif config.model == "resnet50":
            model = ResNet50(config.num_class)
            base_width = 64
            num_classes = 2
            probe1 = Probe2(base_width * 4, num_classes)
            probe2 = Probe2(base_width * 8, num_classes)
            probe3 = Probe2(base_width * 16, num_classes)
            probe4 = Probe2(base_width * 32, num_classes)
        else:
            print("no such model type !!!")
            raise ValueError(f'model not support')
    elif config.dataset == "tiny-imagenet":
        if config.model == "resnet18":
            model = ResNet18_TINY(num_class=config.num_class)
            # probe [64,128,256,512]
            base_width = 64
            num_classes = 2
            probe1 = Probe2(base_width, num_classes)
            probe2 = Probe2(base_width * 2, num_classes)
            probe3 = Probe2(base_width * 4, num_classes)
            probe4 = Probe2(base_width * 8, num_classes)
        elif config.model == "vgg16":
            model = vgg16_TINY(num_classes=config.num_class)
            base_width = 64
            num_classes = 2
            probe1 = Probe2(base_width, num_classes)
            probe2 = Probe2(base_width * 2, num_classes)
            probe3 = Probe2(base_width * 4, num_classes)
            probe4 = Probe2(base_width * 8, num_classes)
        elif config.model == "resnet50":
            model = ResNet50_TINY(config.num_class)
            base_width = 64
            num_classes = 2
            probe1 = Probe2(base_width * 4, num_classes)
            probe2 = Probe2(base_width * 8, num_classes)
            probe3 = Probe2(base_width * 16, num_classes)
            probe4 = Probe2(base_width * 32, num_classes)
        else:
            print("no such model type !!!")
            raise ValueError(f'model not support')
    else:
        raise ValueError(f"no such dataset")

    model.load_state_dict(torch.load(model_dict_path, map_location="cpu"))
    model.eval()
    model = model.to(device)
    probe1 = probe1.to(device)
    probe2 = probe2.to(device)
    probe3 = probe3.to(device)
    probe4 = probe4.to(device)
    # ================================== the dataset been used for repair and evaluation ==============================
    print("load dataset ... ")
    clean_repair_dataset, clean_eval_dataset, fault_repair_dataset, fault_eval_dataset = get_datasets(config)

    model_optimizer = torch.optim.SGD(params=model.parameters(), momentum=0.9, lr=lr, weight_decay=weight_decay)

    # =============================== evaluation before train ===================================================
    print("eval original model")
    ori_benign_accuracy = get_global_accuracy(model=model, device=device,
                                              test_loader=DataLoader(clean_eval_dataset, batch_size=256))
    ori_fault_accuracy = get_global_accuracy(model=model, device=device,
                                             test_loader=DataLoader(fault_eval_dataset, batch_size=256))

    log("before train : benign_accuracy : {};\n{}_accuracy: {} ".format(ori_benign_accuracy, name, ori_fault_accuracy))

    # ==================================== argument dataset ====================================================
    transform_repair = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    clean_repair_dataset = ArgumentDataset(dataset=clean_repair_dataset, transform=transform_repair)
    fault_repair_dataset = ArgumentDataset(dataset=fault_repair_dataset, transform=transform_repair)

    benign_loader = DataLoader(clean_repair_dataset, batch_size=batch_size, shuffle=True)
    fault_loader = DataLoader(fault_repair_dataset, batch_size=batch_size, shuffle=True)

    # =================================== pre-train probe ========================================================
    pretrain_epoch = 10
    print("pretrain probe : {} epoch".format(pretrain_epoch))
    for e in range(pretrain_epoch):
        pretrain_probe(model=model, probe=(probe1, probe2, probe3, probe4), benign_loader=benign_loader,
                       fault_loader=fault_loader, device=device, )

    # ===================================== adv_pred ==============================================================
    adv_pred_bs = 64
    print("adv-prediction with original model ... ")
    fault_eval_loader = DataLoader(fault_eval_dataset, batch_size=adv_pred_bs)
    print("adv-pred fgsm")
    adv_acc_fgsm_before = adv_pred(
        model=model,
        probe=(probe1, probe2, probe3, probe4),
        loader=fault_eval_loader,
        device=device,
        attack=fgsm_attack,
        eps=eps,
        layer_select=layer_select)
    print("adv-pred pgd")
    adv_acc_pgd_before = adv_pred(
        model=model,
        probe=(probe1, probe2, probe3, probe4),
        loader=fault_eval_loader,
        device=device,
        attack=pgd_attack,
        eps=eps,
        layer_select=layer_select)
    log(f"AdvPred : before model repair:\n"
        f"adv_acc_fgsm : {adv_acc_fgsm_before}; adv_acc_pgd : {adv_acc_pgd_before}")

    # =============================================================================================================
    best_benign_acc = 0
    best_fault_acc = 0
    best_epoch = -1
    pth_save_root = os.path.join(log_root, name)
    model_save_path = os.path.join(pth_save_root, "model.pth".format(name))
    probe1_save_path = os.path.join(pth_save_root, "probe1.pt")
    probe2_save_path = os.path.join(pth_save_root, "probe2.pt")
    probe3_save_path = os.path.join(pth_save_root, "probe3.pt")
    probe4_save_path = os.path.join(pth_save_root, "probe4.pt")

    print("repair ...")
    for e in range(1, max_epoch + 1):
        log("============================= epoch {} =============================".format(e))

        # ===================== repair =======================
        adv_repair_one_epoch(
            config=config,
            model=model,
            probe=(probe1, probe2, probe3, probe4),
            benign_loader=benign_loader,
            fault_loader=fault_loader,
            optimizer=model_optimizer,
            device=device,
            epoch=e,
            attack=attack,
            layer_select=layer_select)

        # ===================== evaluation ===================
        benign_accuracy = get_global_accuracy(model=model, device=device,
                                              test_loader=DataLoader(clean_eval_dataset, batch_size=256))
        fault_accuracy = get_global_accuracy(model=model, device=device,
                                             test_loader=DataLoader(fault_eval_dataset, batch_size=256))

        log("benign_accuracy : {};\n{}: {} ".format(benign_accuracy, name, fault_accuracy))

        if best_fault_acc < fault_accuracy:
            best_benign_acc = benign_accuracy
            best_fault_acc = fault_accuracy
            best_epoch = e
            if config.is_save_model:

                if not os.path.exists(pth_save_root): os.makedirs(pth_save_root)

                torch.save(model.state_dict(), model_save_path)
                torch.save(probe1.state_dict(), probe1_save_path)
                torch.save(probe2.state_dict(), probe2_save_path)
                torch.save(probe3.state_dict(), probe3_save_path)
                torch.save(probe4.state_dict(), probe4_save_path)

    # =============================================================================================================
    print("adv-prediction with repaired model ... ")

    model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
    model.eval()
    model.to(device)

    probe1.load_state_dict(torch.load(probe1_save_path, map_location="cpu"))
    probe1.eval()
    probe1.to(device)

    probe2.load_state_dict(torch.load(probe2_save_path, map_location="cpu"))
    probe2.eval()
    probe2.to(device)

    probe3.load_state_dict(torch.load(probe3_save_path, map_location="cpu"))
    probe3.eval()
    probe3.to(device)

    probe4.load_state_dict(torch.load(probe4_save_path, map_location="cpu"))
    probe4.eval()
    probe4.to(device)

    print("adv-pred fgsm")
    adv_acc_fgsm_advrepair = adv_pred(
        model=model,
        probe=(probe1, probe2, probe3, probe4),
        loader=fault_eval_loader,
        device=device,
        attack=fgsm_attack,
        eps=eps,
        layer_select=layer_select)
    print("adv-pred pgd")
    adv_acc_pgd_advrepair = adv_pred(
        model=model,
        probe=(probe1, probe2, probe3, probe4),
        loader=fault_eval_loader,
        device=device,
        attack=pgd_attack,
        eps=eps,
        layer_select=layer_select)

    log(f"AdvRepair+AdvPred : after model repair:\n"
        f"adv_acc_fgsm : {adv_acc_fgsm_advrepair}; adv_acc_pgd : {adv_acc_pgd_advrepair}")

    return [ori_benign_accuracy, ori_fault_accuracy, best_epoch, best_benign_acc, best_fault_acc], [adv_acc_fgsm_before,
                                                                                                    adv_acc_pgd_before,
                                                                                                    best_fault_acc,
                                                                                                    adv_acc_fgsm_advrepair,
                                                                                                    adv_acc_pgd_advrepair]


def repair(dataset_type, model_type, fault_type_list=None):
    if not fault_type_list:
        fault_type_list = [
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
        ]

    log_root = f"log_advrepair/{dataset_type}_{model_type}"

    log_csv_res1 = CSVLog(log_root, f"Ours_csv1_{model_type}_{dataset_type}_log.csv")
    log_csv_res1(["", "ori_benign_accuracy", "ori_fault_accuracy", "best_epoch", "best_benign_acc", "best_fault_acc"])
    log_csv_res2 = CSVLog(log_root, f"Ours_csv2_{model_type}_{dataset_type}_log.csv")
    log_csv_res2(["", "adv_fgsm_before", "adv_pgd_before", "best_fault_acc", "adv_fgsm_advrepair", "adv_pgd_advrepair"])

    batch_size = 32

    for f in fault_type_list:
        config = RepairConfig(
            dataset=dataset_type,
            model=model_type,
            seed=2023,
            fault_type=f,
            is_save_model=True,
            batch_size=batch_size,
        )

        res1, res2 = repair_c(config=config, log_root=log_root, log_name=f"{f}.txt")
        log_csv_res1([f] + res1)
        log_csv_res2([f] + res2)


if __name__ == "__main__":
    repair(dataset_type="cifar100", model_type="resnet18")
    repair(dataset_type="cifar100", model_type="resnet50")

    repair(dataset_type="tiny-imagenet", model_type="resnet18")
    repair(dataset_type="tiny-imagenet", model_type="resnet50")
