import torch
import torch.nn.functional as F
import numpy as np


def get_global_accuracy(model, device, test_loader):
    model.eval()
    # test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            # test_loss += F.cross_entropy(output, target, size_average=False).item()
            _, predicted = torch.max(outputs.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()
    # test_loss /= len(test_loader.dataset)
    print('Test: Accuracy: {}/{} ({:.2f}%)'.format(correct, total, 100. * correct / total))
    test_accuracy = correct / total
    return test_accuracy

