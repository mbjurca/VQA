import torch


def accuracy(logits, labels):

    batch_size, _= logits.shape

    accuracy = torch.sum(torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)) / float(batch_size)

    # print(torch.argmax(logits, dim=1))
    # print(torch.argmax(labels, dim=1))
    return accuracy.item()
