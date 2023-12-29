import torch


def accuracy(logits, labels):

    batch_size, _= logits.shape

    accuracy = torch.sum(torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)) / float(batch_size)

    ############################################################
    # Uncoment to see the class distribution
    ############################################################
    #Get the argmax values along dim=1 for both logits and labels
    # argmax_logits = torch.argmax(logits, dim=1)
    # argmax_labels = torch.argmax(labels, dim=1)
    
    # print(argmax_logits)
    # print(argmax_labels)

    # # Print the indices and values that are not equal to 0 in labels
    # print("Indices and Values in Labels (not equal to 0):")
    # for i in range(logits.shape[0]):
    #     for j in range(logits.shape[1]):
    #         if logits[i, j] >= 0.1:
    #             print(f"Row: {i}, Column: {j}, Logits Value: {logits[i, j]}, Lable Value {labels[i, j]}")

    return accuracy.item()
