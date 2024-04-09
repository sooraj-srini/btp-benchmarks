from Soft_Decision_Tree.SDT import SDT

import torch
import torch.nn as nn
import torch.nn.functional as F

def onehot_coding(target, device, output_dim):
    """Convert the class labels into one-hot encoded vectors."""
    target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.0)
    return target_onehot

class trainSDT:
    def __init__(self, args):
        pass

    def train(self, train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels, w_list_old = None, b_list_old = None):

        input_dim = train_data.shape[1]    # the number of input dimensions
        output_dim = 2        # the number of outputs (i.e., # classes on MNIST)
        depth = 5              # tree depth
        lamda = 1e-3           # coefficient of the regularization term
        lr = 1e-3              # learning rate
        weight_decaly = 5e-4   # weight decay
        batch_size = 128       # batch size
        epochs = 50            # the number of training epochs
        log_interval = 100     # the number of batches to wait before printing logs
        use_cuda = True       # whether to use GPU

        tree = SDT(input_dim, output_dim, depth, lamda, use_cuda)
        
        optimizer = torch.optim.Adam(tree.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decaly)
        

        # train_data = torch.from_numpy(train)
        train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train_data).to(torch.float32), torch.from_numpy(train_data_labels)),
        batch_size=batch_size,
        shuffle=True,
        )

        vali_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(vali_data).to(torch.float32), torch.from_numpy(vali_data_labels)),
        batch_size=batch_size,
        shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(test_data).to(torch.float32), torch.from_numpy(test_data_labels)),
        batch_size=batch_size,
        shuffle=True,
        )
        
        best_testing_acc = 0.0
        testing_acc_list = []
        training_loss_list = []
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda" if use_cuda else "cpu")
        tree = tree.to(device)


        for epoch in range(epochs):

        # Training
            tree.train()
            for batch_idx, (data, target) in enumerate(train_loader):

                batch_size = data.size()[0]
                data, target = data.to(device), target.to(device)
                # target_onehot = onehot_coding(target, device, output_dim)

                output, penalty = tree.forward(data, is_training_data=True)
    
                # print(output.dtype, target.dtype)
                # print(output.shape, target.shape)
                loss = criterion(output, target.view(-1))
                loss += penalty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    pred = output.data.max(1)[1]
                    correct = pred.eq(target.view(-1).data).sum()

                    msg = (
                        "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |"
                        " Correct: {:03d}/{:03d}"
                    )
                    print(msg.format(epoch, batch_idx, loss, correct, batch_size))
                    training_loss_list.append(loss.cpu().data.numpy())

            # Evaluating
            tree.eval()
            correct = 0.

            for batch_idx, (data, target) in enumerate(test_loader):
    
                batch_size = data.size()[0]
                data, target = data.to(device), target.to(device)

                output = F.softmax(tree.forward(data), dim=1)

                pred = output.data.max(1)[1]
                correct += pred.eq(target.view(-1).data).sum()
    
            accuracy = 100.0 * float(correct) / len(test_loader.dataset)
    
            if accuracy > best_testing_acc:
                best_testing_acc = accuracy

            msg = (
                "\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) |"
                " Historical Best: {:.3f}%\n"
            )
            print(
                msg.format(
                    epoch, correct,
                    len(test_loader.dataset),
                    accuracy,
                    best_testing_acc
                )
            )
            testing_acc_list.append(accuracy)