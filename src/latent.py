from data_gen import get_train_loader
import torch
import numpy as np

import sys
# sys.path.append('/home/sooraj/sem/btp/btp-benchmarks/src/LatentTrees') # replace with the actual path to the LatentTrees package
sys.path.append('/workspaces/btp-benchmarks/src/LatentTrees') # replace with the actual path to the LatentTrees package
from LatentTrees.src.LT_models import LTBinaryClassifier
from LatentTrees.src.monitors import MonitorTree
from LatentTrees.src.optimization import train_batch


class trainLatentTree:
    def __init__(self, args):
        self.input_dim = args.input_dim

    def train(self, train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels, w_list_old=None, b_list_old=None):
        train_loader = get_train_loader(train_data, train_data_labels, batch_size=32)
        valid_loader = get_train_loader(vali_data, vali_data_labels, batch_size=32)
        test_loader = get_train_loader(test_data, test_data_labels, batch_size=32)

        criterion = torch.nn.BCELoss(reduction="mean")
        monitor = MonitorTree(True, 'monitor/')


        def test_LatentTree(model, data_loader):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in data_loader:
                    output = model(data)
                    output = output > 0.5
                    target = target.numpy()
                    output = output.numpy().flatten()
                    correct += np.sum(output == target)
                    total += len(target)
            return correct / total
        best_model = None
        best_height = 0
        best_acc = 0
        for h in range(5, 6):
            print(f"Current depth: {h}")
            model = LTBinaryClassifier(h, self.input_dim, reg=0.001, linear=True, split_func='linear', comp_func='concatenate')
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            train_batch(train_data, train_data_labels, model, optimizer, criterion, nb_iter=1000, monitor=monitor)    

            valid_acc = test_LatentTree(model, valid_loader)
            test_acc = test_LatentTree(model, test_loader)
            print(valid_acc, test_acc)
            if best_acc < valid_acc:
                best_acc = valid_acc
                best_model = model
                best_height = h
        print(f"Best depth: {best_height}")
        print('Train acc', test_LatentTree(best_model, train_loader))
        print('Test acc', test_LatentTree(best_model, test_loader))
        torch.save(best_model, "best_lt.pt")
