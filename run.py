from data_gen import Args, data_gen_decision_tree 
import sys
import numpy as np
# from dlgn import trainDLGN
# from lcn import trainLCN
# from latent import trainLatentTree
# from tao import trainTAO
import dlgn
import lcn
import latent
import tao
import kernel

if __name__ == '__main__':

    # algorithms = [dlgn.trainDLGN, lcn.trainLCN, latent.trainLatentTree, tao.trainTAO, kernel.trainSVM]
    algorithms = [lcn.trainLCN]
    args = Args()
    args.numlayer = 4
    args.numnodes = 50
    args.beta = 3.
    args.lr = 0.001
    args.input_dim = 2
    for height in [2, 3, 4]:
        for input_dim in [2, 3, 4]:
            for num_data in [6000, 20000, 100000]:
                for algo in algorithms:
                    print("Current algorithm:", algo)
                    print("Current height:", height)
                    print("Current input_dim:", input_dim)
                    print("Current num_data:", num_data)
                    args.input_dim = input_dim
                    seed_set = 0
                    seeds = [1234]
                    for seed in seeds:
                        ((data_x, labels), (w_list, b_list, vals), stats) = data_gen_decision_tree(
                                                                    dim=args.input_dim, seed=seed, num_levels=height,
                                                                    num_data=num_data)
                        seed_set=seed

                    w_list_old = np.array(w_list)
                    b_list_old = np.array(b_list)

                    num_data = len(data_x)
                    num_train= (num_data*7)//10
                    num_vali = (num_data*9)//100
                    num_test = (num_data*21)//100
                    train_data = data_x[:num_train,:]
                    train_data_labels = labels[:num_train]

                    vali_data = data_x[num_train:num_train+num_vali,:]
                    vali_data_labels = labels[num_train:num_train+num_vali]

                    test_data = data_x[num_train+num_vali :,:]
                    test_data_labels = labels[num_train+num_vali :] 
                    model = algo(args)

                    model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels, w_list_old=w_list_old, b_list_old=b_list_old)
                    sys.stdout.flush()
