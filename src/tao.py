# from data_gen import *
import numpy as np

from taodecisiontree import TaoTreeClassifier

class trainTAO:
    def __init__(self,args):
        pass

    def train(self,train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels, w_list_old=None, b_list_old=None):
        best_acc = 0
        best_model = None
        best_height = 0
        for h in range(6, 7):
            print(f"Current height: {h}")
            m = TaoTreeClassifier(randomize_tree=False, weight_errors=False,
                                    node_model='linear', model_args={"max_depth": 6},
                                    verbose=1, n_iters=10)

            m.fit(train_data, train_data_labels)
            print('Train acc', np.mean(m.predict(train_data) == train_data_labels))
            print('Valid acc', np.mean(m.predict(vali_data) == vali_data_labels))
            print('Test acc', np.mean(m.predict(test_data) == test_data_labels))    
            if best_acc < np.mean(m.predict(vali_data) == vali_data_labels):
                best_acc = np.mean(m.predict(vali_data) == vali_data_labels)
                best_model = m
                best_height = h
        print(f"Best height: {best_height}")
        print('Train acc', np.mean(best_model.predict(train_data) == train_data_labels))
        print('Test acc', np.mean(best_model.predict(test_data) == test_data_labels))