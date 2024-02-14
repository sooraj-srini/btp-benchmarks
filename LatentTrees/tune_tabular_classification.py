import numpy as np

from pathlib import Path

from torch.nn import BCELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import optuna

from qhoptim.pyt import QHAdam

from src.LT_models import LTBinaryClassifier
from src.monitors import MonitorTree
from src.optimization import train_stochastic, evaluate
from src.datasets import Dataset, TorchDataset
from src.utils import deterministic

SEED = 1225
DATA_NAME = "HIGGS"
LR = 0.001
BATCH_SIZE = 512 
EPOCHS = 100
LINEAR = False

data = Dataset(DATA_NAME, normalize=True, quantile_transform=True, seed=459107)
print('classes', np.unique(data.y_test))

trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=BATCH_SIZE, num_workers=12, shuffle=True)
valloader = DataLoader(TorchDataset(data.X_valid, data.y_valid), batch_size=BATCH_SIZE*2, num_workers=12, shuffle=False)

root_dir = Path("./results/optuna/tabular/") / "{}/linear={}/".format(DATA_NAME, LINEAR)

deterministic(SEED)

def objective(trial):

    TREE_DEPTH = trial.suggest_int('TREE_DEPTH', 2, 6)
    REG = trial.suggest_loguniform('REG', 1e-3, 1e3)
    print(f'TREE_DEPTH={TREE_DEPTH}, REG={REG}')

    if not LINEAR:
        MLP_LAYERS = trial.suggest_int('MLP_LAYERS', 2, 7)
        DROPOUT = trial.suggest_uniform('DROPOUT', 0.0, 0.5)
        print(f'MLP_LAYERS={MLP_LAYERS}, DROPOUT={DROPOUT}')
    
    pruning = REG > 0

    if LINEAR:
        save_dir = root_dir / "depth={}/reg={}/seed={}".format(TREE_DEPTH, REG, SEED)
        model = LTBinaryClassifier(TREE_DEPTH, data.X_train.shape[1], reg=REG, linear=LINEAR)

    else:
        save_dir = root_dir / "depth={}/reg={}/mlp-layers={}/dropout={}/seed={}".format(TREE_DEPTH, REG, MLP_LAYERS, DROPOUT, SEED)
        model = LTBinaryClassifier(TREE_DEPTH, data.X_train.shape[1], reg=REG, linear=LINEAR, layers=MLP_LAYERS, dropout=DROPOUT)

    print(model.count_parameters(), "model's parameters")
    
    save_dir.mkdir(parents=True, exist_ok=True)

    # init optimizer
    optimizer = QHAdam(model.parameters(), lr=LR, nus=(0.7, 1.0), betas=(0.995, 0.998))

    # init learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    # init loss
    criterion = BCELoss(reduction="sum")

    # evaluation criterion => error rate
    eval_criterion = lambda x, y: (x.long() != y.long()).sum()

    # init train-eval monitoring 
    monitor = MonitorTree(pruning, save_dir)

    state = {
        'batch-size': BATCH_SIZE,
        'loss-function': 'BCE',
        'learning-rate': LR,
        'seed': SEED,
        'dataset': DATA_NAME,
    }

    best_val_loss = float("inf")
    best_e = -1
    no_improv = 0
    for e in range(EPOCHS):
        train_stochastic(trainloader, model, optimizer, criterion, epoch=e, monitor=monitor)

        val_loss = evaluate(valloader, model, {'ER': eval_criterion}, epoch=e, monitor=monitor)
        
        no_improv += 1
        if val_loss['ER'] < best_val_loss:
            best_val_loss = val_loss['ER']
            best_e = e
            no_improv = 0
            # save_model(model, optimizer, state, save_dir)
        
        # reduce learning rate if needed
        lr_scheduler.step(val_loss['ER'])
        monitor.write(model, e, train={"lr": optimizer.param_groups[0]['lr']})

        trial.report(val_loss['ER'], e)
        # Handle pruning based on the intermediate value.
        if trial.should_prune() or np.isnan(val_loss['ER']):
            monitor.close()
            raise optuna.TrialPruned()

        if no_improv == 10:
            break

    print("Best validation ER:", best_val_loss)
    monitor.close()

    return best_val_loss

if __name__ == "__main__":

    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(study_name=DATA_NAME, pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)

    print(study.best_params, study.best_value)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    print(df)
    df.to_csv(root_dir / 'trials.csv')
