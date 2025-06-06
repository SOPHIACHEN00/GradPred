import typing
setattr(__builtins__, 'Union', typing.Union)
setattr(__builtins__, 'Tuple', typing.Tuple)

import sys
sys.path.append("./uni2ts/src")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import torch
import os
import csv
import numpy as np
from sklearn.metrics import accuracy_score
from script.datapre import *
from module.model.model import return_model
# from module.method.sv_v2 import ShapleyValue
# from module.method.loo_v2 import LeaveOneOutWithAttack

from module.method.attack_v3 import GradientReplayAttack, ARIMAAttack, RandomAttack, MoiraiAttack
import copy
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--attacker_id', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.4)
parser.add_argument('--dataset', type=str, default="tictactoe")
parser.add_argument('--attack_method', type=str, default="random")
parser.add_argument('--client_num', type=int, default=6)
parser.add_argument('--contribution_method', type=str, default="shapley")
parser.add_argument('--use_attack', action='store_true', help='Enable attack or not')
parser.add_argument('--trial_id', type=int, default=0, help='Repetition ID')


args = parser.parse_args()

attacker_id = args.attacker_id
alpha = args.alpha
dataset = args.dataset.lower()
attack_method = args.attack_method.lower()
num_parts = args.client_num
use_attack = args.use_attack
trial_id = args.trial_id

uniform_thresholds = {
    "tictactoe": 20000,
    "adult": 20000000,
    "dota2": 200000000
}
distribution = "uniform" if alpha == uniform_thresholds[dataset] else "quantity skew"


model_name_map = {
    "tictactoe": "TicTacToeLR",
    "adult": "AdultLR",
    "dota2": "Dota2LR"
}
model_name = model_name_map[dataset]



# ======== Configuration ========
seed = 42
num_epoch = 50
num_local_epochs = 1

lr = 0.008
hidden_layer_size = 16
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

attack_time_log = []


# ======== Nohup Logs ========
print(f"=== Running {dataset}, attack={attack_method}, N={num_parts}, alpha={alpha}, "
      f"use_attack={use_attack}, attacker_id={attacker_id}, trial_id={trial_id} ===", flush=True)



# ======== Attack Configuration ========
attack_k = 4
start_attack_round = 10
random_round_arima = 10
random_round_moirai = 10


# ======== Load Data & Model ========
loader = get_data(seed=seed, dataset=dataset, distribution=distribution,
                  alpha=alpha, num_parts=num_parts)

model = return_model(model_name, seed=seed, num_epoch=num_epoch, lr=lr,
                     batch_size=batch_size, hidden_layer_size=hidden_layer_size,
                     device=device, dataset=dataset)

model.init_args = {
    "seed": seed,
    "num_epoch": num_epoch,
    "lr": lr,
    "device": device,
    "hidden_layer_size": hidden_layer_size,
    "batch_size": batch_size
}


def prepare_loader_without_attacker(loader, attacker_id):
    ids = [i for i in range(len(loader.X_train_parts)) if i != attacker_id]
    new_loader = copy.deepcopy(loader)
    new_loader.X_train_parts = [loader.X_train_parts[i] for i in ids]
    new_loader.y_train_parts = [loader.y_train_parts[i] for i in ids]
    return new_loader, ids

def run_federated_learning(loader, model, attacker_id, attacker, use_attack,
                           num_epoch, num_local_epochs, device, record_gradients=False):
    num_parts = len(loader.X_train_parts)
    client_models = [copy.deepcopy(model) for _ in range(num_parts)]
    shard_sizes = torch.tensor([len(loader.X_train_parts[i]) for i in range(num_parts)], dtype=torch.float)
    weights = shard_sizes / shard_sizes.sum()
    attack_time_log = []
    global_accuracies = []
    
    for round_num in range(num_epoch):
        gradients = []
        for i in range(num_parts):
            X_i, y_i = loader.X_train_parts[i], loader.y_train_parts[i]
            model_i = client_models[i].to(device)
            backup = copy.deepcopy(model_i)
            model_i.fit(X_i, y_i, incremental=True, num_epochs=num_local_epochs)
            if use_attack and i == attacker_id:
                attack_start_time = time.time()
                gradient = attacker.get_fake_gradient(round_num, device, model_i)
                attack_end_time = time.time()
                attack_time_log.append(attack_end_time - attack_start_time)
            else:
                gradient = [(new.data - old.data) for new, old in zip(model_i.parameters(), backup.parameters())]
            gradients.append(gradient)
        # Aggregation
        aggregated_gradient = [torch.zeros_like(param).to(device) for param in model.parameters()]
        for grad, weight in zip(gradients, weights):
            for ag, g in zip(aggregated_gradient, grad):
                ag += g * weight
        if use_attack and record_gradients:
            attacker.record_global_gradient(aggregated_gradient)
        for param, update in zip(model.parameters(), aggregated_gradient):
            param.data += update.data
        for cm in client_models:
            cm.load_state_dict(model.state_dict())
        # Evaluation
        y_pred = model.predict(loader.X_test)
        acc = accuracy_score(loader.y_test, y_pred)
        global_accuracies.append(acc)
        print(f"Round {round_num}: Global Accuracy = {acc:.4f}")
    return model, attack_time_log, global_accuracies


# ==== Load ContributionCalculator ====
if args.contribution_method == "shapley":
    from module.method.sv_v2 import ShapleyValue as ContributionCalculator
elif args.contribution_method == "loo":
    from module.method.loo_v2 import LeaveOneOutWithAttack as ContributionCalculator
else:
    raise ValueError("Invalid contribution method")


if use_attack:
    # Create attacker
    if attack_method == "random":
        attacker = RandomAttack()
    elif attack_method == "fedavg":
        attacker = GradientReplayAttack(k=4, start_attack_round=10)
    elif attack_method == "arima":
        attacker = ARIMAAttack(k=4, random_round=10)
    elif attack_method == "moirai":
        attacker = MoiraiAttack(k=4, random_round=10)

    # ==== Run FL with attacker ====
    model, attack_time_log_with_attacker, global_accuracies_log_with_attacker = run_federated_learning(
        loader, model, attacker_id, attacker, use_attack=True,
        num_epoch=num_epoch, num_local_epochs=num_local_epochs,
        device=device, record_gradients=True
    )

    # ==== Run FL without attacker ====
    subset_loader, subset_ids = prepare_loader_without_attacker(loader, attacker_id)
    model_no_attacker = return_model(model_name, seed=seed, num_epoch=num_epoch, lr=lr,
                                    batch_size=batch_size, hidden_layer_size=hidden_layer_size,
                                    device=device, dataset=dataset)
    model_no_attacker.init_args = model.init_args
    model_no_attacker, attack_time_log_without_attacker, global_accuracies_log_without_attacker = run_federated_learning(
        subset_loader, model_no_attacker, attacker_id=None, attacker=None,
        use_attack=False, num_epoch=num_epoch, num_local_epochs=num_local_epochs, device=device
    )


    contrib_with = ContributionCalculator(loader, model, dict(), ["accuracy"], attacker, attacker_id)
    contribs_w, _ = contrib_with.get_contributions()

    contrib_without = ContributionCalculator(subset_loader, model_no_attacker, dict(), ["accuracy"])
    contribs_wo, _ = contrib_without.get_contributions()
else:
    # FL without attack
    model, attack_time_log_with_attacker, global_accuracies_log_with_attacker = run_federated_learning(
        loader, model, attacker_id=None, attacker=None, use_attack=False,
        num_epoch=num_epoch, num_local_epochs=num_local_epochs,
        device=device, record_gradients=True
    )

    contrib_without = ContributionCalculator(loader, model, dict(), ["accuracy"])
    contribs_wo, _ = contrib_without.get_contributions()

# ==== Save logs ====
os.makedirs("Results/Contribution_sv", exist_ok=True)
os.makedirs("Results/Time_sv", exist_ok=True)
os.makedirs("Results/Accuracy_sv", exist_ok=True)

with open(f"Results/Contribution_sv/combined_contributions_trial{trial_id}.csv", "a", newline="") as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(["Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID", "TrialID", "ClientID", "UseAttack", "WithAttacker", "WithoutAttacker"])
    for i in range(num_parts):
        if use_attack:
            w_val = contribs_w[0][i]
            if i == attacker_id:
                wo_val = "N/A"
            else:
                j = i if i < attacker_id else i - 1
                wo_val = contribs_wo[0][j]
        else:
            w_val = "N/A"
            wo_val = contribs_wo[0][i]

        writer.writerow([dataset, attack_method, alpha, num_parts, attacker_id, trial_id, i, use_attack, w_val, wo_val])

with open(f"Results/Time_sv/attacking_time_log_trial{trial_id}.csv", "a", newline="") as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(["Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID", "TrialID", "UseAttack", "WithAttackerSec", "WithoutAttackerSec"])
    if use_attack:
        writer.writerow([
            dataset, attack_method, alpha, num_parts, attacker_id, trial_id,
            use_attack, sum(attack_time_log_with_attacker), sum(attack_time_log_without_attacker)
        ])
    else:
        writer.writerow([
            dataset, attack_method, alpha, num_parts, attacker_id, trial_id,
            use_attack, "N/A", "N/A"
        ])


with open(f"Results/Accuracy_sv/global_accuracy_log_trial{trial_id}.csv", "a", newline="") as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow([
            "Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID", "TrialID", "UseAttack", "Round", "WithAttackerAccuracy", "WithoutAttackerAccuracy"
        ])

    for i in range(num_epoch):
        if use_attack:
            acc_with = global_accuracies_log_with_attacker[i] if i < len(global_accuracies_log_with_attacker) else ""
            acc_without = global_accuracies_log_without_attacker[i] if i < len(global_accuracies_log_without_attacker) else ""
        else:
            acc_with = "N/A"
            acc_without = global_accuracies_log_with_attacker[i] if i < len(global_accuracies_log_with_attacker) else ""

        writer.writerow([
            dataset, attack_method, alpha, num_parts, attacker_id, trial_id, use_attack, i, acc_with, acc_without
        ])
