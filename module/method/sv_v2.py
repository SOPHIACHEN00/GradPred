import time
from module.method.measure import Measure
from itertools import combinations as cbs
from math import comb
import torch
import copy


class ShapleyValue(Measure):
    name = 'ShapleyValue'

    def __init__(self, loader, model, cache, value_functions, attacker=None, attacker_id=None):
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        self.cache[str(set())] = [0 for _ in
                                  range(len(value_functions))]  # cache for computed v(S) results where S = set(...)
        self.attacker = attacker
        self.attacker_id = attacker_id

    def phi(self, i, value_function):
        accu = 0.
        all_set = set(range(self.num_parts))
        all_set.remove(i)
        for S_size in range(len(all_set) + 1):
            weight = 1. / self.num_parts / comb(self.num_parts - 1, S_size)
            for cb in cbs(all_set, S_size):
                set_with_i = set(cb)
                set_with_i.add(i)
                set_without_i = set(cb)
                accu += (self.evaluate_subset(set_with_i, value_function) - self.evaluate_subset(set_without_i,
                                                                                                 value_function)) * weight
        return accu

    def get_contributions(self, **kwargs):
        device = self.model.device
        t0 = time.time()

        for idx_val in range(len(self.value_functions)):
            for i in range(self.num_parts):
                self.contributions[idx_val][i] = self.phi_with_attack(i, self.value_functions[idx_val])

        if "cuda" in str(device):
            torch.cuda.synchronize()

        t_cal = time.time() - t0

        return self.contributions.tolist(), t_cal

    def phi_with_attack(self, i, value_function):
        accu = 0.
        all_set = set(range(self.num_parts))
        all_set.remove(i)
        for S_size in range(len(all_set) + 1):
            weight = 1. / self.num_parts / comb(self.num_parts - 1, S_size)
            for cb in cbs(all_set, S_size):
                set_with_i = set(cb)
                set_with_i.add(i)
                set_without_i = set(cb)
                
                if self.attacker and self.attacker_id in set_with_i:
                    val_with = self.evaluate_subset_with_attack(set_with_i, value_function)
                else:
                    val_with = self.evaluate_subset(set_with_i, value_function)

                if self.attacker and self.attacker_id in set_without_i:
                    val_without = self.evaluate_subset_with_attack(set_without_i, value_function)
                else:
                    val_without = self.evaluate_subset(set_without_i, value_function)

                accu += (val_with - val_without) * weight
        return accu

    def evaluate_subset_with_attack(self, parts: set, value_function: str):
        assert isinstance(value_function, str)
        assert isinstance(parts, set)

        function_index = self.value_functions.index(value_function)
        parts = set(sorted(parts))

        if self.cache.get(str(parts)) is not None:
            val_list = self.cache[str(parts)]
        else:
            X_train_parts_coal = [copy.deepcopy(self.X_train_parts[i]) for i in parts]
            y_train_parts_coal = [copy.deepcopy(self.y_train_parts[i]) for i in parts]

            client_models = []
            self.model.load_state_dict(self.model.initial_state_dict)

            gradients = []

            for i in parts:
                model_i = copy.deepcopy(self.model).to(self.model.device)
                backup = copy.deepcopy(model_i)

                model_i.fit(
                    X_train_parts_coal[list(parts).index(i)],
                    y_train_parts_coal[list(parts).index(i)],
                    incremental=True,
                    num_epochs=1
                )

                if i == self.attacker_id:
                    gradient = self.attacker.get_fake_gradient(
                        round_num=len(self.attacker.global_gradient_history),
                        device=self.model.device,
                        model=model_i
                    )
                else:
                    gradient = self.compute_grad_update(backup, model_i, device=self.model.device)

                gradients.append(gradient)

            # aggregate
            shard_sizes = torch.tensor([len(self.X_train_parts[i]) for i in parts], dtype=torch.float)
            weights = shard_sizes / shard_sizes.sum()

            aggregated_gradient = [torch.zeros_like(param).to(self.model.device) for param in self.model.parameters()]
            for grad, weight in zip(gradients, weights):
                self.add_gradient_updates(aggregated_gradient, grad, weight=weight)

            self.add_update_to_model(self.model, aggregated_gradient)

            val = self.model.score(self.X_test, self.y_test, self.value_functions)[function_index]
            self.cache[str(parts)] = [val]

        return self.cache[str(parts)][function_index]
