
import time
import torch
from module.method.measure import Measure

class LeaveOneOutWithAttack(Measure):
    name = 'LeaveOneOut'

    def __init__(self, loader, model, cache, value_functions, attacker=None, attacker_id=None):
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        self.attacker = attacker
        self.attacker_id = attacker_id

    def get_contributions(self, **kwargs):
        device = self.model.device
        t0 = time.time()

        for idx_val in range(len(self.value_functions)):
            baseline_value = self.evaluate_subset(set(range(self.num_parts)), self.value_functions[idx_val])

            for i in range(self.num_parts):
                subset = set(range(self.num_parts))
                subset.discard(i)
                removed_value = (
                    self.evaluate_subset_with_attack(subset, self.value_functions[idx_val])
                    if self.attacker and self.attacker_id in subset
                    else self.evaluate_subset(subset, self.value_functions[idx_val])
                )
                self.contributions[idx_val][i] = baseline_value - removed_value

        if "cuda" in str(device):
            torch.cuda.synchronize()

        t_cal = time.time() - t0
        return self.contributions.tolist(), t_cal

    def evaluate_subset_with_attack(self, parts: set, value_function: str):
        import copy
        assert isinstance(value_function, str)
        assert isinstance(parts, set)

        function_index = self.value_functions.index(value_function)
        parts = set(sorted(parts))

        if self.cache.get(str(parts)) is not None:
            val_list = self.cache[str(parts)]
        else:
            X_train_parts_coal = [copy.deepcopy(self.X_train_parts[i]) for i in parts]
            y_train_parts_coal = [copy.deepcopy(self.y_train_parts[i]) for i in parts]

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

            shard_sizes = torch.tensor([len(self.X_train_parts[i]) for i in parts], dtype=torch.float)
            weights = shard_sizes / shard_sizes.sum()

            aggregated_gradient = [torch.zeros_like(param).to(self.model.device) for param in self.model.parameters()]
            for grad, weight in zip(gradients, weights):
                self.add_gradient_updates(aggregated_gradient, grad, weight=weight)

            self.add_update_to_model(self.model, aggregated_gradient)

            val = self.model.score(self.X_test, self.y_test, self.value_functions)[function_index]
            self.cache[str(parts)] = [val]

        return self.cache[str(parts)][function_index]