import sys
sys.path.append("./uni2ts/src")
from uni2ts.model.moirai.forecast import MoiraiForecast

from uni2ts.distribution.normal import NormalOutput
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule



import torch
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import torch.nn.functional as F



# 
# NEW! Method1: RANDOM attack
class RandomAttack:
    def __init__(self):
        self.global_gradient_history = []

    def get_fake_gradient(self, round_num, device, model):
        return [torch.randn_like(param, device=device) for param in model.parameters()]

    def record_global_gradient(self, global_gradient):
        self.global_gradient_history.append(global_gradient)



# NEW! Method2: FED AVG attack
# e.g. (k=5, start_attack_round=3:
#          round_num=3开始攻击（前3轮是随机）
#           每次都使用“过去5轮”的 global gradient 来伪造
#           不够5就随机, 第5轮使用round0-4的avg，第6轮使用round1-5的avg...
class GradientReplayAttack:
    def __init__(self, k, start_attack_round):
        self.k = k 
        self.start_attack_round = start_attack_round  # start-attacking
        self.global_gradient_history = []  #global_gradient_history: List[List[Tensor]]  # shape: [round][param_i]


    def get_fake_gradient(self, round_num, device, model):
        if round_num < self.start_attack_round or len(self.global_gradient_history) < self.k:
            # first M rounds：randomly generare gradient
            return [torch.randn_like(param, device=device) for param in model.parameters()]
        else: # calculate avg value
            avg_gradient = [
                sum(t[i] for t in self.global_gradient_history[-self.k:]) / self.k
                for i in range(len(self.global_gradient_history[0]))
            ]
            return avg_gradient

    def record_global_gradient(self, global_gradient):
        self.global_gradient_history.append(global_gradient)



# NEW! Method3: ARIMA attack
#自动判断何时切换阶段? 历史 global gradient 方差趋于稳定后开始使用 ARIMA (this one)
class ARIMAAttack:
    def __init__(self, k, random_round, var_window=3, var_threshold=5e-3): #1e-4 3e-4
        self.k = k
        self.random_round = random_round
        self.var_window = var_window         
        self.var_threshold = var_threshold   
        self.global_gradient_history = []
        self.arima_started = False           
        self.arima_start_round = None        
        
        self.offline_model_trained = False 
        self.offline_model = {}


    def record_global_gradient(self, global_gradient):
        self.global_gradient_history.append(global_gradient)

    def train_offline_arima_model(self):
        print("[ARIMA] Starting offline ARIMA model training...")
        self.offline_model = {}  

        for param_index in range(len(self.global_gradient_history[0])):
            param_series_list = []
            param = self.global_gradient_history[0][param_index]
            for i in range(param.numel()):
                series = np.array([
                    self.global_gradient_history[round][param_index].view(-1)[i].item()
                    for round in range(len(self.global_gradient_history))
                ])
                try:
                    model = auto_arima(
                        series,
                        start_p=0, start_q=0, max_p=2, max_q=2,
                        d=None, max_d=1,
                        seasonal=False,
                        stationary=False,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,
                        maxiter=20
                    )
                    param_series_list.append(model)
                except Exception as e:
                    print(f"[ARIMA] param {param_index} element {i} auto_arima failed, fallback to None: {e}")
                    param_series_list.append(None)
            self.offline_model[param_index] = param_series_list 

        print("[ARIMA] Offline model training complete.")



    def check_arima_ready(self):
        # If the rate of change of the global gradient variance of the consecutive var_window rounds is lower than the threshold, it indicates that it tends to be stable
        if len(self.global_gradient_history) < self.k + self.var_window:
            return False

        var_list = []
        for param_idx in range(len(self.global_gradient_history[0])):
            # Take the data sequence within the sliding window
            series = [g[param_idx].view(-1).cpu().numpy()
                        for g in self.global_gradient_history[-(self.k + self.var_window):]]
            var_series = [np.var(np.stack(series[i:i + self.k])) for i in range(self.var_window)]
            var_list.append(var_series)

        # calculate abs diff
        var_diff = np.abs(np.diff(np.mean(var_list, axis=0)))
        print(f"[ARIMA] var diff: {var_diff}") 
        return np.all(var_diff < self.var_threshold)


    def get_fake_gradient(self, round_num, device, model):
        if not self.arima_started and self.check_arima_ready():
            self.arima_started = True
            self.arima_start_round = round_num
            self.train_offline_arima_model()
            self.offline_model_trained = True
            print(f"[ARIMA] round={round_num} SWITCHED TO AUTO-ARIMA")

        if self.arima_started and self.offline_model_trained:
            predicted_gradient = []
            for param_index, param in enumerate(model.parameters()):
                predicted_param = torch.zeros_like(param, device=device)
                for i in range(param.numel()):
                    try:
                        model_fit = self.offline_model[param_index][i] 
                    except (IndexError, KeyError):
                        model_fit = None

                    if model_fit is not None:
                        forecast = model_fit.predict(n_periods=1)
                        predicted_param.view(-1)[i] = torch.tensor(forecast[0], device=device)
                    else:
                        predicted_param.view(-1)[i] = torch.tensor(0.0, device=device)
                predicted_gradient.append(predicted_param)
            return predicted_gradient

        if len(self.global_gradient_history) < self.k:
            print(f"[ARIMA] round={round_num} USING RANDOM (not enough history)")
            return [torch.randn_like(param) for param in model.parameters()]
        elif round_num < self.random_round:
            print(f"[ARIMA] round={round_num} USING RANDOM")
            return [torch.randn_like(param) for param in model.parameters()]
        else:
            print(f"[ARIMA] round={round_num} USING AVG")
            return [
                sum(t[i] for t in self.global_gradient_history[-self.k:]) / self.k
                for i in range(len(self.global_gradient_history[0]))
            ]




# # NEW! Method4: Moirai attack
class MoiraiAttack:
    def __init__(self, k, random_round, history_length=5, model_type="moirai", size="small", patch_size="auto"):
        self.history_length = history_length
        self.k = k
        self.model_type = model_type
        self.size = size
        self.patch_size = patch_size
        self.random_round = random_round
        self.global_gradient_history = [] 

        # initialization
        if self.model_type == "moirai":
            self.model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{self.size}"),
                prediction_length=1,  #Predict the next gradient
                context_length=self.history_length,
                patch_size=self.patch_size,
                num_samples=1,  # Generate only one sample/the next gradient
                target_dim=1,
                feat_dynamic_real_dim=0, 
                past_feat_dynamic_real_dim=0,
            )
        else:
            self.model = MoiraiMoEForecast(
                module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{self.size}"),
                prediction_length=1,
                context_length=self.history_length,
                patch_size=16,
                num_samples=1,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )

    def record_global_gradient(self, global_gradient):
        self.global_gradient_history.append(global_gradient)

    def get_fake_gradient(self, round_num, device, model):  
        if len(self.global_gradient_history) < self.history_length:
            print(f"[Moirai] round={round_num} USING RANDOM (not enough history)")
            return [torch.randn_like(param, device=device) for param in model.parameters()]

        if round_num < self.random_round:
            print(f"[Moirai] round={round_num} USING AVG")
            avg_gradient = [
                sum(t[i] for t in self.global_gradient_history[-self.k:]) / self.k
                for i in range(len(self.global_gradient_history[0]))
            ]
            return avg_gradient

        print(f"[Moirai] round={round_num} USING MOIRAI PREDICTION")

        gradient_series = []
        for round_grad in self.global_gradient_history[-self.history_length:]:
            flat_grad = torch.cat([g.view(-1) for g in round_grad], dim=0)
            gradient_series.append(flat_grad.cpu().numpy())

        gradient_series = np.stack(gradient_series)


        input_data = [{
            "start": pd.Period("2000-01-01", freq="D"),
            "target": gradient_series.flatten(),
        }]

        predictor = self.model.create_predictor(batch_size=1)
        prediction_iterator = predictor.predict(input_data)
        forecast = next(prediction_iterator)
        # Extract the predicted values Split the predicted_value back according to the shape of each param
        predicted_value = forecast.samples[0,0]
        fake_gradient = []
        start_idx = 0
        for param in model.parameters():
            numel = param.numel()
            param_data = predicted_value * torch.ones_like(param, device=device)
            fake_gradient.append(param_data)
        
        return fake_gradient
