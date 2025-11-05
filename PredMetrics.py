# Importing Libraries
import copy
import math
import torch
from abc import ABC, abstractmethod
from utils.losses import ModifiedBCELoss
from typing import Literal, Union, Callable
from sklearn.model_selection import train_test_split


# BaseClass Definition
class BasePredictabilityMetric(ABC):

    def __init__(
        self,
        model_acc: Union[float, dict],
        eval_metric: Union[Callable, str] = "mse",
        test_size=0.2,
    ):
        self.model_acc = model_acc
        self.test_size = test_size
        self.loss_functions = {
            "mse": torch.nn.MSELoss(),
            "cross-entropy": torch.nn.CrossEntropyLoss(),
            "bce": torch.nn.BCELoss(),
        }
        self.eval_functions = {
            "accuracy": lambda y_pred, y: (y_pred == y).float().mean(),
            "mse": lambda y_pred, y: ((y_pred - y) ** 2).float().mean(),
            "bce": ModifiedBCELoss,
        }

        self.initEvalMetric(eval_metric)
        self.defineModel()

    def train(
        self,
        x: torch.tensor,
        y: torch.tensor,
        attacker_mode: str,
    ) -> torch.tensor:
        self.defineModel()
        model = getattr(self, "attacker_" + attacker_mode)
        criterion = self.loss_functions[self.train_params["loss_function"]]
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.train_params["learning_rate"]
        )
        batches = math.ceil(len(x) / self.train_params["batch_size"])

        print(f"Training Activated for Mode: {attacker_mode}")

        # Training loop
        for epoch in range(1, self.train_params["epochs"] + 1):
            perm = torch.randperm(x.shape[0])
            x = x[perm]
            y = y[perm]
            start = 0
            running_loss = 0.0
            for batch_num in range(batches):
                x_batch = x[start : (start + self.train_params["batch_size"])]
                y_batch = y[start : (start + self.train_params["batch_size"])]

                optimizer.zero_grad()
                # Forward pass
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                start += self.train_params["batch_size"]
                running_loss += loss.item()

            avg_loss = running_loss / batches
            if epoch % 10 == 0:
                print(f"\rCurrent Epoch {epoch}: Loss = {avg_loss}", end="")

        print("\nModel training completed")

    def permuteData(self, data: torch.tensor, mode: str = "AtoT") -> torch.tensor:
        """
        Currently assumes ground truth data to be binary values in a pytorch tensor.
        Should work for any NxM type array.

        Parameters
        ----------
        data : torch.tensor
            Original ground truth data.

        Returns
        -------
        new_data : torch.tensor
            Randomly pertubed data for quality equalization.
        """
        if type(self.model_acc) in [float, int, torch.Tensor]:
            if self.model_acc > 1:
                self.model_acc = self.model_acc / 100
            curr_model_acc = self.model_acc
        else:
            curr_model_acc = self.model_acc[mode]
        num_observations = data.shape[0]
        rand_vect = torch.zeros((num_observations, 1))
        rand_vect[: int(curr_model_acc * num_observations)] = 1
        rand_vect = rand_vect[torch.randperm(num_observations)]
        new_data = rand_vect * (data) + (1 - rand_vect) * (1 - data)
        return new_data

    def split(
        self,
        feat: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
        test_size: Union[float, None] = None,
    ) -> tuple[torch.tensor]:
        if test_size == None:
            test_size = self.test_size
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = (
            train_test_split(feat, data, pred, test_size=test_size)
        )
        return feat_train, feat_test, data_train, data_test, pred_train, pred_test

    def calcLeak(
        self,
        feat_train: torch.tensor,
        data_train: torch.tensor,
        pred_train: torch.tensor,
        feat_test: torch.tensor,
        data_test: torch.tensor,
        pred_test: torch.tensor,
        mode: Literal["AtoT", "TtoA"],
    ) -> torch.tensor:
        """
        Parameters
        ----------
        feat : torch.tensor
            Protected Attribute.
        data : torch.tensor
            Ground truth data.
        pred : torch.tensor
            Predicted Values.
        mode : Literal["AtoT","TtoA"]
            Sets Direction of calculation.

        Returns
        -------
        leakage : torch.tensor
            Evaluated Leakage.

        """
        if mode:
            mode = "_" + mode
        else:
            mode = ""
        pert_data_train = self.permuteData(data_train, mode)
        pert_data_test = self.permuteData(data_test, mode)
        self.train(feat_train, pert_data_train, "D" + mode)
        lambda_d = self.calcLambda(
            getattr(self, "attacker_D" + mode), feat_test, pert_data_test
        )
        self.train(feat_train, pred_train, "M" + mode)
        lambda_m = self.calcLambda(
            getattr(self, "attacker_M" + mode), feat_test, pred_test
        )
        print(f"{lambda_d=},\n{lambda_m=}")
        if self.normalized:
            leakage = (lambda_m - lambda_d) / (lambda_m + lambda_d)
        else:
            leakage = lambda_m - lambda_d
        return leakage

    def initEvalMetric(self, metric: Union[Callable, str]) -> None:
        if callable(metric):
            self.eval_metric = metric
        elif type(metric) == str:
            if metric in self.eval_functions.keys():
                self.eval_metric = self.eval_functions[metric]
            else:
                raise ValueError("Metric Option given is unavailable.")
        else:
            raise ValueError("Invalid Metric Given.")

    @abstractmethod
    def calcLambda(
        self, model: torch.nn.Module, x: torch.tensor, y: torch.tensor, **kwargs
    ) -> torch.tensor:
        pass

    @abstractmethod
    def defineModel(self):
        pass


# Leakage
class Leakage(BasePredictabilityMetric):

    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        model_acc: float,
        eval_metric: Union[Callable, str] = "mse",
        threshold=True,
        normalized=True,
    ) -> None:
        """
        Parameters
        ----------
        model_params : dict
            Dictionary of the following forms-
            {"attacker" : model}
        train_params : dict
            {
                "learning_rate": The learning rate hyperparameter,
                "loss_function": The loss function to be used.
                        Existing options: ["mse", "cross-entropy"],
                "epochs": Number of training epochs to be set,
                "batch_size: Number of batches per epoch
            }
        model_acc : float
            The accuracy of the model being tested for quality equalization.
            For bidirectional case, send dict of the form {'AtoT': acc_AtoT, 'TtoA': acc_TtoA}
        eval_metric : Union[Callable,str], optional
            Either a Callable of the form eval_metric(y_pred, y)
            or a string to utilize exiting methods.
            Existing options include ["accuracy"]
            The default is "mse".

        Returns
        -------
        None
            Initializes the class.

        """
        self.model_params = model_params
        self.train_params = train_params
        self.threshold = threshold
        self.model_acc = model_acc
        self.normalized = False
        super().__init__(self.model_acc, eval_metric)

    def defineModel(self) -> None:
        self.attacker_D = self.model_params["attacker"]
        self.attacker_M = copy.deepcopy(self.attacker_D)

    def getAmortizedLeakage(
        self,
        feat_train: torch.tensor,
        data_train: torch.tensor,
        pred_train: torch.tensor,
        num_trials: int = 10,
        method: str = "mean",
        feat_test: torch.tensor = None,
        data_test: torch.tensor = None,
        pred_test: torch.tensor = None,
    ) -> tuple[torch.tensor, torch.tensor]:
        if feat_test == None:
            feat_train, feat_test, data_train, data_test, pred_train, pred_test = (
                self.split(feat_train, data_train, pred_train, test_size=0.2)
            )
        vals = torch.zeros(num_trials)
        for i in range(num_trials):
            print(f"Working on Trial: {i}")
            vals[i] = self.calcLeak(
                feat_train,
                data_train,
                pred_train,
                feat_test,
                data_test,
                pred_test,
                mode=None,
            )
            print(f"Trial {i} val: {vals[i]}")
        if method == "mean":
            return torch.mean(vals), torch.std(vals)
        elif method == "median":
            return torch.median(vals), torch.std(vals)
        else:
            raise ValueError("Invalid Method given for Amortization.")

    def calcLambda(
        self, model: torch.nn.Module, x: torch.tensor, y: torch.tensor
    ) -> torch.tensor:
        y_pred = model(x)
        if self.threshold:
            y_pred = y_pred > 0.5
        return self.eval_metric(y_pred, y)


# DPA
class DPA(BasePredictabilityMetric):

    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        model_acc: Union[float, dict],
        eval_metric: Union[Callable, str] = "mse",
        threshold=True,
        normalized=True,
    ) -> None:
        """
        Parameters
        ----------
        model_params : dict
            Dictionary of the following forms-
            {"attacker_AtoT" : model_AT, "attacker_TtoA" : model_TA}
        train_params : dict
            {
                "AtoT":
                    {
                        "learning_rate": The learning rate hyperparameter,
                        "loss_function": The loss function to be used.
                                Existing options: ["mse", "cross-entropy"],
                        "epochs": Number of training epochs to be set,
                        "batch_size: Number of batches per epoch
                    },
                "TtoA": {same format as AtoT}
            }
        model_acc : Union[float, dict]
            The accuracy of the model being tested for quality equalization.
            For bidirectional case, send dict of the form {'AtoT': acc_AtoT, 'TtoA': acc_TtoA}
        eval_metric : Union[Callable,str], optional
            Either a Callable of the form eval_metric(y_pred, y)
            or a string to utilize exiting methods.
            Existing options include ["accuracy"]
            The default is "mse".

        Returns
        -------
        None
            Initializes the class.

        """
        self.model_params = model_params
        self.train_params = train_params
        self.threshold = threshold
        self.model_acc = model_acc
        self.normalized = normalized
        super().__init__(self.model_acc, eval_metric)

    def defineModel(self) -> None:
        if type(self.model_params.get("attacker_AtoT", None)) == None:
            raise Exception("attacker_AtoT Model Missing!")
        if type(self.model_params.get("attacker_TtoA", None)) == None:
            raise Exception("attacker_TtoA Model Missing!")
        self.attacker_D_AtoT = self.model_params["attacker_AtoT"]
        self.attacker_M_AtoT = copy.deepcopy(self.attacker_D_AtoT)
        self.attacker_D_TtoA = self.model_params["attacker_TtoA"]
        self.attacker_M_TtoA = copy.deepcopy(self.attacker_D_TtoA)

    def getAmortizedLeakage(
        self,
        feat_train: torch.tensor,
        data_train: torch.tensor,
        pred_train: torch.tensor,
        mode: Literal["AtoT", "TtoA"],
        num_trials: int = 10,
        method: str = "mean",
        feat_test: torch.tensor = None,
        data_test: torch.tensor = None,
        pred_test: torch.tensor = None,
    ) -> tuple[torch.tensor, torch.tensor]:
        if feat_test == None:
            feat_train, feat_test, data_train, data_test, pred_train, pred_test = (
                self.split(feat_train, data_train, pred_train, test_size=0.2)
            )
        vals = torch.zeros(num_trials)
        for i in range(num_trials):
            print(f"Working on Trial: {i}")
            vals[i] = self.calcLeak(
                feat_train,
                data_train,
                pred_train,
                feat_test,
                data_test,
                pred_test,
                mode,
            )
            print(f"Trial {i} val: {vals[i]}")
        if method == "mean":
            return torch.mean(vals), torch.std(vals)
        elif method == "median":
            return torch.median(vals), torch.std(vals)
        else:
            raise ValueError("Invalid Method given for Amortization.")

    def calcLambda(
        self, model: torch.nn.Module, x: torch.tensor, y: torch.tensor
    ) -> torch.tensor:
        y_pred = model(x)
        if self.threshold:
            y_pred = y_pred > 0.5
        return self.eval_metric(y_pred, y)

    def calcBidirectional(
        self,
        A: torch.tensor,
        T: torch.tensor,
        A_pred: torch.tensor,
        T_pred: torch.tensor,
        num_trials: int = 10,
        method: str = "mean",
    ) -> tuple[tuple[torch.tensor, torch.tensor], tuple[torch.tensor, torch.tensor]]:
        AtoT_vals = self.getAmortizedLeakage(A, T, T_pred, num_trials, method)
        TtoA_vals = self.getAmortizedLeakage(T, A, A_pred, num_trials, method)
        return (AtoT_vals, TtoA_vals)


# LIC
class LIC(BasePredictabilityMetric):

    def __init__(self):
        pass


if __name__ == "__main__":
    # Test case
    from attackerModels.ANN import simpleDenseModel

    # Data Initialization
    from utils.datacreator import dataCreator

    P, D, D2, M1, M2 = dataCreator(16384, 0.2, False, 0.05)
    P = torch.tensor(P, dtype=torch.float).reshape(-1, 1)
    D = torch.tensor(D, dtype=torch.float).reshape(-1, 1)
    D2 = torch.tensor(D2, dtype=torch.float).reshape(-1, 1)
    M1 = torch.tensor(M1, dtype=torch.float).reshape(-1, 1)
    M2 = torch.tensor(M2, dtype=torch.float).reshape(-1, 1)

    # Calculating Params
    model_1_acc = torch.sum(D == M1) / D.shape[0]
    model_2_acc = torch.sum(D == M2) / D.shape[0]

    # Attacker Model Initialization
    attackerModel = simpleDenseModel(
        1, 1, 1, numFirst=1, activations=["sigmoid", "sigmoid", "sigmoid"]
    )

    #Leakage Parameter Initialization
    leakage_obj = Leakage(
        {"attacker": attackerModel},
        {
            "learning_rate": 0.05,
            "loss_function": "bce",
            "epochs": 100,
            "batch_size": 64,
        },
        model_1_acc,
        "accuracy",
        threshold=True,
    )
    leakage_1 = leakage_obj.getAmortizedLeakage(P, D, M1)
    print(f"Leakage for case 1: {leakage_1}")
    print("______________________________________")
    print("______________________________________")
    leakage_2 = leakage_obj.getAmortizedLeakage(P, D, M2)
    print(f"Leakage for case 2: {leakage_2}")
    print("______________________________________")
    print("______________________________________")
    leakage_3 = leakage_obj.getAmortizedLeakage(P, D2, M1)
    print(f"Leakage for case 3: {leakage_3}")
    print("______________________________________")
    print("______________________________________")
    leakage_4 = leakage_obj.getAmortizedLeakage(P, D2, M2)
    print(f"Leakage for case 4: {leakage_4}")
    print("______________________________________")
    print("______________________________________")


    
    # Parameter Initialization
    dpa_obj = DPA(
        {"attacker_AtoT": attackerModel, "attacker_TtoA": attackerModel},
        {
            "learning_rate": 0.05,
            "loss_function": "bce",
            "epochs": 100,
            "batch_size": 64,
        },
        model_1_acc,
        "accuracy",
        threshold=True,
    )

    dpa_1 = dpa_obj.getAmortizedLeakage(P, D, M1, "AtoT")
    print(f"DPA for case 1: {dpa_1}")
    print("______________________________________")
    print("______________________________________")
    dpa_2 = dpa_obj.getAmortizedLeakage(P, D, M2, "AtoT")
    print(f"DPA for case 2: {dpa_2}")
    print("______________________________________")
    print("______________________________________")
    dpa_3 = dpa_obj.getAmortizedLeakage(P, D2, M1, "AtoT")
    print(f"DPA for case 3: {dpa_3}")
    print("______________________________________")
    print("______________________________________")
    dpa_4 = dpa_obj.getAmortizedLeakage(P, D2, M2, "AtoT")
    print(f"DPA for case 4: {dpa_4}")
    print("______________________________________")
    print("______________________________________")
