# Importing Libraries
import copy
import math
import torch
from abc import ABC, abstractmethod
from typing import Literal, Union, Callable, Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split


from utils.losses import ModifiedBCELoss
import utils.config as config

# ============================================================================
# BASE CLASS
# ============================================================================
class BasePredictabilityMetric(ABC):
    """
    Base class for predictability metrics.
    It provides base functionality for computing leakage with fairness evaluation.
    """

    # ============================================================================
    # INITIALIAL SETUP
    # ============================================================================

    def __init__(
        self,
        model_params: Dict[str, torch.nn.Module],
        train_params: Dict[str, Any],
        model_acc: Union[float, dict],
        eval_metric: Union[Callable, str] = config.DEFAULT_EVAL_METRIC,
        threshold: bool = True,
        normalized: bool = False,
        test_size=config.DEFAULT_TEST_SIZE,
    ):
        self.model_params = model_params
        self.train_params = train_params
        self.threshold = threshold
        self.model_acc = model_acc
        self.normalized = normalized
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

        self.initialize_eval_metrics(eval_metric)
        self.defineModel()

    # ============================================================================
    # TRAINING METHODS
    # ============================================================================
    def train(self, x: torch.tensor, y: torch.tensor, attacker_mode: str) -> torch.tensor:
        """
        Trains the attacker model for a given mode (eg. AtoT, TtoA).

        Parameters
        ----------
        x : torch.tensor
            Input data.
        y : torch.tensor
            Target data.
        attacker_mode : str
            Mode of the attacker model.

        Returns
        -------
        None
            Trains the attacker model for a given mode.
        """
        
        attacker_model, optimizer, criterion = self.train_setup(attacker_mode)

        print(f"Training Activated for Mode: {attacker_mode}")

        # Training loop
        for epoch in range(1, self.train_params["epochs"] + 1):
            x, y = self.shuffle_data(x, y)
            avg_loss = self.train_epochs(attacker_model, optimizer, criterion, x, y)

            if epoch % config.EPOCH_LOG_INTERVAL == 0:
                print(f"\rCurrent Epoch {epoch}: Loss = {avg_loss}", end="")

        print("\nModel training completed!!")



    def train_setup(
        self, attacker_mode: str
        ) -> Tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer]:
        """
        Setup model, criterion and optimizer for training.
        
        Parameters
        ----------
        attacker_mode : str
            Mode identifier for the attacker model
            
        Returns
        -------
        Tuple[Attacker Model, loss criterion, and optimizer]
            
        """
        self.defineModel()
        attacker_model = getattr(self, "attacker_" + attacker_mode)
        optimizer = torch.optim.Adam(attacker_model.parameters(), lr=self.train_params["learning_rate"])
        criterion = self.loss_functions[self.train_params["loss_function"]]

        return attacker_model, optimizer, criterion


    def train_batch_with_loss(self, 
            attacker_model: torch.nn.Module, 
            optimizer: torch.optim.Optimizer, 
            criterion: torch.nn.Module, 
            x_batch: torch.tensor, 
            y_batch: torch.tensor
            ) -> float:
        """
        Trains a batch of data with a given loss function.

        Parameters
        ----------
        attacker_model : torch.nn.Module
            The attacker model to be trained.
        optimizer : torch.optim.Optimizer
            The optimizer to be used.
        criterion : torch.nn.Module
            The loss function to be used.
        x_batch : torch.tensor
            The input data batch.
        y_batch : torch.tensor
            The target data batch.

        Returns
        -------
        loss : float
            The loss value.
        """

        optimizer.zero_grad()
        # Forward pass
        outputs = attacker_model(x_batch)
        loss = criterion(outputs, y_batch)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        return loss.item()


    def train_epochs(self, 
        attacker_model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module, 
        x: torch.tensor, 
        y: torch.tensor
        ) -> float:
        """
        Trains the attacker model for a epoch and return the average loss.

        Parameters
        ----------
        attacker_model : torch.nn.Module
            The attacker model to be trained.
        optimizer : torch.optim.Optimizer
            The optimizer to be used.
        criterion : torch.nn.Module
            The loss function to be used.
        x : torch.tensor
            The input data.
        y : torch.tensor
            The target data.

        Returns
        -------
        avg_loss : float
            The average loss.
        """
        batches = math.ceil(len(x) / self.train_params["batch_size"])
        running_loss = 0.0
        for batch_num in range(batches):
            start = batch_num * self.train_params["batch_size"]
            end = start + self.train_params["batch_size"]
            x_batch = x[start : end]
            y_batch = y[start : end]

            loss = self.train_batch_with_loss(attacker_model, optimizer, criterion, x_batch, y_batch)
            running_loss += loss

        return running_loss / batches

    # ============================================================================
    # DATA MANIPULATION METHODS
    # ============================================================================

    def shuffle_data(self, x: torch.tensor, y: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Randomly shuffles x and y data.
        """
        perm = torch.randperm(x.shape[0])
        return x[perm], y[perm]

    def permuteData(self, data: torch.tensor, mode: str = "AtoT") -> torch.tensor:
        """
        This function permutes data for quality equalization to maintain the accuracy of the model.
        Ground truth data assumed to be binary values in a pytorch tensor but intended to work for any NxM type array.

        Parameters
        ----------
        data : torch.tensor
            Original ground truth data.
        mode : str
            Mode identifier for the attacker model

        Returns
        -------
        new_data : torch.tensor
            Randomly pertubed data matching the specified model accuracy.
        """
        if type(self.model_acc) in [float, int, torch.Tensor]:
            self.model_acc = config.normalise(self.model_acc)
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
        """
        Splits the data into training and testing sets.
        """
        if test_size == None:
            test_size = self.test_size

        feat_train, feat_test, data_train, data_test, pred_train, pred_test = (
            train_test_split(feat, data, pred, test_size=test_size)
        )
        return feat_train, feat_test, data_train, data_test, pred_train, pred_test


    # ============================================================================
    # LEAKAGE CALCULATION METHODS
    # ============================================================================

    def calcLeak(
        self,
        feat_train: torch.tensor,
        data_train: torch.tensor,
        pred_train: torch.tensor,
        feat_test: torch.tensor,
        data_test: torch.tensor,
        pred_test: torch.tensor,
        mode: Optional[str] = None,
    ) -> torch.tensor:
        """
        Calculates the leakage of the attacker model for a given mode and given data splits.
        
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
            Evaluated as if normalized, returns 
            (λ_M - λ_D) / (λ_M + λ_D), otherwise returns λ_M - λ_D

        """
        mode_suffix = "_" + mode if mode else ""

        # compute data leakage
        pert_data_train = self.permuteData(data_train, mode_suffix)
        pert_data_test = self.permuteData(data_test, mode_suffix)
        self.train(feat_train, pert_data_train, "D" + mode_suffix)
        lambda_d = self.calcLambda(
            getattr(self, "attacker_D" + mode_suffix), feat_test, pert_data_test
        )
        print(f"lambda_d: {lambda_d}")

        # compute model leakage
        self.train(feat_train, pred_train, "M" + mode_suffix)
        lambda_m = self.calcLambda(
            getattr(self, "attacker_M" + mode_suffix), feat_test, pred_test
        )
        print(f"lambda_m: {lambda_m}")

        # return computed leakage
        return self.compute_leakage(lambda_d, lambda_m, self.normalized)


    def calcLambda(
        self, model: torch.nn.Module, x: torch.tensor, y: torch.tensor, **kwargs
    ) -> torch.tensor:
        """ Calculate the lambda value for a given attacker model, input data and target data."""
        y_pred = model(x)
        if self.threshold:
            y_pred = y_pred > config.DEFAULT_PREDICTION_THRESHOLD
        return self.eval_metric(y_pred, y)


    def compute_leakage(self, lambda_d: torch.tensor, lambda_m: torch.tensor, normalized: bool = False) -> torch.tensor:
        """ Compute the leakage value for a given predictability metric."""
        if normalized:
            return (lambda_m - lambda_d) / (lambda_m + lambda_d)
        else:
            return lambda_m - lambda_d


    def getAmortizedLeakage(
        self,
        feat_train: torch.tensor,
        data_train: torch.tensor,
        pred_train: torch.tensor,
        mode: Optional[str] = None,
        num_trials: int = config.DEFAULT_NUM_TRIALS,
        method: str = config.DEFAULT_AGGREGATION_METHOD,
        feat_test: torch.tensor = None,
        data_test: torch.tensor = None,
        pred_test: torch.tensor = None,
    ) -> tuple[torch.tensor, torch.tensor]:
        if feat_test == None:
            feat_train, feat_test, data_train, data_test, pred_train, pred_test = (
                self.split(feat_train, data_train, pred_train, test_size=config.DEFAULT_TEST_SIZE)
            )
       
        vals = torch.zeros(num_trials)
        for i in range(num_trials):
            print("-"*50)
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
            print(f"Leakage for Trial {i}: {vals[i]}")
        if method == "mean":
            return self._getFormattedLeakage(torch.mean(vals), torch.std(vals))
        elif method == "median":
            return self._getFormattedLeakage(torch.median(vals), torch.std(vals))
        else:
            raise ValueError("Invalid Method given for Amortization.")

    def _getFormattedLeakage(self, agg: torch.tensor, std: torch.tensor) -> str:
        return f"{agg:.4f} ± {std:.4f}"
        
    # ============================================================================
    # EVALUATION METRICS INITIALIZATION METHODS
    # ============================================================================
    def initialize_eval_metrics(self, metric: Union[Callable, str]) -> None:
        """
        Initializes the evaluation metric for the predictor model.
        """
        if callable(metric):
            self.eval_metric = metric
        elif type(metric) == str:
            if metric in self.eval_functions:
                self.eval_metric = self.eval_functions[metric]
            else:
                raise ValueError(f"Metric '{metric}' Option is unavailable. User must choose from {list(self.eval_functions.keys())}")
        else:
            raise ValueError("Invalid Metric '{metric}' Given.")

        
    # ============================================================================
    # ABSTRACT METHODS
    # ============================================================================
    @abstractmethod
    def defineModel(self):
        """ Define the attacker_model_D and attacker_model_M models by the subclasses."""
        pass



# ============================================================================
# LEAKAGE Predictability Metric
# ============================================================================
class Leakage(BasePredictabilityMetric):

    def __init__(
        self,
        attacker_model: torch.nn.Module,
        train_params: Dict[str, Any],
        model_acc: float,
        eval_metric: Union[Callable, str] = config.DEFAULT_EVAL_METRIC,
        threshold: bool = True,
        normalized: bool = False,
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
        model_params = {"attacker": attacker_model}
        super().__init__(model_params, train_params, model_acc, eval_metric, threshold, normalized)

    def defineModel(self) -> None:
        self.attacker_D = self.model_params["attacker"]
        self.attacker_M = copy.deepcopy(self.attacker_D)


    def getAmortizedLeakage(
        self,
        feat_train: torch.tensor,
        data_train: torch.tensor,
        pred_train: torch.tensor,
        num_trials: int = config.DEFAULT_NUM_TRIALS,
        method: str = config.DEFAULT_AGGREGATION_METHOD,
        feat_test: torch.tensor = None,
        data_test: torch.tensor = None,
        pred_test: torch.tensor = None,
    ) -> tuple[torch.tensor, torch.tensor]:
        
        return super().getAmortizedLeakage(
            feat_train=feat_train, 
            data_train=data_train, 
            pred_train=pred_train,
            mode=None,
            num_trials=num_trials, 
            method=method, 
            feat_test=feat_test, 
            data_test=data_test, 
            pred_test=pred_test
        )


# ============================================================================
# DPA Predictability Metric
# ============================================================================
class DPA(BasePredictabilityMetric):

    def __init__(
        self,
        attacker_AtoT: torch.nn.Module,
        attacker_TtoA: torch.nn.Module,
        train_params: Dict[str, Any],
        model_acc: Union[float, dict],
        eval_metric: Union[Callable, str] = config.DEFAULT_EVAL_METRIC,
        threshold: bool = True,
        normalized: bool = False,
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
        model_params = {"attacker_AtoT": attacker_AtoT, "attacker_TtoA": attacker_TtoA}
        super().__init__(model_params, train_params, model_acc, eval_metric, threshold, normalized)

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
        num_trials: int = config.DEFAULT_NUM_TRIALS,
        method: str = config.DEFAULT_AGGREGATION_METHOD,
        feat_test: torch.tensor = None,
        data_test: torch.tensor = None,
        pred_test: torch.tensor = None,
    ) -> tuple[torch.tensor, torch.tensor]:
        return super().getAmortizedLeakage(
            feat_train=feat_train, 
            data_train=data_train, 
            pred_train=pred_train,
            mode=mode,
            num_trials=num_trials, 
            method=method, 
            feat_test=feat_test, 
            data_test=data_test, 
            pred_test=pred_test
        )

    def calcBidirectional(
        self,
        A: torch.tensor,
        T: torch.tensor,
        A_pred: torch.tensor,
        T_pred: torch.tensor,
        num_trials: int = config.DEFAULT_NUM_TRIALS,
        method: str = config.DEFAULT_AGGREGATION_METHOD,
    ) -> tuple[tuple[torch.tensor, torch.tensor], tuple[torch.tensor, torch.tensor]]:
        AtoT_vals = self.getAmortizedLeakage(A, T, T_pred, num_trials, method)
        TtoA_vals = self.getAmortizedLeakage(T, A, A_pred, num_trials, method)
        return (AtoT_vals, TtoA_vals)



# ============================================================================
# LIC Predictability Metric
# ============================================================================
class LIC(BasePredictabilityMetric):
    def __init__(self):
        raise NotImplementedError("LIC is not implemented yet.")


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

    train_config = {
        "learning_rate": 0.05,
        "loss_function": "bce",
        "epochs": config.DEFAULT_EPOCHS,
        "batch_size": config.DEFAULT_BATCH_SIZE,
    }

    #Leakage Parameter Initialization
    leakage_obj = Leakage(
        attacker_model=attackerModel,
        train_params=train_config,
        model_acc=model_1_acc,
        eval_metric="accuracy"
    )
    # print("="*50)
    # print(f"Getting Amortized Leakage for Leakage Metric")
    # print("="*50)
    # print("Calculating Leakage for case 1")
    # leakage_1 = leakage_obj.getAmortizedLeakage(P, D, M1)
    # print(f"Leakage for case 1: {leakage_1}")
    # print("="*50)
    # print("="*50)
    # print("Calculating Leakage for case 2")
    # leakage_2 = leakage_obj.getAmortizedLeakage(P, D, M2)
    # print("="*50)
    # print(f"Amortised Leakage for case 2: {leakage_2}")
    # print("="*50)
    # print("="*50)
    # print("Calculating Leakage for case 3")
    # leakage_3 = leakage_obj.getAmortizedLeakage(P, D2, M1)
    # print(f"Leakage for case 3: {leakage_3}")
    # print("="*50)
    # print("="*50)
    # print("Calculating Leakage for case 4")
    # leakage_4 = leakage_obj.getAmortizedLeakage(P, D2, M2)
    # print(f"Leakage for case 4: {leakage_4}")
    # print("="*50)
    # print("="*50)


    
    # Parameter Initialization
    dpa_obj = DPA(
        attacker_AtoT=attackerModel,
        attacker_TtoA=attackerModel,
        train_params=train_config,
        model_acc=model_1_acc,
        eval_metric="accuracy"
    )

    print("="*50)
    print(f"Getting Amortized Leakage for DPA Metric")
    print("="*50)
    print("Calculating DPA for case 1")
    dpa_1 = dpa_obj.getAmortizedLeakage(P, D, M1, "AtoT")
    print(f"DPA for case 1: {dpa_1}")
    print("="*50)
    print("="*50)
    print("Calculating DPA for case 2")
    dpa_2 = dpa_obj.getAmortizedLeakage(P, D, M2, "AtoT")
    print(f"DPA for case 2: {dpa_2}")
    print("="*50)
    print("="*50)
    print("Calculating DPA for case 3")
    dpa_3 = dpa_obj.getAmortizedLeakage(P, D2, M1, "AtoT")
    print(f"DPA for case 3: {dpa_3}")
    print("="*50)
    print("="*50)
    print("Calculating DPA for case 4")
    dpa_4 = dpa_obj.getAmortizedLeakage(P, D2, M2, "AtoT")
    print(f"DPA for case 4: {dpa_4}")
    print("="*50)