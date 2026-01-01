# Importing Libraries
import copy
import math
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from typing import Callable, Union, Literal
from utils.losses import ModifiedBCELoss
from utils.text import CaptionProcessor
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper Types
maskModeType = Literal["gender", "object"]


# Main class
class LIC:
    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        gender_words: list[str],
        obj_words: list[str],
        gender_token: str,
        obj_token: str,
        eval_metric: Union[Callable, str] = "mse",
        threshold=True,
        glove_path=None,
        device="cpu",
    ) -> None:
        self.model_params = model_params
        self.train_params = train_params
        self.model_attacker_trained = False
        self.threshold = threshold
        self.device = device

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
        self.embed_model = None
        if self.model_params.get("embedding_model"):
            self.embed_model = SentenceTransformer(self.model_params["embedding_model"])
        self.capProcessor = CaptionProcessor(
            gender_words,
            obj_words,
            gender_token=gender_token,
            obj_token=obj_token,
            glove_path=glove_path,
        )

    def initEvalMetric(self, metric: Union[Callable, str]) -> None:
        """
        Initialize evaluation metric for model evaluation.
        """
        if callable(metric):
            self.eval_metric = metric
        elif isinstance(metric, str):
            if metric in self.eval_functions:
                self.eval_metric = self.eval_functions[metric]
            else:
                raise ValueError(f"Metric {metric} not available.")
        else:
            raise ValueError("Invalid Metric Given.")

    def calcLeak(
        self,
        feat: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
        normalized: bool = False,
    ) -> torch.tensor:
        self.train(data, feat, "D")
        lambda_d = self.calcLambda(getattr(self, "attacker_D"), data, feat)
        self.train(pred, feat, "M")
        lambda_m = self.calcLambda(getattr(self, "attacker_M"), pred, feat)
        print(f"{lambda_d=},\n{lambda_m=}")
        leakage_amp = lambda_m - lambda_d
        if normalized:
            leakage_amp = leakage_amp / (lambda_m + lambda_d)
        return leakage_amp

    def train(self, x, y, attacker_mode):
        self.defineModel()
        model = getattr(self, "attacker_" + attacker_mode)
        criterion = self.loss_functions[self.train_params["loss_function"]]
        optimizer = optim.Adam(model.parameters(), lr=self.train_params["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        batches = math.ceil(len(x) / self.train_params["batch_size"])
        print(f"Training Activated for Mode: {attacker_mode}")

        for epoch in range(1, self.train_params["epochs"] + 1):
            perm = torch.randperm(x.shape[0], device=self.device)
            x, y = x[perm], y[perm]
            start, running_loss = 0, 0.0

            for _ in range(batches):
                x_batch = x[start : start + self.train_params["batch_size"]].to(self.device)
                y_batch = y[start : start + self.train_params["batch_size"]].to(self.device)

                optimizer.zero_grad()
                outputs = model(x_batch)

                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                start += self.train_params["batch_size"]

            scheduler.step()

            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Avg Loss = {running_loss / batches:.4f}")

    def calcLambda(self, model, x, y):
        model.eval()
        batch_size = self.train_params.get("batch_size", 32)

        y_pred_list = []
        total_samples = x.shape[0]

        with torch.no_grad():
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                x_batch = x[start:end].to(self.device)
                batch_pred = model(x_batch)
                y_pred_list.append(batch_pred.cpu())
        y_pred = torch.cat(y_pred_list, dim=0).to(self.device)
        matches = (y_pred.argmax(axis=1) == y.argmax(axis=1)) * 1.0
        vals = y_pred.max(dim=1).values * matches
        return vals.mean()

    def defineModel(self):
        model_class = self.model_params["attacker_class"]
        model_params = self.model_params["attacker_params"]
        if self.embed_model:
            model_params["input_dims"] = self.embed_model.get_sentence_embedding_dimension()
        else:
            model_params["vocab_size"] = self.vocab_size
        self.attacker_D = model_class(**model_params).to(self.device)
        self.attacker_M = copy.deepcopy(self.attacker_D).to(self.device)

    def captionPreprocess(
        self, model_captions, human_captions, similarity_thresh=1, mask_type="constant"
    ):
        model_captions = self.capProcessor.maskWords(model_captions, mode="gender")
        human_captions = self.capProcessor.maskWords(human_captions, mode="gender")
        human_captions, model_captions = self.capProcessor.equalize_vocab(
            human_captions,
            model_captions,
            similarity_threshold=similarity_thresh,
            maskType=mask_type,
        )
        model_vocab = self.capProcessor.build_vocab(model_captions)
        human_vocab = self.capProcessor.build_vocab(human_captions)
        human_vocab_set = set(human_vocab.stoi.keys())
        model_vocab_set = set(model_vocab.stoi.keys())
        self.vocab_size = len(human_vocab_set.union(model_vocab_set))
        if self.embed_model:
            model_cap = torch.tensor(self.embed_model.encode(model_captions))
            human_cap = torch.tensor(self.embed_model.encode(model_captions))
        else:
            model_cap = self.capProcessor.tokens_to_numbers(model_vocab, model_captions)
            human_cap = self.capProcessor.tokens_to_numbers(human_vocab, human_captions)
        return model_cap, human_cap

    def getAmortizedLeakage(
        self,
        feat: torch.tensor,
        data: pd.Series,
        pred: pd.Series,
        num_trials: int = 25,
        method: str = "mean",
        normalized: bool = False,
        similarity_threshold=1,
        mask_type="constant",
    ) -> tuple[torch.tensor, torch.tensor]:
        pred, data = self.captionPreprocess(pred, data, similarity_threshold, mask_type)
        pred = pred.to(self.device)
        data = data.to(self.device)
        feat = feat.to(self.device)
        vals = torch.zeros(num_trials)
        for i in range(num_trials):
            print(f"Working on Trial: {i}")
            vals[i] = self.calcLeak(feat, data, pred, normalized).item()
            print(f"Trial {i} val: {vals[i]}")
        if method == "mean":
            return {
                "Mean": torch.mean(vals),
                "std": torch.std(vals),
                "num_trials": num_trials,
            }
        elif method == "median":
            return {
                "Median": torch.median(vals),
                "std": torch.std(vals),
                "num_trials": num_trials,
            }
        else:
            raise ValueError("Invalid Method given for Amortization.")


class DBAC:
    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        gender_words: list[str],
        obj_words: list[str],
        gender_token: str,
        obj_token: str,
        eval_metric: Union[Callable, str] = "mse",
        glove_path=None,
        device="cpu",
        sub_model="glove",
    ) -> None:
        """
        Parameters
        ----------
        model_params : dict
            Dictionary of the following forms-
            {"attacker_class" : model_class,
             "attacker_params" : model_init_params}
        train_params : dict
            {
                "learning_rate": The learning rate hyperparameter,
                "loss_function": The loss function to be used.
                        Existing options: ["mse", "cross-entropy"],
                "epochs": Number of training epochs to be set,
                "batch_size: Number of batches per epoch
            }
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
        self.model_attacker_trained = False
        self.device = device

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
        self.embed_model = None
        if self.model_params.get("embedding_model"):
            self.embed_model = SentenceTransformer(self.model_params["embedding_model"])
        self.capProcessor = CaptionProcessor(
            gender_words,
            obj_words,
            gender_token=gender_token,
            obj_token=obj_token,
            glove_path=glove_path,
            model_type=sub_model,
        )

    def calcLeak(
        self,
        feat: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
        data_objs: np.array,
        pred_objs: np.array,
        apply_bayes: bool = True,
        normalized: bool = True,
        mask_mode: maskModeType = "gender",
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

        Returns
        -------
        leakage : torch.tensor
            Evaluated Leakage.

        """
        # Perform vocab equalization
        self.train(data, feat, "D")
        lambda_d = self.calcLambda(
            getattr(self, "attacker_D"), data, feat, data_objs, apply_bayes, mask_mode
        )
        self.train(pred, feat, "M")
        lambda_m = self.calcLambda(
            getattr(self, "attacker_M"), pred, feat, pred_objs, apply_bayes, mask_mode
        )
        print(f"{lambda_d=},\n{lambda_m=}")
        leakage_amp = lambda_m - lambda_d
        if normalized:
            leakage_amp = leakage_amp / (lambda_m + lambda_d)
        return leakage_amp

    def getProbsfromObjectOccurences(self, occurence_info: torch.tensor) -> torch.tensor:
        val, inverse, counts = torch.unique(
            occurence_info, return_inverse=True, return_counts=True, dim=0
        )
        counts = counts / counts.sum()
        return counts[inverse]

    def train(
        self,
        x: torch.tensor,
        y: torch.tensor,
        attacker_mode: str,
    ) -> torch.tensor:
        self.defineModel()
        model = getattr(self, "attacker_" + attacker_mode)
        model.train()
        criterion = self.loss_functions[self.train_params["loss_function"]]
        optimizer = optim.Adam(model.parameters(), lr=self.train_params["learning_rate"])
        batches = math.ceil(len(x) / self.train_params["batch_size"])

        print(f"Training Activated for Mode: {attacker_mode}")

        # Training loop
        for epoch in range(1, self.train_params["epochs"] + 1):
            perm = torch.randperm(x.shape[0])
            x = x[perm]
            y = y[perm]
            start = 0
            running_loss = 0.0
            # print(batches)
            for batch_num in range(batches):
                x_batch = x[start : (start + self.train_params["batch_size"])]
                y_batch = y[start : (start + self.train_params["batch_size"])]

                optimizer.zero_grad()
                # Forward pass
                outputs = model(x_batch)
                # print(f"{outputs=}\n{y_batch=}")
                loss = criterion(outputs, y_batch)
                # print(f"{loss.item()=}")

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                start += self.train_params["batch_size"]
                running_loss += loss.item()

            avg_loss = running_loss / batches
            if epoch % 10 == 0:
                print(f"\rCurrent Epoch {epoch}: Loss = {avg_loss}", end="")

        print("\nModel training completed")

    def getProbs(
        self,
        y: torch.tensor,
        y_pred: torch.tensor,
        mask_mode: maskModeType = "gender",
    ) -> torch.tensor:
        if mask_mode == "gender":
            args = y.argmax(axis=1)
            nums = np.arange(len(y))
            y_pred = y_pred.type(torch.float)
            probs = y_pred[nums, args]
        else:
            probs = (y_pred * y) + ((1 - y) * (1 - y_pred))
            probs = probs.prod(axis=1)
        return probs

    def calcLambda(
        self,
        model: torch.nn.Module,
        x: torch.tensor,
        y: torch.tensor,
        objs: np.array,
        apply_bayes: bool = True,
        mask_mode: maskModeType = "gender",
    ) -> torch.tensor:
        model.eval()
        y_pred = torch.zeros_like(y).to(self.device)
        start = 0
        batches = math.ceil(len(x) / self.train_params["batch_size"])
        for batch_num in range(batches):
            x_batch = x[start : (start + self.train_params["batch_size"])]
            y_pred[start : (start + self.train_params["batch_size"])] = model(x_batch)
            start += self.train_params["batch_size"]
        y = y.type(torch.float)
        probs = self.getProbs(y, y_pred, mask_mode)
        if apply_bayes:
            objs = torch.tensor(objs).to(self.device)
            probs_obj = self.getProbsfromObjectOccurences(objs)
            probs_attr = self.getProbsfromObjectOccurences(y)
            probs = (probs * probs_obj) / probs_attr
        return probs.mean()

    def defineModel(self) -> None:
        model_class = self.model_params["attacker_class"]
        model_params = self.model_params["attacker_params"]
        if self.embed_model:
            model_params["input_dims"] = self.embed_model.get_sentence_embedding_dimension()
        else:
            model_params["vocab_size"] = self.vocab_size
        self.attacker_D = model_class(**model_params)
        self.attacker_D.to(self.device)
        self.attacker_M = copy.deepcopy(self.attacker_D)
        self.attacker_M.to(self.device)

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

    def captionPreprocess(
        self,
        model_captions: pd.Series,
        human_captions: pd.Series,
        mode: maskModeType = "gender",
        similarity_threshold=0.5,
        maskType="contextual",
    ) -> tuple[torch.tensor, torch.tensor]:  # type: ignore
        model_captions = self.capProcessor.maskWords(model_captions, mode=mode)
        human_captions = self.capProcessor.maskWords(human_captions, mode=mode)
        human_captions, model_captions = self.capProcessor.equalize_vocab(
            human_captions,
            model_captions,
            similarity_threshold=similarity_threshold,
            maskType=maskType,
            bidirectional=True,
        )
        model_vocab = self.capProcessor.build_vocab(model_captions)
        human_vocab = self.capProcessor.build_vocab(human_captions)
        human_vocab_set = set(human_vocab.stoi.keys())
        model_vocab_set = set(model_vocab.stoi.keys())
        self.vocab_size = len(human_vocab_set.union(model_vocab_set))
        if self.embed_model:
            model_cap = torch.tensor(self.embed_model.encode(model_captions))
            human_cap = torch.tensor(self.embed_model.encode(model_captions))
        else:
            model_cap = self.capProcessor.tokens_to_numbers(model_vocab, model_captions)
            human_cap = self.capProcessor.tokens_to_numbers(human_vocab, human_captions)
        return model_cap, human_cap

    def getAmortizedLeakage(
        self,
        feat: torch.tensor,  # Attribute
        data_frame: pd.DataFrame,  # Human Captions (straight from datacreator)
        pred_frame: pd.DataFrame,  # Model Captions (straight from datacreator)
        num_trials: int = 10,
        method: str = "mean",
        apply_bayes: bool = True,
        normalized: bool = True,
        mask_mode: maskModeType = "gender",
        mask_type="contextual",
        similarity_threshold: float = 0.5,
    ) -> tuple[torch.tensor, torch.tensor]:
        pred = pred_frame["caption"]
        data = data_frame["caption"]
        pred_objs = pred_frame.drop("caption", axis=1).to_numpy()
        data_objs = data_frame.drop("caption", axis=1).to_numpy()
        pred, data = self.captionPreprocess(pred, data, mask_mode, similarity_threshold, mask_type)
        pred = pred.to(self.device)
        data = data.to(self.device)
        feat = feat.to(self.device)
        vals = torch.zeros(num_trials)
        for i in range(num_trials):
            print(f"Working on Trial: {i}")
            vals[i] = self.calcLeak(
                feat,
                data,
                pred,
                data_objs,
                pred_objs,
                apply_bayes,
                normalized,
                mask_mode,
            ).item()
            print(f"Trial {i} val: {vals[i]}")
        if method == "mean":
            return {
                "Mean": torch.mean(vals),
                "std": torch.std(vals),
                "num_trials": num_trials,
            }
        elif method == "median":
            return {
                "Median": torch.median(vals),
                "std": torch.std(vals),
                "num_trials": num_trials,
            }
        else:
            raise ValueError("Invalid Method given for Amortization.")
