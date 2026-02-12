import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Callable, Union, Literal
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from bias_amplification.utils.losses import ModifiedBCELoss
from bias_amplification.text.utils.text import CaptionProcessor
from bias_amplification.text.utils.config import LIC_CONFIG, DBAC_CONFIG, DEFAULT_CONFIG

# Type aliases
MaskModeType = Literal["gender", "object"]


class BiasMetricBase(ABC):
    """
    Base class for bias amplification metrics.

    Provides common functionality for leakage-based bias metrics including
    model initialization, training, and evaluation.
    """

    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        gender_words: list[str] = DEFAULT_CONFIG["GENDER_WORDS"],
        obj_words: list[str] = DEFAULT_CONFIG["OBJ_WORDS"],
        gender_token: str = DEFAULT_CONFIG["GENDER_TOKEN"],
        obj_token: str = DEFAULT_CONFIG["OBJ_TOKEN"],
        eval_metric: Union[Callable, str] = DEFAULT_CONFIG["EVAL_METRIC"],
        device: str = DEFAULT_CONFIG["DEVICE"],
        model_path: str | None = None,
        model_type: str = DEFAULT_CONFIG["MODEL_TYPE"],
    ) -> None:
        """
        Initialize base bias metric.

        Parameters
        ----------
        model_params : dict
            Dictionary containing attacker model class and parameters.
        train_params : dict
            Training parameters including learning_rate, loss_function, epochs, batch_size.
        gender_words : list[str]
            List of gender-related words to mask.
        obj_words : list[str]
            List of object-related words to mask.
        gender_token : str
            Token to replace gender words.
        obj_token : str
            Token to replace object words.
        eval_metric : Union[Callable, str], optional
            Evaluation metric. Default is "mse".
        device : str, optional
            Device to use ("cpu" or "cuda"). Default is "cpu".
        **caption_processor_kwargs
            Additional keyword arguments for CaptionProcessor.
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
        self.init_eval_metric(eval_metric)
        
        # Initialize embedding model if specified
        self.embed_model = None
        if self.model_params.get("embedding_model"):
            self.embed_model = SentenceTransformer(
                self.model_params["embedding_model"]
            )
        
        # Initialize caption processor
        self.cap_processor = CaptionProcessor(
            gender_words=gender_words,
            obj_words=obj_words,
            gender_token=gender_token,
            obj_token=obj_token,
            model_path=model_path,
            model_type=model_type,
        )

    def init_eval_metric(self, metric: Union[Callable, str]) -> None:
        """
        Initialize evaluation metric for model evaluation.

        Parameters
        ----------
        metric : Union[Callable, str]
            Evaluation metric function or string identifier.

        Raises
        ------
        ValueError
            If metric is not available or invalid.
        """
        if callable(metric):
            self.eval_metric = metric
        elif isinstance(metric, str):
            if metric in self.eval_functions:
                self.eval_metric = self.eval_functions[metric]
            else:
                raise ValueError(f"Metric {metric} not available.")
        else:
            raise ValueError("Invalid metric given.")

    def define_model(self) -> None:
        """
        Define the attacker models for data (D) and model (M) predictions.

        Creates two instances of the attacker model: one for ground truth
        data (attacker_D) and one for model predictions (attacker_M).
        """
        model_class = self.model_params["attacker_class"]
        model_params = self.model_params["attacker_params"]
        if self.embed_model:
            model_params["input_dims"] = (
                self.embed_model.get_sentence_embedding_dimension()
            )
        else:
            model_params["vocab_size"] = self.vocab_size
        self.attacker_D = model_class(**model_params).to(self.device)
        self.attacker_M = copy.deepcopy(self.attacker_D).to(self.device)

    def split_data(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the data into training and testing sets.
        """
        train, test = train_test_split(
            data.numpy(), 
            train_size=0.8, 
            shuffle=True,
            random_state=42
        )
        return torch.tensor(train), torch.tensor(test)

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attacker_mode: str,
    ) -> None:
        """
        Train the attacker model for a given mode.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Target data.
        attacker_mode : str
            Mode of the attacker model ("D" or "M").

        Notes
        -----
        If attacker_mode is "D", the model is trained to predict the protected
        attribute from the ground truth data.
        If attacker_mode is "M", the model is trained to predict the protected
        attribute from the model's captions.
        """
        self.define_model()
        model = getattr(self, f"attacker_{attacker_mode}")
        model.train()
        criterion = self.loss_functions[self.train_params["loss_function"]]
        optimizer = optim.Adam(
            model.parameters(), lr=self.train_params["learning_rate"]
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        batches = math.ceil(len(x) / self.train_params["batch_size"])
        print(f"Training activated for mode: {attacker_mode}")

        for epoch in range(1, self.train_params["epochs"] + 1):
            perm = torch.randperm(x.shape[0], device=self.device)
            x, y = x[perm], y[perm]
            start, running_loss = 0, 0.0

            for _ in range(batches):
                x_batch = x[
                    start : start + self.train_params["batch_size"]
                ].to(self.device)
                y_batch = y[
                    start : start + self.train_params["batch_size"]
                ].to(self.device)

                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                start += self.train_params["batch_size"]

            scheduler.step()
            print(f"Epoch {epoch} completed!!")
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Avg Loss = {running_loss / batches:.4f}")


    def caption_preprocess(
        self,
        model_captions: pd.Series,
        human_captions: pd.Series,
        mode: MaskModeType = DEFAULT_CONFIG["MASK_MODE"],
        similarity_threshold: float = DBAC_CONFIG["SIMILARITY_THRESHOLD"],
        mask_type: str = DBAC_CONFIG["MASK_TYPE"],
        bidirectional: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess model and human captions.

        Parameters
        ----------
        model_captions : pd.Series
            Model captions to preprocess.
        human_captions : pd.Series
            Human captions to preprocess.
        mode : MaskModeType, optional
            Mask mode ("gender" or "object"). Default is "gender".
        similarity_threshold : float, optional
            Similarity threshold for vocabulary equalization. Default is 0.5.
        mask_type : str, optional
            Type of mask to apply. Default is "contextual".

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Preprocessed model and human captions.
        """
        model_captions = self.cap_processor.mask_words(model_captions, mode=mode)
        human_captions = self.cap_processor.mask_words(human_captions, mode=mode)
        human_captions, model_captions = self.cap_processor.equalize_vocab(
            human_captions,
            model_captions,
            similarity_threshold=similarity_threshold,
            maskType=mask_type,
            bidirectional=bidirectional,
        )
        model_vocab = self.cap_processor.build_vocab(model_captions)
        human_vocab = self.cap_processor.build_vocab(human_captions)
        human_vocab_set = set(human_vocab.get_stoi().keys() if hasattr(human_vocab, 'get_stoi') else human_vocab.stoi.keys())
        model_vocab_set = set(model_vocab.get_stoi().keys() if hasattr(model_vocab, 'get_stoi') else model_vocab.stoi.keys())
        self.vocab_size = len(human_vocab_set.union(model_vocab_set))
        if self.embed_model:
            model_cap = torch.tensor(self.embed_model.encode(model_captions))
            human_cap = torch.tensor(self.embed_model.encode(model_captions))
        else:
            model_cap = self.cap_processor.tokens_to_numbers(
                model_vocab, model_captions
            )
            human_cap = self.cap_processor.tokens_to_numbers(
                human_vocab, human_captions
            )
        return model_cap, human_cap

    @abstractmethod
    def calc_leak(self, feat: torch.tensor, data: torch.tensor, pred: torch.tensor, normalized: bool = False) -> torch.tensor:
        """
        Calculates the leakage of protected attributes (gender, object) from the model's captions to the ground truth captions.
        """
        pass

    @abstractmethod
    def calc_lambda(self, model: torch.nn.Module, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        """
        Calculates the lambda value for a given attacker model, input data and target data.
        """
        pass

    @abstractmethod
    def get_amortized_leakage(self, feat: torch.tensor, data: pd.Series, pred: pd.Series, num_trials, method, normalized, similarity_threshold, mask_type) -> tuple[torch.tensor, torch.tensor]:
        """
        Calculates the amortized leakage of protected attributes (gender, object) from the model's captions to the ground truth captions.
        """
        pass


class LIC(BiasMetricBase):
    """
    Leakage In Captioning (LIC) metric for bias amplification.

    Measures the leakage of protected attributes (gender, object) from the
    model's captions to the ground truth captions.
    """

    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        gender_words: list[str],
        obj_words: list[str],
        gender_token: str,
        obj_token: str,
        eval_metric: Union[Callable, str] = DEFAULT_CONFIG["EVAL_METRIC"],
        model_path: str | None = None,
        model_type: str = DEFAULT_CONFIG["MODEL_TYPE"],
        device: str = DEFAULT_CONFIG["DEVICE"],
    ) -> None:
        """
        Initialize LIC metric.

        Parameters
        ----------
        model_params : dict
            Dictionary containing attacker model class and parameters.
        train_params : dict
            Training parameters including learning_rate, loss_function, epochs, batch_size.
        gender_words : list[str]
            List of gender-related words to mask.
        obj_words : list[str]
            List of object-related words to mask.
        gender_token : str
            Token to replace gender words.
        obj_token : str
            Token to replace object words.
        eval_metric : Union[Callable, str], optional
            Evaluation metric. Default is "mse".
        model_path : str | None, optional
            Path to embedding model file. Default is None.
        model_type : str, optional
            Type of embedding model ("glove", "fasttext", "bert"). Default is "glove".
        device : str, optional
            Device to use ("cpu" or "cuda"). Default is "cpu".
        """
        super().__init__(
            model_params=model_params,
            train_params=train_params,
            gender_words=gender_words,
            obj_words=obj_words,
            gender_token=gender_token,
            obj_token=obj_token,
            eval_metric=eval_metric,
            device=device,
            model_path=model_path,
            model_type=model_type,
        )

    def calc_lambda(self, model, x, y):
        """
        Calculates the lambda value for a given attacker model, input data and target data.

        Parameters
        ----------
        model : torch.nn.Module
            Attacker model.
        x : torch.tensor
            Input data.
        y : torch.tensor
            Target data.

        Returns
        -------
        vals : torch.tensor
            Lambda value.
        """
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

    def calc_leak(
        self,
        feat: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
        normalized: bool = False
    ) -> torch.tensor:
        """
        Calculates the leakage of protected attributes (gender, object) from the model's captions to the ground truth captions. Returns the difference between the leakage from the model's captions and the ground truth captions.
        
        Parameters
        ----------
        feat : torch.tensor
            Protected Attribute.
        data : torch.tensor
            Ground truth data.
        pred : torch.tensor
            Predicted Values.
        normalized : bool, optional
            Whether to normalize the leakage. Default is False.

        Returns
        -------
        leakage : torch.tensor
            Evaluated Leakage.

        Notes
        -----
        λ_D represents baseline leakage from perturbed data
        λ_M represents leakage from model predictions
        """
        self.train(data, feat, "D")
        lambda_d = self.calc_lambda(getattr(self, "attacker_D"), data, feat)
        self.train(pred, feat, "M")
        lambda_m = self.calc_lambda(getattr(self, "attacker_M"), pred, feat)
        print(f"{lambda_d=},\n{lambda_m=}")
        leakage_amp = lambda_m - lambda_d
        if normalized:
            leakage_amp = leakage_amp / (lambda_m + lambda_d)
        return leakage_amp


    def get_amortized_leakage(
        self,
        feat: torch.tensor,
        data: pd.Series,
        pred: pd.Series,
        num_trials: int = DEFAULT_CONFIG["NUM_TRIALS"],
        method: str = DEFAULT_CONFIG["METHOD"],
        normalized: bool = False,
        similarity_threshold=LIC_CONFIG["SIMILARITY_THRESHOLD"],
        mask_type=LIC_CONFIG["MASK_TYPE"],
        mask_mode=DEFAULT_CONFIG["MASK_MODE"],
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Calculates the amortized leakage of protected attributes (gender, object) from the model's captions to the ground truth captions.

        Parameters
        ----------
        feat : torch.tensor
            Protected Attribute.
        data : pd.Series
            Ground truth data.
        pred : pd.Series
            Predicted Values.
        num_trials : int, optional
            Number of trials to run. Default is 25.
        method : str, optional
            Method to use for amortization. Default is "mean".
        normalized : bool, optional
            Whether to normalize the leakage. Default is False.
        similarity_threshold : float, optional
            Similarity threshold for vocabulary equalization. Default is 1 for LIC.
        mask_type : str, optional
            Type of mask to apply. For LIC, it is "constant".
        Returns
        -------
        Tuple[torch.tensor, torch.tensor]
        Mean/Median : torch.tensor
            Mean/median amortized leakage based on the method provided.
        std : torch.tensor
            Standard deviation of the amortized leakage.
        num_trials : int
            Number of trials used.  
        """
        
        pred, data = self.caption_preprocess(
            model_captions= pred,
            human_captions=data,
            mode=mask_mode,
            similarity_threshold=similarity_threshold,
            mask_type=mask_type,
            bidirectional=False,
        )
        pred = pred.to(self.device)
        data = data.to(self.device)
        feat = feat.to(self.device)
        vals = torch.zeros(num_trials)
        for i in range(num_trials):
            print(f"Working on Trial: {i}")
            vals[i] = self.calc_leak(feat, data, pred, normalized).item()
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


class DBAC(BiasMetricBase):
    """
    Directional Bias Amplification Content (DBAC) metric.

    Measures directional bias amplification in text data with support for
    object-based analysis.
    """

    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        gender_words: list[str],
        obj_words: list[str],
        gender_token: str,
        obj_token: str,
        eval_metric: Union[Callable, str] = DEFAULT_CONFIG["EVAL_METRIC"],
        model_path: str | None = None,
        device: str = DEFAULT_CONFIG["DEVICE"],
        model_type: str = DEFAULT_CONFIG["MODEL_TYPE"]
    ) -> None:
        """
        Initialize DBAC metric.

        Parameters
        ----------
        model_params : dict
            Dictionary containing attacker model class and parameters.
        train_params : dict
            Training parameters including learning_rate, loss_function, epochs, batch_size.
        gender_words : list[str]
            List of gender-related words to mask.
        obj_words : list[str]
            List of object-related words to mask.
        gender_token : str
            Token to replace gender words.
        obj_token : str
            Token to replace object words.
        eval_metric : Union[Callable, str], optional
            Evaluation metric. Default is "mse".
        model_path : str | None, optional
            Path to embedding model file. Default is None.
        model_type : str, optional
            Type of embedding model ("glove", "fasttext", "bert"). Default is "glove".
        device : str, optional
            Device to use ("cpu" or "cuda"). Default is "cpu".
        mask_mode : str, optional
            Mask mode ("gender" or "object"). Default is "gender".
        """
        super().__init__(
            model_params=model_params,
            train_params=train_params,
            gender_words=gender_words,
            obj_words=obj_words,
            gender_token=gender_token,
            obj_token=obj_token,
            eval_metric=eval_metric,
            device=device,
            model_path=model_path,
            model_type=model_type,
        )
    
    def calc_leak(
        self,
        feat: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
        data_objs: np.array = None,
        pred_objs: np.array = None,
        apply_bayes: bool = True,
        normalized: bool = True,
        mask_mode: MaskModeType = DEFAULT_CONFIG["MASK_MODE"],
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
        data_objs : np.array
            Object occurrence information for ground truth data.
        pred_objs : np.array
            Object occurrence information for predicted data.
        apply_bayes : bool, optional
            Whether to apply Bayesian correction. Default is True.
        normalized : bool, optional
            Whether to normalize the leakage. Default is True.
        mask_mode : MaskModeType, optional

        Returns
        -------
        leakage : torch.tensor
            Evaluated Leakage.

        """
        # Perform vocab equalization
        data_train, data_test = self.split_data(data)
        self.train(data_train, feat, "D")
        lambda_d = self.calc_lambda(
            model=getattr(self, "attacker_D"),
            x=data_test,
            y=feat,
            objs=data_objs,
            apply_bayes=apply_bayes,
            mask_mode=mask_mode
        )
        pred_train, pred_test = self.split_data(pred)
        self.train(pred_train, feat, "M")
        lambda_m = self.calc_lambda(
            model=getattr(self, "attacker_M"), 
            x=pred_test,
            y=feat,
            objs=pred_objs,
            apply_bayes=apply_bayes,
            mask_mode=mask_mode
        )
        print(f"{lambda_d=:.4f},\n{lambda_m=:.4f}")
        leakage_amp = lambda_m - lambda_d
        if normalized:
            leakage_amp = leakage_amp / (lambda_m + lambda_d)
        return leakage_amp

    def get_probs_occurrences(
        self, occurrence_info: torch.Tensor
    ) -> torch.Tensor:
        """
        Get probabilities from object occurrence information.

        Parameters
        ----------
        occurrence_info : torch.Tensor
            Object occurrence information.

        Returns
        -------
        torch.Tensor
            Probability values.
        """
        val, inverse, counts = torch.unique(
            occurrence_info, return_inverse=True, return_counts=True, dim=0
        )
        counts = counts / counts.sum()
        return counts[inverse]

    def get_probs(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        mask_mode: MaskModeType = DEFAULT_CONFIG["MASK_MODE"],
    ) -> torch.Tensor:
        """
        Get probabilities based on mask mode.

        Parameters
        ----------
        y : torch.Tensor
            Ground truth labels.
        y_pred : torch.Tensor
            Predicted labels.
        mask_mode : MaskModeType, optional
            Mask mode ("gender" or "object"). Default is "gender".

        Returns
        -------
        torch.Tensor
            Probability values.
        """
        if mask_mode == "gender":
            args = y.argmax(axis=1)
            nums = np.arange(len(y))
            y_pred = y_pred.type(torch.float)
            probs = y_pred[nums, args]
        else:
            probs = (y_pred * y) + ((1 - y) * (1 - y_pred))
            probs = probs.prod(axis=1)
        return probs

    def calc_lambda(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        objs: np.ndarray = None,
        apply_bayes: bool = True,
        mask_mode: MaskModeType = DEFAULT_CONFIG["MASK_MODE"],
    ) -> torch.Tensor:
        """
        Calculate lambda value with object-based analysis.

        Parameters
        ----------
        model : torch.nn.Module
            Attacker model.
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Target data.
        objs : np.ndarray, optional
            Object occurrence information.
        apply_bayes : bool, optional
            Whether to apply Bayesian correction. Default is True for DBAC.
        mask_mode : MaskModeType, optional
            Mask mode ("gender" or "object"). Default is "gender".

        Returns
        -------
        torch.Tensor
            Lambda value.
        """
        model.eval()
        y_pred = torch.zeros_like(y).to(self.device)
        start = 0
        batches = math.ceil(len(x) / self.train_params["batch_size"])
        for batch_num in range(batches):
            x_batch = x[start : (start + self.train_params["batch_size"])]
            y_pred[start : (start + self.train_params["batch_size"])] = model(
                x_batch
            )
            start += self.train_params["batch_size"]
        y = y.type(torch.float)
        probs = self.get_probs(y, y_pred, mask_mode)
        if apply_bayes:
            objs = torch.tensor(objs).to(self.device)
            probs_obj = self.get_probs_occurrences(objs)
            probs_attr = self.get_probs_occurrences(y)
            probs = (probs * probs_obj) / probs_attr
        return probs.mean()

    def get_amortized_leakage(
        self,
        feat: torch.tensor,  # Attribute
        data: pd.Series,  # Human Captions (straight from datacreator)
        pred: pd.Series,  # Model Captions (straight from datacreator)
        pred_objs: np.array = None,
        data_objs: np.array = None,
        num_trials: int = DEFAULT_CONFIG["NUM_TRIALS"],
        method: str = DEFAULT_CONFIG["METHOD"],
        apply_bayes: bool = True,
        normalized: bool = True,
        mask_mode: MaskModeType = DEFAULT_CONFIG["MASK_MODE"],
        mask_type=DBAC_CONFIG["MASK_TYPE"],
        similarity_threshold: float = DBAC_CONFIG["SIMILARITY_THRESHOLD"],
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Calculates the amortized leakage of protected attributes (gender, object) from the model's captions to the ground truth captions.

        Parameters
        ----------
        feat : torch.tensor
            Protected Attribute.
        data : pd.Series
            Ground truth data for DBAC.
        pred : pd.Series
            Model captions for DBAC.
        pred_objs : np.array, optional
            Object occurrence information for predicted data.
        data_objs : np.array, optional
            Object occurrence information for ground truth data.
        num_trials : int, optional
            Number of trials to run. Default is 25.
        method : str, optional
            Method to use for amortization. Default is "mean".
        apply_bayes : bool, optional
            Whether to apply Bayesian correction. Default is True for DBAC.
        normalized : bool, optional
            Whether to normalize the leakage. Default is True.
        mask_mode : MaskModeType, optional
            Mask mode ("gender" or "object"). Default is "gender".
        mask_type : str, optional
            Type of mask to apply. Default is "contextual".
        similarity_threshold : float, optional
            Similarity threshold for vocabulary equalization. Default is 0.5 for DBAC.

        Returns
        -------
        tuple[torch.tensor, torch.tensor]
            Amortized leakage as mean/median and number of trials.
        """
        # pred = pred_frame["caption"]
        # data = data_frame["caption"]
        # pred_objs = pred_frame.drop("caption", axis=1).to_numpy()
        # data_objs = data_frame.drop("caption", axis=1).to_numpy()
        pred, data = self.caption_preprocess(
            model_captions=pred,
            human_captions=data,
            mode=mask_mode,
            similarity_threshold=similarity_threshold,
            mask_type=mask_type,
            bidirectional=True,
        )
        pred = pred.to(self.device)
        data = data.to(self.device)
        feat = feat.to(self.device)
        vals = torch.zeros(num_trials)
        for i in range(num_trials):
            print(f"Working on Trial: {i}")
            vals[i] = self.calc_leak(
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

    
if __name__ == "__main__":
    from bias_amplification.text.attacker_models import LSTM_ANN_Model
    import os
    from bias_amplification.text.Dataset_test.datacreator import CaptionGenderDataset
    script_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "text/Dataset_test"
        )
    HUMAN_ANN_PATH = os.path.join(script_dir, "gender_obj_cap_mw_entries.pkl")
    MODEL_ANN_PATH = os.path.join(script_dir, "gender_val_transformer_cap_mw_entries.pkl")
    # HUMAN_ANN_PATH = "./bias_data/Human_Ann/gender_obj_cap_mw_entries.pkl"
    # MODEL_ANN_PATH = "./bias_data/Transformer/gender_val_transformer_cap_mw_entries.pkl"
    MODEL_PATH = os.path.join(script_dir, "cc.en.300.bin")
    # MODEL_PATH = os.path.join(script_dir, "glove.6B.50d.w2vformat.txt")
    MODEL="fasttext"
    MASCULINE = [
        "man",
        "men",
        "male",
        "father",
        "gentleman",
        "boy",
        "uncle",
        "husband",
        "actor",
        "prince",
        "waiter",
        "he",
        "his",
        "him",
    ]
    FEMININE = [
        "woman",
        "women",
        "female",
        "mother",
        "lady",
        "girl",
        "aunt",
        "wife",
        "actress",
        "princess",
        "waitress",
        "she",
        "her",
        "hers",
    ]
    GENDER_WORDS = MASCULINE + FEMININE
    GENDER_TOKEN = "<unk>"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_obj = CaptionGenderDataset(HUMAN_ANN_PATH, MODEL_ANN_PATH)
    ann_data = data_obj.getDataCombined()
    object_presence_df = data_obj.get_object_presence_df()
    OBJ_WORDS = object_presence_df.columns.tolist()
    OBJ_TOKEN = "<obj>"

    human_ann = ann_data["caption_human"]
    model_ann = ann_data["caption_model"]
    gender = torch.tensor(ann_data["gender"].values, dtype = torch.float32).reshape(-1, 1)

    model_params = {
        "attacker_class": LSTM_ANN_Model,
        "attacker_params": {
            "embedding_dim": 250,
            "pad_idx": 0,
            "lstm_hidden_size": 256,
            "lstm_num_layers": 2,
            "lstm_bidirectional": True,
            "ann_output_size": 1,
            "num_ann_layers": 5,
            "ann_numFirst": 64,
        },
    }
    # Change format to intialize within LIC to allow vocab size to be passed later on.
    train_params = {
        "learning_rate": 0.001,
        "loss_function": "bce",
        "epochs": 100,
        "batch_size": 1024,
    }

    LIC_obj = LIC(
        model_params=model_params,
        train_params=train_params,
        gender_words=GENDER_WORDS,
        obj_words=OBJ_WORDS,
        gender_token=GENDER_TOKEN,
        obj_token=OBJ_TOKEN,
        eval_metric="mse",
        model_path=MODEL_PATH,
        model_type=MODEL,
        device=DEVICE,
    )

    lic_analysis_data = LIC_obj.get_amortized_leakage(
        feat=gender,
        data=human_ann,
        pred=model_ann,
        num_trials=1,
        method="mean",
        normalized=False,
        similarity_threshold=1,
        mask_type="constant",
        mask_mode="gender",
    )

    DBAC_obj = DBAC(
        model_params=model_params,
        train_params=train_params,
        gender_words=GENDER_WORDS,
        obj_words=OBJ_WORDS,
        gender_token=GENDER_TOKEN,
        obj_token=OBJ_TOKEN,
        eval_metric="bce",
        model_path=MODEL_PATH,
        model_type=MODEL,
        device=DEVICE
    )

    dbac_analysis_data = DBAC_obj.get_amortized_leakage(
        feat=gender,
        data=human_ann,
        pred=model_ann,
        num_trials=1,
        method="mean",
        apply_bayes=True,
        normalized=False,
        similarity_threshold=0.5,
        mask_type="contextual",
        mask_mode="gender",
    )