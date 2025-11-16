import torch
import pytest
from PredMetrics_v1 import *
from attackerModels.ANN import simpleDenseModel
from utils.datacreator import dataCreator
from utils.losses import ModifiedBCELoss
from unittest.mock import Mock

#===============================================================================
# Reusable test data and models
#===============================================================================

def get_test_data():
    # Data Initialization
    P, D, D_bias, M1, M2 = dataCreator(16384, 0.2, False, 0.05)
    return{
        "P": torch.tensor(P, dtype=torch.float).reshape(-1, 1),
        "D": torch.tensor(D, dtype=torch.float).reshape(-1, 1),
        "D_bias": torch.tensor(D_bias, dtype=torch.float).reshape(-1, 1),
        "M1": torch.tensor(M1, dtype=torch.float).reshape(-1, 1),
        "M2": torch.tensor(M2, dtype=torch.float).reshape(-1, 1)
    }

attacker_model = simpleDenseModel(
    1, 1, 1, numFirst=1, activations=["sigmoid", "sigmoid", "sigmoid"]
    )

train_params = {
    "learning_rate": 0.01,
    "loss_function": "bce",
    "epochs": 5,
    "batch_size": 32
}
model_acc_m1 = lambda test_data: torch.sum(test_data["D"] == test_data["M1"]) / test_data["D"].shape[0]
model_acc_m2 = lambda test_data: torch.sum(test_data["D"] == test_data["M2"]) / test_data["D"].shape[0]

leakage = Leakage(
    attacker_model=attacker_model,
    train_params=train_params,
    model_acc=0.8,
    eval_metric="accuracy"
)

dpa = DPA(
    attacker_AtoT=attacker_model,
    attacker_TtoA=attacker_model,
    train_params=train_params,
    model_acc={"AtoT": 0.8, "TtoA": 0.7},
    eval_metric="accuracy"
)

#===============================================================================
# Test Data Manipulation Methods
#===============================================================================

class TestDataManipulation:
    def test_permute_data(self):
        D = get_test_data()["D"]
        perturbed_D = leakage.permuteData(D)
        # check shape is preserved
        assert perturbed_D.shape == D.shape
        # check approximate match with model accuracy
        matches = torch.sum(perturbed_D == D)/D.shape[0]
        assert abs(leakage.model_acc - matches) < 0.1

    def test_permute_data_with_mode_AtoT(self):
        D = get_test_data()["D"]
        perturbed_D = dpa.permuteData(D, mode="AtoT")
        # check shape is preserved
        matches = torch.sum(perturbed_D == D)/D.shape[0]
        # check approximate match with model accuracy
        assert abs(dpa.model_acc["AtoT"] - matches) < 0.1

    def test_permute_data_with_mode_TtoA(self):
        D = get_test_data()["D"]
        perturbed_D = dpa.permuteData(D, mode="TtoA")
        # check shape is preserved
        matches = torch.sum(perturbed_D == D)/D.shape[0]
        # check approximate match with model accuracy
        assert abs(dpa.model_acc["TtoA"] - matches) < 0.1

    def test_split_data(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        test_size = 0.29
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        # check shape of each split for feat
        assert feat_test.shape[0] == int(test_size * feat.shape[0]) + 1
        assert feat_train.shape[0] == int((feat.shape[0] - feat_test.shape[0]))
        # check shape of each split for data
        assert data_test.shape[0] == int(test_size * data.shape[0]) + 1
        assert data_train.shape[0] == int((data.shape[0] - data_test.shape[0]))
        # check shape of each split for pred
        assert pred_test.shape[0] == int(test_size * pred.shape[0]) + 1
        assert pred_train.shape[0] == int((pred.shape[0] - pred_test.shape[0])) 

    def test_shuffle_data(self):
        data = get_test_data()
        x = data["P"]
        y = data["D"]
        x_shuffled, y_shuffled = leakage.shuffle_data(x, y)
        # check shape is preserved
        assert x_shuffled.shape == x.shape
        assert y_shuffled.shape == y.shape
        # check that x and y are shuffled
        assert not torch.all(x_shuffled == x)
        assert not torch.all(y_shuffled == y)

#===============================================================================
# Test Evaluation Metrics
#===============================================================================

class TestEvaluationMetrics:
    
    def test_initialize_eval_metrics_with_valid_string(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy"
        )
        
        assert callable(leakage.eval_metric)

    def test_initialize_eval_metrics_with_invalid_string(self):
        with pytest.raises(ValueError, match="unavailable"):
            Leakage(
                attacker_model=attacker_model,
                train_params=train_params,
                model_acc=0.8,
                eval_metric="unavailable_metric"
            )

    def test_initialize_eval_metrics_with_valid_callable(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric=ModifiedBCELoss
        )
        assert leakage.eval_metric == ModifiedBCELoss

    def test_initialize_eval_metrics_with_invalid_metric(self):
        with pytest.raises(ValueError, match="Invalid Metric"):
            Leakage(
                attacker_model=attacker_model,
                train_params=train_params,
                model_acc=0.8,
                eval_metric=123
            )


class TestLambdaCalculation:
    def test_leakage_model_definition(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy"
        )
        # testing that the models are defined
        assert isinstance(leakage.attacker_D, torch.nn.Module)
        assert isinstance(leakage.attacker_M, torch.nn.Module)
        # testing that the models are not the same object but deep copies
        assert leakage.attacker_D is not leakage.attacker_M


    def test_calc_lambda_for_eval_acc_with_threshold_true(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.2], [0.8]], dtype=torch.float)
        lambda_value = leakage.calcLambda(mock_model, x, y)
        assert lambda_value == torch.tensor(0.5)


    def test_calc_lambda_for_eval_acc_with_threshold_false(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy",
            threshold=False
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.2], [0.8]], dtype=torch.float)
        lambda_value = leakage.calcLambda(mock_model, x, y)
        assert 0.0 <= lambda_value.item() <= 1.0


    def test_calc_lambda_for_eval_acc_with_all_correct(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.8], [0.2]], dtype=torch.float)
        lambda_value = leakage.calcLambda(mock_model, x, y)
        assert lambda_value == torch.tensor(1.0)


    def test_calc_lambda_for_eval_acc_with_all_wrong(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy"
        )       
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.7], [0.3], [0.2], [0.8]], dtype=torch.float)
        lambda_value = leakage.calcLambda(mock_model, x, y)
        assert lambda_value == torch.tensor(0.0)


    def test_calc_lambda_for_eval_mse_with_threshold_true(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="mse"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.2], [0.8]], dtype=torch.float)
        lambda_value = leakage.calcLambda(mock_model, x, y)
        assert lambda_value.item() == 2.0


    def test_calc_lambda_for_eval_mse_with_threshold_false(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="mse",
            threshold=False
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.2], [0.8]], dtype=torch.float)
        lambda_value = leakage.calcLambda(mock_model, x, y)
        assert 0.0 <= lambda_value.item() <= 5.0


    def test_calc_lambda_for_eval_mse_with_all_correct(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="mse"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.8], [0.2]], dtype=torch.float)
        lambda_value = leakage.calcLambda(mock_model, x, y)
        assert lambda_value.item() == float('inf')


    def test_calc_lambda_for_eval_mse_with_all_wrong(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="mse"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.7], [0.3], [0.2], [0.8]], dtype=torch.float)
        lambda_value = leakage.calcLambda(mock_model, x, y)
        assert lambda_value.item() == 1.0

    def test_calc_lambda_for_eval_bce_with_threshold_true(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="bce"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.2], [0.8]], dtype=torch.float)
        lambda_value = leakage.calcLambda(mock_model, x, y)
        assert 0.0 <= lambda_value.item() <= 0.9


    def test_calc_lambda_for_eval_bce_with_threshold_false(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="bce",
            threshold=False
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.2], [0.8]], dtype=torch.float)
        lambda_value = leakage.calcLambda(mock_model, x, y)
        assert lambda_value.item() >= 1.0


    def test_calc_lambda_for_eval_bce_with_all_correct(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="bce"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.8], [0.2]], dtype=torch.float)
        lambda_value = leakage.calcLambda(mock_model, x, y)
        assert lambda_value.item() == float('inf')


    def test_calc_lambda_for_eval_bce_with_all_wrong(self):
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="bce"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.7], [0.3], [0.2], [0.8]], dtype=torch.float)
        lambda_value = leakage.calcLambda(mock_model, x, y)
        assert lambda_value.item() < 0.1