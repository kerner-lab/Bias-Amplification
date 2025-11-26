import torch
import pytest
from metrics.PredMetrics_v1 import *
from attackerModels.ANN import simpleDenseModel
from utils.datacreator import dataCreator
from utils.losses import ModifiedBCELoss
from unittest.mock import Mock, patch

#===============================================================================
# Reusable test data and models
#===============================================================================

def get_test_data():
    # Data Initialization
    P, D, D_bias, M1, M2 = dataCreator(128, 0.2, False, 0.05)
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

dpa_1 = DPA(
    attacker_AtoT=attacker_model,
    attacker_TtoA=attacker_model,
    train_params=train_params,
    model_acc=0.8,
    eval_metric="accuracy"
)

#===============================================================================
# Test Data Manipulation Methods
#===============================================================================

class TestDataManipulation:
    def test_permute_data_acc_correct_float(self):
        D = get_test_data()["D"]
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy"
        )
        perturbed_D = leakage.permuteData(D)
        # check shape is preserved
        assert perturbed_D.shape == D.shape
        # check approximate match with model accuracy
        matches = torch.sum(perturbed_D == D)/D.shape[0]
        assert abs(leakage.model_acc - matches) < 0.1
    
    def test_permute_data_acc_wrong_float(self):
        D = get_test_data()["D"]
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=8.8,
            eval_metric="accuracy"
        )
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0 for float type"):
            leakage.permuteData(D)

    def test_permute_data_acc_int(self):
        D = get_test_data()["D"]
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=80,
            eval_metric="accuracy"
        )
        perturbed_D = leakage.permuteData(D)
        # check shape is preserved
        assert perturbed_D.shape == D.shape
        # check approximate match with model accuracy
        matches = torch.sum(perturbed_D == D)/D.shape[0]
        assert abs(leakage.model_acc - matches) < 0.1

    def test_permute_data_acc_wrong_int(self):
        D = get_test_data()["D"]
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=800,
            eval_metric="accuracy"
        )
        with pytest.raises(ValueError, match="must be between 1 and 100 for int type"):
            leakage.permuteData(D)

    def test_permute_data_acc_correct_tensor(self):
        D = get_test_data()["D"]
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=torch.tensor(0.8),
            eval_metric="accuracy"
        )
        perturbed_D = leakage.permuteData(D)
        # check shape is preserved
        assert perturbed_D.shape == D.shape
        # check approximate match with model accuracy
        matches = torch.sum(perturbed_D == D)/D.shape[0]
        assert abs(leakage.model_acc - matches) < 0.1

    def test_permute_data_acc_wrong_tensor(self):
        D = get_test_data()["D"]
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc=torch.tensor(8.8),
            eval_metric="accuracy"
        )
        with pytest.raises(ValueError, match="tensor value must be between 0.0 and 1.0"):
            leakage.permuteData(D)
    
    def test_permute_data_acc_invalid_type(self):
        D = get_test_data()["D"]
        leakage = Leakage(
            attacker_model=attacker_model,
            train_params=train_params,
            model_acc="invalid_type",
            eval_metric="accuracy"
        )
        with pytest.raises(ValueError, match="Invalid model accuracy type given"):
            leakage.permuteData(D)

    def test_permute_data_with_mode_AtoT_acc_correct_float(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": 0.8, "TtoA": 0.7},
            eval_metric="accuracy"
        )
        perturbed_D = dpa.permuteData(D, mode="AtoT")
        # check shape is preserved
        matches = torch.sum(perturbed_D == D)/D.shape[0]
        # check approximate match with model accuracy
        assert abs(dpa.model_acc["AtoT"] - matches) < 0.1

    def test_permute_data_with_mode_AtoT_acc_wrong_float(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": 8.5, "TtoA": 7.5},
            eval_metric="accuracy"
        )

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0 for float type"):
            dpa.permuteData(D, mode="AtoT")

    def test_permute_data_with_mode_AtoT_acc_correct_int(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": 80, "TtoA": 70},
            eval_metric="accuracy"
        )
        perturbed_D = dpa.permuteData(D, mode="AtoT")
        # check shape is preserved
        matches = torch.sum(perturbed_D == D)/D.shape[0]
        # check approximate match with model accuracy
        assert abs(dpa.model_acc["AtoT"] - matches) < 0.1

    def test_permute_data_with_mode_AtoT_acc_wrong_int(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": 800, "TtoA": 700},
            eval_metric="accuracy"
        )
        with pytest.raises(ValueError, match="must be between 1 and 100 for int type"):
            dpa.permuteData(D, mode="AtoT")

    def test_permute_data_with_mode_AtoT_acc_correct_tensor(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": torch.tensor(0.8), "TtoA": torch.tensor(0.7)},
            eval_metric="accuracy"
        )
        perturbed_D = dpa.permuteData(D, mode="AtoT")
        # check shape is preserved
        matches = torch.sum(perturbed_D == D)/D.shape[0]
        # check approximate match with model accuracy
        assert abs(dpa.model_acc["AtoT"] - matches) < 0.1

    def test_permute_data_with_mode_AtoT_acc_wrong_tensor(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": torch.tensor(8.5), "TtoA": torch.tensor(7.5)},
            eval_metric="accuracy"
        )
        with pytest.raises(ValueError, match="tensor value must be between 0.0 and 1.0"):
            dpa.permuteData(D, mode="AtoT")

    def test_permute_data_with_mode_TtoA_acc_correct_float(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": 0.8, "TtoA": 0.7},
            eval_metric="accuracy"
        )
        perturbed_D = dpa.permuteData(D, mode="TtoA")
        # check shape is preserved
        matches = torch.sum(perturbed_D == D)/D.shape[0]
        # check approximate match with model accuracy
        assert abs(dpa.model_acc["TtoA"] - matches) < 0.1

    def test_permute_data_with_mode_TtoA_acc_wrong_float(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": 8.5, "TtoA": 7.5},
            eval_metric="accuracy"
        )
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0 for float type"):
            dpa.permuteData(D, mode="TtoA")

    def test_permute_data_with_mode_TtoA_acc_correct_int(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": 70, "TtoA": 80},
            eval_metric="accuracy"
        )
        perturbed_D = dpa.permuteData(D, mode="TtoA")
        # check shape is preserved
        matches = torch.sum(perturbed_D == D)/D.shape[0]
        # check approximate match with model accuracy
        assert abs(dpa.model_acc["TtoA"] - matches) < 0.1
    
    def test_permute_data_with_mode_TtoA_acc_wrong_int(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": 800, "TtoA": 700},
            eval_metric="accuracy"
        )
        with pytest.raises(ValueError, match="must be between 1 and 100 for int type"):
            dpa.permuteData(D, mode="TtoA")


    def test_permute_data_with_mode_TtoA_acc_correct_tensor(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": torch.tensor(0.8), "TtoA": torch.tensor(0.7)},
            eval_metric="accuracy"
        )
        perturbed_D = dpa.permuteData(D, mode="TtoA")
        # check shape is preserved
        matches = torch.sum(perturbed_D == D)/D.shape[0]
        # check approximate match with model accuracy
        assert abs(dpa.model_acc["TtoA"] - matches) < 0.1

    def test_permute_data_with_mode_TtoA_acc_wrong_tensor(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": torch.tensor(8.5), "TtoA": torch.tensor(7.5)},
            eval_metric="accuracy"
        )
        with pytest.raises(ValueError, match="tensor value must be between 0.0 and 1.0"):
            dpa.permuteData(D, mode="TtoA")

    def test_permute_data_with_mode_AtoT_invalid_type(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": "invalid_type", "TtoA": "invalid_type"},
            eval_metric="accuracy"
        )
        with pytest.raises(ValueError, match="must be float, int, or torch.Tensor"):
            dpa.permuteData(D, mode="AtoT")

    def test_permute_data_with_mode_TtoA_invalid_type(self):
        D = get_test_data()["D"]
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc={"AtoT": "invalid_type", "TtoA": "invalid_type"},
            eval_metric="accuracy"
        )
        with pytest.raises(ValueError, match="must be float, int, or torch.Tensor"):
            dpa.permuteData(D, mode="TtoA")

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

    def test_initialize_eval_metrics_with_valid_string_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy"
        )    
        assert callable(dpa.eval_metric)

    def test_initialize_eval_metrics_with_invalid_string(self):
        with pytest.raises(ValueError, match="unavailable"):
            Leakage(
                attacker_model=attacker_model,
                train_params=train_params,
                model_acc=0.8,
                eval_metric="unavailable_metric"
            )

    def test_initialize_eval_metrics_with_invalid_string_dpa(self):
        with pytest.raises(ValueError, match="unavailable"):
            DPA(
                attacker_AtoT=attacker_model,
                attacker_TtoA=attacker_model,
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

    def test_initialize_eval_metrics_with_valid_callable_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric=ModifiedBCELoss
        )
        assert dpa.eval_metric == ModifiedBCELoss

    def test_initialize_eval_metrics_with_invalid_metric(self):
        with pytest.raises(ValueError, match="Invalid Metric"):
            Leakage(
                attacker_model=attacker_model,
                train_params=train_params,
                model_acc=0.8,
                eval_metric=123
            )

    def test_initialize_eval_metrics_with_invalid_metric_dpa(self):
        with pytest.raises(ValueError, match="Invalid Metric"):
            DPA(
                attacker_AtoT=attacker_model,
                attacker_TtoA=attacker_model,
                train_params=train_params,
                model_acc=0.8,
                eval_metric=123
            )

#===============================================================================
# Test Leakage Calculation Methods
#===============================================================================
class TestModelDefinition:
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

    def test_dpa_leakage_model_definition(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy"
        )
        # testing that the models are defined for AtoT
        assert isinstance(dpa.attacker_D_AtoT, torch.nn.Module)
        assert isinstance(dpa.attacker_M_AtoT, torch.nn.Module)
        # testing that the models are defined for TtoA
        assert isinstance(dpa.attacker_D_TtoA, torch.nn.Module)
        assert isinstance(dpa.attacker_M_TtoA, torch.nn.Module)
        # testing that the models are not the same object but deep copies
        assert dpa.attacker_D_AtoT is not dpa.attacker_M_AtoT
        assert dpa.attacker_D_TtoA is not dpa.attacker_M_TtoA



class TestLambdaCalculation:
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

    def test_calc_lambda_for_eval_acc_with_threshold_true_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.2], [0.8]], dtype=torch.float)
        lambda_value = dpa.calcLambda(mock_model, x, y)
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

    def test_calc_lambda_for_eval_acc_with_threshold_false_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy",
            threshold=False
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.2], [0.8]], dtype=torch.float)
        lambda_value = dpa.calcLambda(mock_model, x, y)
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

    def test_calc_lambda_for_eval_acc_with_all_correct_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.8], [0.2]], dtype=torch.float)
        lambda_value = dpa.calcLambda(mock_model, x, y)
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

    def test_calc_lambda_for_eval_acc_with_all_wrong_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="accuracy"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.7], [0.3], [0.2], [0.8]], dtype=torch.float)
        lambda_value = dpa.calcLambda(mock_model, x, y)
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


    def test_calc_lambda_for_eval_mse_with_threshold_true_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="mse"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.2], [0.8]], dtype=torch.float)
        lambda_value = dpa.calcLambda(mock_model, x, y)
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

    def test_calc_lambda_for_eval_mse_with_threshold_false_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="mse",
            threshold=False
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.2], [0.8]], dtype=torch.float)
        lambda_value = dpa.calcLambda(mock_model, x, y)
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


    def test_calc_lambda_for_eval_mse_with_all_correct_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="mse"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.8], [0.2]], dtype=torch.float)
        lambda_value = dpa.calcLambda(mock_model, x, y)
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

    def test_calc_lambda_for_eval_mse_with_all_wrong_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="mse"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.7], [0.3], [0.2], [0.8]], dtype=torch.float)
        lambda_value = dpa.calcLambda(mock_model, x, y)
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

    def test_calc_lambda_for_eval_bce_with_threshold_true_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="bce"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.2], [0.8]], dtype=torch.float)
        lambda_value = dpa.calcLambda(mock_model, x, y)
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
        assert lambda_value.item() >= 0.0


    def test_calc_lambda_for_eval_bce_with_threshold_false_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="bce",
            threshold=False
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.2], [0.8]], dtype=torch.float)
        lambda_value = dpa.calcLambda(mock_model, x, y)
        assert lambda_value.item() >= 0.0


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

    def test_calc_lambda_for_eval_bce_with_all_correct_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="bce"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.3], [0.7], [0.8], [0.2]], dtype=torch.float)
        lambda_value = dpa.calcLambda(mock_model, x, y)
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

    
    def test_calc_lambda_for_eval_bce_with_all_wrong_dpa(self):
        dpa = DPA(
            attacker_AtoT=attacker_model,
            attacker_TtoA=attacker_model,
            train_params=train_params,
            model_acc=0.8,
            eval_metric="bce"
        )
        mock_model = Mock(spec=torch.nn.Module)
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float)
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
        mock_model.return_value = torch.tensor([[0.7], [0.3], [0.2], [0.8]], dtype=torch.float)
        lambda_value = dpa.calcLambda(mock_model, x, y)
        assert lambda_value.item() < 0.1



class TestLeakageCalculation:
    #===============================================================================
    # Compute Leakage Test
    #===============================================================================
    def test_comp_leakage(self):
       leak = leakage.compute_leakage(lambda_d=torch.tensor(4.0), lambda_m=torch.tensor(12.0))
       assert leak.item() > 1.0

    def test_comp_leakage_dpa(self):
       leak = dpa_1.compute_leakage(lambda_d=torch.tensor(4.0), lambda_m=torch.tensor(12.0), normalized=dpa_1.normalized)
       assert 0.0 <= leak.item() <= 1.0

    #===============================================================================
    # Calculate Leakage Test On Unbiased Data and Unbiased Predictions
    #===============================================================================
    def test_calc_leakage_on_unbiased_data(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        leak = leakage.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode=None)
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_unbiased_data_acc_float_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        leak = dpa_1.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="AtoT")
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_unbiased_data_acc_float_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        leak = dpa_1.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="TtoA")
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_unbiased_data_acc_dict_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa.split(feat, data, pred, test_size=test_size)
        leak = dpa.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="AtoT")
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_unbiased_data_acc_dict_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa.split(feat, data, pred, test_size=test_size)
        leak = dpa.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="TtoA")
        assert -1.0 <= leak.item() <= 1.0

    #===============================================================================
    # Calculate Leakage Test On Biased Data and Unbiased Predictions
    #===============================================================================
    def test_calc_leakage_on_biased_data_unbiased_pred(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        leak = leakage.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode=None)
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_biased_data_unbiased_pred_acc_float_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        leak = dpa_1.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="AtoT")
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_biased_data_unbiased_pred_acc_float_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        leak = dpa_1.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="TtoA")
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_biased_data_unbiased_pred_acc_dict_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa.split(feat, data, pred, test_size=test_size)
        leak = dpa.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="AtoT")
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_biased_data_unbiased_pred_acc_dict_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa.split(feat, data, pred, test_size=test_size)
        leak = dpa.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="TtoA")
        assert -1.0 <= leak.item() <= 1.0

    #===============================================================================
    # Calculate Leakage Test On Unbiased Data and Biased Predictions
    #===============================================================================
    def test_calc_leakage_on_unbiased_data_biased_pred(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        leak = leakage.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode=None)
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_unbiased_data_biased_pred_acc_float_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        leak = dpa_1.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="AtoT")
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_unbiased_data_biased_pred_acc_float_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        leak = dpa_1.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="TtoA")
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_unbiased_data_biased_pred_acc_dict_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa.split(feat, data, pred, test_size=test_size)
        leak = dpa.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="AtoT")
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_unbiased_data_biased_pred_acc_dict_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa.split(feat, data, pred, test_size=test_size)
        leak = dpa.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="TtoA")
        assert -1.0 <= leak.item() <= 1.0

    #===============================================================================
    # Calculate Leakage Test On Biased Data and Biased Predictions
    #===============================================================================
    def test_calc_leakage_on_biased_data_biased_pred(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        leak = leakage.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode=None)
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_biased_data_biased_pred_acc_float_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        leak = dpa_1.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="AtoT")
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_biased_data_biased_pred_acc_float_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        leak = dpa_1.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="TtoA")
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_biased_data_biased_pred_acc_dict_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa.split(feat, data, pred, test_size=test_size)
        leak = dpa.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="AtoT")
        assert -1.0 <= leak.item() <= 1.0

    def test_calc_dpa_leakage_on_biased_data_biased_pred_acc_dict_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa.split(feat, data, pred, test_size=test_size)
        leak = dpa.calcLeak(feat_train, data_train, pred_train, feat_test, data_test, pred_test, mode="TtoA")
        assert -1.0 <= leak.item() <= 1.0

class TestAmortisedLeakageCalculation:
    #===============================================================================
    # Amortized Leakage Test on unbiased data and unbiased predictions
    # With train and test data
    # With default method
    #===============================================================================
    def test_amortized_leakage_on_unbiased_data_unbiased_pred_with_train_and_test_default_method(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.45), torch.tensor(0.8)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(feat_train=feat_train, data_train=data_train, pred_train=pred_train, feat_test=feat_test, data_test=data_test, pred_test=pred_test, num_trials=num_trials)
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_unbiased_pred_with_train_and_test_default_method_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.55), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials,
                mode="AtoT"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_unbiased_pred_with_train_and_test_default_method_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.55), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials, 
                mode="TtoA"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on unbiased data and unbiased predictions
    # With train and test data
    # With median method
    #===============================================================================
    def test_amortized_leakage_on_unbiased_data_unbiased_pred_with_train_and_test_method_median(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials,
                method=method
                )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_unbiased_pred_with_train_and_test_method_median_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.54), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials,
                mode="AtoT",
                method=method
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_unbiased_pred_with_train_and_test_method_median_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials, 
                mode="TtoA",
                method=method
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on unbiased data and unbiased predictions
    # Without train and test data
    # With default method
    #===============================================================================
    def test_amortized_leakage_on_unbiased_data_unbiased_pred_default_method(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.53), torch.tensor(0.6)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(feat_train=feat, data_train=data, pred_train=pred, num_trials=num_trials)
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_unbiased_pred_default_method_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.59), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials,
                mode="AtoT"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_unbiased_pred_default_method_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials, 
                mode="TtoA"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on unbiased data and unbiased predictions
    # Without train and test data
    # With median method
    #===============================================================================
    def test_amortized_leakage_on_unbiased_data_unbiased_pred_method_median(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials,
                method=method
                )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_unbiased_pred_method_median_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.54), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials,
                method=method,
                mode="AtoT"
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_unbiased_pred_method_median_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials, 
                method=method,
                mode="TtoA"
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on biased data and unbiased predictions
    # With train and test data
    # With default method
    #===============================================================================
    def test_amortized_leakage_on_biased_data_unbiased_pred_with_train_and_test_default_method(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.45), torch.tensor(0.8)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(feat_train=feat_train, data_train=data_train, pred_train=pred_train, feat_test=feat_test, data_test=data_test, pred_test=pred_test, num_trials=num_trials)
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_unbiased_pred_with_train_and_test_default_method_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.55), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials,
                mode="AtoT"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_unbiased_pred_with_train_and_test_default_method_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.55), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials, 
                mode="TtoA"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on biased data and unbiased predictions
    # With train and test data
    # With median method
    #===============================================================================
    def test_amortized_leakage_on_biased_data_unbiased_pred_with_train_and_test_method_median(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials,
                method=method
                )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_unbiased_pred_with_train_and_test_method_median_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.54), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials,
                mode="AtoT",
                method=method
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_unbiased_pred_with_train_and_test_method_median_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials, 
                mode="TtoA",
                method=method
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on biased data and unbiased predictions
    # Without train and test data
    # With default method
    #===============================================================================
    def test_amortized_leakage_on_biased_data_unbiased_pred_default_method(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.53), torch.tensor(0.6)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(feat_train=feat, data_train=data, pred_train=pred, num_trials=num_trials)
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_unbiased_pred_default_method_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.59), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials,
                mode="AtoT"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_unbiased_pred_default_method_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials, 
                mode="TtoA"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on biased data and unbiased predictions
    # Without train and test data
    # With median method
    #===============================================================================
    def test_amortized_leakage_on_biased_data_unbiased_pred_method_median(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials,
                method=method
                )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_unbiased_pred_method_median_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.54), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials,
                method=method,
                mode="AtoT"
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_unbiased_pred_method_median_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M1"]
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials, 
                method=method,
                mode="TtoA"
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on unbiased data and biased predictions
    # With train and test data
    # With default method
    #===============================================================================
    def test_amortized_leakage_on_unbiased_data_biased_pred_with_train_and_test_default_method(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.45), torch.tensor(0.8)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(feat_train=feat_train, data_train=data_train, pred_train=pred_train, feat_test=feat_test, data_test=data_test, pred_test=pred_test, num_trials=num_trials)
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_biased_pred_with_train_and_test_default_method_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.55), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials,
                mode="AtoT"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_biased_pred_with_train_and_test_default_method_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.55), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials, 
                mode="TtoA"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on unbiased data and biased predictions
    # With train and test data
    # With median method
    #===============================================================================
    def test_amortized_leakage_on_unbiased_data_biased_pred_with_train_and_test_method_median(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials,
                method=method
                )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_biased_pred_with_train_and_test_method_median_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.54), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials,
                mode="AtoT",
                method=method
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_biased_pred_with_train_and_test_method_median_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials, 
                mode="TtoA",
                method=method
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on unbiased data and biased predictions
    # Without train and test data
    # With default method
    #===============================================================================
    def test_amortized_leakage_on_unbiased_data_biased_pred_default_method(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.53), torch.tensor(0.6)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(feat_train=feat, data_train=data, pred_train=pred, num_trials=num_trials)
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_biased_pred_default_method_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.59), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials,
                mode="AtoT"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_biased_pred_default_method_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials, 
                mode="TtoA"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on unbiased data and biased predictions
    # Without train and test data
    # With median method
    #===============================================================================
    def test_amortized_leakage_on_unbiased_data_biased_pred_method_median(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials,
                method=method
                )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_biased_pred_method_median_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.54), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials,
                method=method,
                mode="AtoT"
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_unbiased_data_biased_pred_method_median_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M2"]
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials, 
                method=method,
                mode="TtoA"
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on biased data and biased predictions
    # With train and test data
    # With default method
    #===============================================================================
    def test_amortized_leakage_on_biased_data_biased_pred_with_train_and_test_default_method(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.45), torch.tensor(0.8)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(feat_train=feat_train, data_train=data_train, pred_train=pred_train, feat_test=feat_test, data_test=data_test, pred_test=pred_test, num_trials=num_trials)
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_biased_pred_with_train_and_test_default_method_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.55), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials,
                mode="AtoT"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_biased_pred_with_train_and_test_default_method_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.55), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials, 
                mode="TtoA"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on biased data and biased predictions
    # With train and test data
    # With median method
    #===============================================================================
    def test_amortized_leakage_on_biased_data_biased_pred_with_train_and_test_method_median(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = leakage.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials,
                method=method
                )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_biased_pred_with_train_and_test_method_median_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.54), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials,
                mode="AtoT",
                method=method
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_biased_pred_with_train_and_test_method_median_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        test_size = 0.2
        feat_train, feat_test, data_train, data_test, pred_train, pred_test = dpa_1.split(feat, data, pred, test_size=test_size)
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat_train, 
                data_train=data_train, 
                pred_train=pred_train, 
                feat_test=feat_test, 
                data_test=data_test, 
                pred_test=pred_test, 
                num_trials=num_trials, 
                mode="TtoA",
                method=method
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on biased data and biased predictions
    # Without train and test data
    # With default method
    #===============================================================================
    def test_amortized_leakage_on_biased_data_biased_pred_default_method(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.53), torch.tensor(0.6)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(feat_train=feat, data_train=data, pred_train=pred, num_trials=num_trials)
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_biased_pred_default_method_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.59), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials,
                mode="AtoT"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_biased_pred_default_method_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        num_trials = 3
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials, 
                mode="TtoA"
            )
        assert isinstance(amortized_leakage, str)
        mean, std = amortized_leakage.split(" ± ")
        assert float(mean) == round(torch.mean(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test on biased data and biased predictions
    # Without train and test data
    # With median method
    #===============================================================================
    def test_amortized_leakage_on_biased_data_biased_pred_method_median(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.Leakage.calcLeak", side_effect=mock_values):
            amortized_leakage = leakage.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials,
                method=method
                )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_biased_pred_method_median_dpa_AtoT(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.54), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials,
                method=method,
                mode="AtoT"
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    def test_amortized_leakage_on_biased_data_biased_pred_method_median_dpa_TtoA(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D_bias"]
        pred = get_test_data()["M2"]
        num_trials = 3
        method = "median"
        mock_values = [torch.tensor(0.4), torch.tensor(0.57), torch.tensor(0.6)]
        with patch("PredMetrics_v1.DPA.calcLeak", side_effect=mock_values):
            amortized_leakage = dpa_1.getAmortizedLeakage(
                feat_train=feat, 
                data_train=data, 
                pred_train=pred, 
                num_trials=num_trials, 
                method=method,
                mode="TtoA"
            )
        assert isinstance(amortized_leakage, str)
        median, std = amortized_leakage.split(" ± ")
        assert float(median) == round(torch.median(torch.tensor(mock_values)).item(), 4)
        assert float(std) == round(torch.std(torch.tensor(mock_values)).item(), 4)

    #===============================================================================
    # Amortized Leakage Test EndToEnd
    #===============================================================================
    def test_amortized_leakage_end_to_end(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        result = leakage.getAmortizedLeakage(feat, data, pred, num_trials=3, method="mean")
        assert isinstance(result, str)
        mean, std = result.split(" ± ")
        mean = float(mean)
        std = float(std)
        assert -1.0 <= mean <= 1.0
        assert std >= 0.0

    def test_amortized_dpa_AtoT_end_to_end(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        result = dpa.getAmortizedLeakage(feat, data, pred, mode="AtoT", num_trials=3, method="mean")
        assert isinstance(result, str)
        mean, std = result.split(" ± ")
        mean = float(mean)
        std = float(std)
        assert -1.0 <= mean <= 1.0
        assert std >= 0.0

    def test_amortized_dpa_TtoA_end_to_end(self):
        feat = get_test_data()["P"]
        data = get_test_data()["D"]
        pred = get_test_data()["M1"]
        result = dpa.getAmortizedLeakage(feat, data, pred, mode="TtoA", num_trials=3, method="mean")
        assert isinstance(result, str)
        mean, std = result.split(" ± ")
        mean = float(mean)
        std = float(std)
        assert -1.0 <= mean <= 1.0
        assert std >= 0.0

class TestTrainingMethods:
    def test_train_setup_leakage_with_loss_func(self):
        loss_funcs = ["mse", "cross-entropy", "bce"]
        for loss_func in loss_funcs:
            train_params = {
                "learning_rate": 0.01,
                "loss_function": loss_func,
                "epochs": 5,
                "batch_size": 32
            }
            leakage = Leakage(
                attacker_model=attacker_model,
                train_params=train_params,
                model_acc=0.8,
                eval_metric="accuracy"
            )
            for mode in ["D", "M"]:
                model, optimizer, criterion = leakage.train_setup(attacker_mode=mode)
                assert isinstance(model, torch.nn.Module)
                assert isinstance(optimizer, torch.optim.Optimizer)
                assert isinstance(criterion, torch.nn.Module)

    def test_train_setup_dpa_with_loss_func(self):
        loss_funcs = ["mse", "cross-entropy", "bce"]
        for loss_func in loss_funcs:
            train_params = {
                "learning_rate": 0.01,
                "loss_function": loss_func,
                "epochs": 5,
                "batch_size": 32
            }
            dpa = DPA(
                attacker_AtoT=attacker_model,
                attacker_TtoA=attacker_model,
                train_params=train_params,
                model_acc=0.8,
                eval_metric="accuracy"
            )
            for mode in ["D_AtoT", "M_AtoT", "D_TtoA", "M_TtoA"]:
                model, optimizer, criterion = dpa.train_setup(attacker_mode=mode)
                assert isinstance(model, torch.nn.Module)
                assert isinstance(optimizer, torch.optim.Optimizer)
                assert isinstance(criterion, torch.nn.Module)
            

    def test_train_batch_with_loss_returns_valid_output_leakage(self):
        x = torch.randn(16, 1)
        y = torch.randint(0, 2, (16, 1)).float()
        for mode in ["D", "M"]:
            model, optimizer, criterion = leakage.train_setup(attacker_mode=mode)
            loss = leakage.train_batch_with_loss(model, optimizer, criterion, x, y)
            assert isinstance(loss, float)
            assert loss >= 0.0

    def test_train_batch_with_loss_returns_valid_output_dpa(self):
        x = torch.randn(16, 1)
        y = torch.randint(0, 2, (16, 1)).float()
        for mode in ["D_AtoT", "M_AtoT", "D_TtoA", "M_TtoA"]:
            model, optimizer, criterion = dpa.train_setup(attacker_mode=mode)
            loss = dpa.train_batch_with_loss(model, optimizer, criterion, x, y)
            assert isinstance(loss, float)
            assert loss >= 0.0

    def test_train_epochs_returns_valid_output_leakage(self):
        x = torch.randn(16, 1)
        y = torch.randint(0, 2, (16, 1)).float()
        for mode in ["D", "M"]:
            model, optimizer, criterion = leakage.train_setup(attacker_mode=mode)
            loss = leakage.train_epochs(model, optimizer, criterion, x, y)
            assert isinstance(loss, float)
            assert loss >= 0.0

    def test_train_epochs_returns_valid_output_dpa(self):
        x = torch.randn(16, 1)
        y = torch.randint(0, 2, (16, 1)).float()
        for mode in ["D_AtoT", "M_AtoT", "D_TtoA", "M_TtoA"]:
            model, optimizer, criterion = dpa.train_setup(attacker_mode=mode)
            loss = dpa.train_epochs(model, optimizer, criterion, x, y)
            assert isinstance(loss, float)
            assert loss >= 0.0

    def test_train_epochs_process_all_batches(self):
        x = torch.randn(16, 1)
        y = torch.randint(0, 2, (16, 1)).float()
        call_count = 0
        orig_train_batch = dpa.train_batch_with_loss
        model, optimizer, criterion = dpa.train_setup(attacker_mode="D_AtoT")
        def mock_train_batch(model, optimizer, criterion, x_batch, y_batch):
            nonlocal call_count
            call_count += 1
            return orig_train_batch(model, optimizer, criterion, x_batch, y_batch)
        with patch.object(dpa, 'train_batch_with_loss', side_effect=mock_train_batch):
            avg_loss = dpa.train_epochs(model, optimizer, criterion, x, y)
        expected_batches = math.ceil(len(x) / dpa.train_params["batch_size"])
        assert call_count == expected_batches
        
    def test_train_creates_correct_models_with_different_modes_leakage(self):
        x = get_test_data()["P"]
        y = get_test_data()["D"]
        for mode in ["D", "M"]:
            leakage.train(x, y, mode)
            assert hasattr(leakage, "attacker_" + mode)
        for mode in ["D", "M"]:
            model = getattr(leakage, "attacker_" + mode)
            model.eval()
            with torch.no_grad():
                y_pred = model(x)
                assert isinstance(y_pred, torch.Tensor)
                assert y_pred.shape == (len(x), 1)


    def test_train_creates_correct_models_with_different_modes_dpa(self):
        x = get_test_data()["P"]
        y = get_test_data()["D"]
        for mode in ["D_AtoT", "M_AtoT", "D_TtoA", "M_TtoA"]:
            dpa.train(x, y, mode)
            assert hasattr(dpa, "attacker_" + mode)
        for mode in ["D_AtoT", "M_AtoT", "D_TtoA", "M_TtoA"]:
            model = getattr(dpa, "attacker_" + mode)
            model.eval()
            with torch.no_grad():
                y_pred = model(x)
                assert isinstance(y_pred, torch.Tensor)
                assert y_pred.shape == (len(x), 1)