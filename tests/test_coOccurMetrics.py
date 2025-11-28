"""
Pytest test cases for CoOccurMetrics module.

Tests cover:
- BaseCoOccurMetric probability computation methods
- BA_Zhao bias amplification metric
- DBA (Differential Bias Amplification) metric
- MDBA (Multi-Dimensional Bias Amplification) metric
"""

from cgi import test
from codecs import ascii_encode
import torch
import pytest
from bias_amplification.metrics.CoOccurMetrics_v1 import (
    BaseCoOccurMetric,
    BA_Zhao,
    DBA,
    MDBA
)
from utils.datacreator import dataCreator
#===============================================================================
# Reusable test dataset
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

def test_data():
    return get_test_data()

def test_metrics():
    return {
        "BA_Zhao": BA_Zhao(),
        "DBA": DBA(),
        "MDBA": MDBA()
    }

def simple_binary_data():
    """Create simple binary test data."""
    # 10 observations, 2 attributes, 2 tasks
    A = torch.tensor([
        [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],  # First 5: attribute 0
        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],  # Last 5: attribute 1
    ], dtype=torch.float)
    
    T = torch.tensor([
        [1, 0], [1, 0], [0, 1], [0, 1], [0, 1],  # Mixed for first group
        [0, 1], [0, 1], [1, 0], [1, 0], [1, 0],  # Mixed for second group
    ], dtype=torch.float)
    
    return A, T

def independent_data():
    """Create independent attribute-task data."""
    # Attributes and tasks are independent
    A = torch.tensor([
        [1, 0], [1, 0], [1, 0], [1, 0],
        [0, 1], [0, 1], [0, 1], [0, 1],
    ], dtype=torch.float)
    
    T = torch.tensor([
        [1, 0], [1, 0], [0, 1], [0, 1],
        [1, 0], [1, 0], [0, 1], [0, 1],
    ], dtype=torch.float)
    
    return A, T


def correlated_data():
    """Create correlated attribute-task data."""
    # Strong correlation: A[0] -> T[0], A[1] -> T[1]
    A = torch.tensor([
        [1, 0], [1, 0], [1, 0], [1, 0],
        [0, 1], [0, 1], [0, 1], [0, 1],
    ], dtype=torch.float)
    
    T = torch.tensor([
        [1, 0], [1, 0], [1, 0], [1, 0],
        [0, 1], [0, 1], [0, 1], [0, 1],
    ], dtype=torch.float)
    
    return A, T


def prediction_data(simple_binary_data):
    """Create prediction data that differs from ground truth."""
    A, T = simple_binary_data
    # Predictions are slightly different
    T_pred = torch.tensor([
        [1, 0], [0, 1], [1, 0], [0, 1], [0, 1],
        [1, 0], [0, 1], [1, 0], [1, 0], [0, 1],
    ], dtype=torch.float)
    return A, T, T_pred

def negatively_correlated_data():
    """
    Creating negatively correlated attribute-task data.
    A[0] is negatively correlated with T[0] (when A[0]=1, T[0]=0)
    A[1] is negatively correlated with T[1] (when A[1]=1, T[1]=0)
    """
    A = torch.tensor([
        [1, 0], [1, 0], [1, 0], [1, 0],  # A[0]=1
        [0, 1], [0, 1], [0, 1], [0, 1],  # A[1]=1
    ], dtype=torch.float)
    
    T = torch.tensor([
        [0, 1], [0, 1], [0, 1], [0, 1],  # When A[0]=1, T[0]=0 (negative correlation)
        [1, 0], [1, 0], [1, 0], [1, 0],  # When A[1]=1, T[1]=0 (negative correlation)
    ], dtype=torch.float)
    
    return A, T

def mixed_correlation_data():
    """
    Creating data with both positive and negative correlations.
    A[0] positively correlated with T[0]
    A[1] negatively correlated with T[0]
    """
    A = torch.tensor([
        [1, 0], [1, 0], [1, 0], [1, 0],  # A[0]=1
        [0, 1], [0, 1], [0, 1], [0, 1],  # A[1]=1
    ], dtype=torch.float)
    
    T = torch.tensor([
        [1, 0], [1, 0], [1, 0], [1, 0],  # A[0] -> T[0] (positive)
        [0, 1], [0, 1], [0, 1], [0, 1],  # A[1] -> T[1] but T[0]=0 (negative for T[0])
    ], dtype=torch.float)
    
    return A, T

def multi_attribute_data():
    """Create data with multiple attributes for MDBA testing."""
    # 20 observations, 2 groups, 3 attributes
    A = torch.tensor([
        [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
        [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
    ], dtype=torch.float)
    
    T = torch.tensor([
        [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1],
        [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0],
        [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1],
        [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0],
    ], dtype=torch.float)
    
    return A, 
    


#===============================================================================
# Base ClassTest Cases
#===============================================================================

class TestBaseCoOccurMetric:
    def test_computePairProbs_shape(self):
        data = test_data()
        metrics = test_metrics()
        A = data["P"]
        T = data["D"]
        metric_dba = metrics["DBA"]
        probs = metric_dba.computePairProbs(A, T)
        assert probs is not None
        assert probs.shape == (A.shape[1], T.shape[1])
        assert probs.dtype == torch.float32

    def test_computePairProbs_values(self):
        data = test_data()
        metrics = test_metrics()
        A = data["P"]
        T = data["D"]
        metric_dba = metrics["DBA"]
        probs = metric_dba.computePairProbs(A, T)
        # all values should be between 0 and 1
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)

    def test_computeProbs_shape(self):
        data = test_data()
        metrics = test_metrics()
        A = data["P"]
        metric_dba = metrics["DBA"]
        probs = metric_dba.computeProbs(A)
        assert probs is not None
        assert probs.shape == (A.shape[1],)
        assert probs.dtype == torch.float32

    def test_computeProbs_values(self):
        data = test_data()
        metrics = test_metrics()
        A = data["P"]
        metric_dba = metrics["DBA"]
        probs = metric_dba.computeProbs(A)
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)

    def test_computeAgivenT_shape(self):
        data = test_data()
        metrics = test_metrics()
        A = data["P"]
        T = data["D"]
        metric_dba = metrics["DBA"]
        probs = metric_dba.computeAgivenT(A, T)
        assert probs is not None
        assert probs.shape == (A.shape[1], T.shape[1])
        assert probs.dtype == torch.float32

    def test_computeAgivenT_values(self):
        data = test_data()
        metrics = test_metrics()
        A = data["P"]
        T = data["D"]
        metric_dba = metrics["DBA"]
        probs = metric_dba.computeAgivenT(A, T)
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)

    def test_compute_conditional_handles_zero_division(self):
        # Create data where some categories don't occur
        A = torch.tensor([[1, 0], [1, 0], [1, 0]], dtype=torch.float)
        T = torch.tensor([[1, 0], [1, 0], [1, 0]], dtype=torch.float)
        metrics = test_metrics()
        metric_dba = metrics["DBA"]
        # Should not raise error due to clamp(min=1e-10)
        result = metric_dba.computeAgivenT(A, T)
        assert torch.all(torch.isfinite(result))
        result = metric_dba.computeTgivenA(A, T)
        assert torch.all(torch.isfinite(result))

# ============================================================================
# BA_ZHAO TESTS
# ============================================================================

class TestBA_Zhao:
    #===========================================================================
    # TESTING BASIC FUNCTIONALITY
    #===========================================================================
    def test_check_bias_shape(self):
        A, T = simple_binary_data()
        metrics = test_metrics()
        metric_ba_zhao = metrics["BA_Zhao"]
        bias = metric_ba_zhao.check_bias(A, T)
        assert bias is not None
        assert bias.shape == (A.shape[1], T.shape[1])
        assert bias.dtype == torch.float32

    def test_check_bias_uniform_distribution(self):
        A, T = independent_data()
        metrics = test_metrics()
        metric_ba_zhao = metrics["BA_Zhao"]
        bias = metric_ba_zhao.check_bias(A, T)
        assert isinstance(bias, torch.Tensor)
        assert bias.shape == (A.shape[1], T.shape[1])

    def test_check_bias_correlated_data(self):
        """Test bias check with correlated data."""
        A, T = correlated_data()
        metrics = test_metrics()
        metric_ba_zhao = metrics["BA_Zhao"]
        result = metric_ba_zhao.check_bias(A, T)
        assert isinstance(result, torch.Tensor)
        # At least some pairs should be biased
        assert result.sum() > 0

    def test_computeBiasAmp_shape(self):
        A, T, T_pred = prediction_data(simple_binary_data())
        metrics = test_metrics()
        metric_ba_zhao = metrics["BA_Zhao"]
        bias_amp_combined, bias_amp = metric_ba_zhao.computeBiasAmp(A, T, T_pred)
        assert isinstance(bias_amp_combined, torch.Tensor)
        assert bias_amp_combined.shape == ()
        assert bias_amp_combined.dtype == torch.float32
        assert isinstance(bias_amp, torch.Tensor)
        assert bias_amp.shape == (A.shape[1], T.shape[1])
        assert bias_amp.dtype == torch.float32

    def test_computeBiasAmp_consistency(self):
        """Test that computeBiasAmp is consistent across calls."""
        A, T = simple_binary_data()
        T_pred = T.clone()
        metrics = test_metrics()
        metric_ba_zhao = metrics["BA_Zhao"]
        bias_amp_combined_1, bias_amp_1 = metric_ba_zhao.computeBiasAmp(A, T, T_pred)
        bias_amp_combined_2, bias_amp_2 = metric_ba_zhao.computeBiasAmp(A, T, T_pred)
        assert torch.allclose(bias_amp_1, bias_amp_2)
        assert torch.allclose(bias_amp_combined_1, bias_amp_combined_2)

    #===========================================================================
    # TESTING BA_ZHAO SHORTCOMINGS
    #===========================================================================
    def test_shortcoming1_ignores_negative_correlations(self):
        pass
    
# ============================================================================
# DBA TESTS
# ============================================================================

class TestDBA:
    pass