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
    
    return A, T

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
        A1_T0 = torch.zeros((10, 1), dtype=torch.float)
        A1_T1 = torch.ones((40, 1), dtype=torch.float)
        A1_data = torch.cat([A1_T0, A1_T1], dim=0)
        
        # A2 = 1: 40 T=0, 10 T=1
        A2_T0 = torch.zeros((40, 1), dtype=torch.float)
        A2_T1 = torch.ones((10, 1), dtype=torch.float)
        A2_data = torch.cat([A2_T0, A2_T1], dim=0)
        
        # A3 = 1: 10 T=0, 20 T=1
        A3_T0 = torch.zeros((10, 1), dtype=torch.float)
        A3_T1 = torch.ones((20, 1), dtype=torch.float)
        A3_data = torch.cat([A3_T0, A3_T1], dim=0)
        
        # Combine all groups
        A = torch.cat([
            torch.tensor([[1, 0, 0]] * 50, dtype=torch.float),  # A1
            torch.tensor([[0, 1, 0]] * 50, dtype=torch.float),  # A2
            torch.tensor([[0, 0, 1]] * 30, dtype=torch.float),  # A3
        ], dim=0)
        
        T = torch.cat([
            torch.cat([A1_T0, A1_T1], dim=0),  # A1: 10 T=0, 40 T=1
            torch.cat([A2_T0, A2_T1], dim=0),  # A2: 40 T=0, 10 T=1
            torch.cat([A3_T0, A3_T1], dim=0),  # A3: 10 T=0, 20 T=1
        ], dim=0)
        metrics = test_metrics()
        metric_ba_zhao = metrics["BA_Zhao"]
        bias_mask = metric_ba_zhao.check_bias(A, T)
        P_A_given_T = metric_ba_zhao.computeAgivenT(A, T)
        num_A = A.shape[1]
        threshold = 1.0 / num_A
        assert bias_mask[0, 0] == 1.0, "A1 should be flagged (positive correlation)"
        assert bias_mask[1, 0] == 0.0, "A2 should NOT be flagged (negative correlation)"
        assert bias_mask[2, 0] == 0.0, "A3 should NOT be flagged (weak correlation)"
        # Create model predictions that amplify bias in A2 and A3
        # Model: correct on A1, wrong on A2 (predicts T=0), wrong on A3 (predicts T=1)
        T_pred = T.clone()
        # For A2 group: predict T=0 (amplifies negative correlation)
        A2_indices = (A[:, 1] == 1)
        T_pred[A2_indices] = 0.0
        # For A3 group: predict T=1 (amplifies positive correlation)
        A3_indices = (A[:, 2] == 1)
        T_pred[A3_indices] = 1.0
        
        # Compute bias amplification
        bias_amp_combined, bias_amp = metric_ba_zhao.computeBiasAmp(A, T, T_pred)

        print(f"\n[SHORTCOMING 1] Model predictions:")
        print(f"  A1: Correct predictions (no bias amplification)")
        print(f"  A2: Always predicts T=0 (amplifies negative correlation)")
        print(f"  A3: Always predicts T=1 (amplifies positive correlation)")
        print(f"  Bias amplification matrix: {bias_amp}")
        print(f"  Combined bias amplification: {bias_amp_combined:.6f}")
        print(f"  PROBLEM: BA_Zhao measures ~0 because it only looks at A1!")
        print(f"  A2 and A3 bias amplification is IGNORED (SHORTCOMING)")
        
        # The bias amplification should be small/zero because only A1 contributes
        # and A1 predictions are correct
        assert torch.abs(bias_amp_combined) < 0.1, \
            "BA_Zhao should measure ~0 even though bias is amplified in A2 and A3"

    def test_shortcoming2_group_imbalance_wrong_group_selected(self):
        """
        SHORTCOMING 2 DEMONSTRATION:
        BA_Zhao focuses on the group with most examples (A1) rather than
        the group actually more correlated with T=1 (A2).
        
        Scenario:
        - A1: 60 T=0, 30 T=1 → y1=1 (30/90 > 1/2, but P(T=1|A1)=30/90=0.33)
        - A2: 10 T=0, 20 T=1 → y2=0 (20/30 < 1/2, but P(T=1|A2)=20/30=0.67)
        
        A2 is actually MORE correlated with T=1, but BA_Zhao focuses on A1!
        """
            # A1: 60 T=0, 30 T=1
        A1_T0 = torch.zeros((60, 1), dtype=torch.float)
        A1_T1 = torch.ones((30, 1), dtype=torch.float)
        
        # A2: 10 T=0, 20 T=1
        A2_T0 = torch.zeros((10, 1), dtype=torch.float)
        A2_T1 = torch.ones((20, 1), dtype=torch.float)
        
        A = torch.cat([
            torch.tensor([[1, 0]] * 90, dtype=torch.float),  # A1
            torch.tensor([[0, 1]] * 30, dtype=torch.float),  # A2
        ], dim=0)
        
        T = torch.cat([
            torch.cat([A1_T0, A1_T1], dim=0),  # A1: 60 T=0, 30 T=1
            torch.cat([A2_T0, A2_T1], dim=0),  # A2: 10 T=0, 20 T=1
        ], dim=0)
    
        metrics = test_metrics()
        metric_ba_zhao = metrics["BA_Zhao"]
        
        # Check which group is flagged
        bias_mask = metric_ba_zhao.check_bias(A, T)
        print(f"{bias_mask=}")
        P_A_given_T = metric_ba_zhao.computeAgivenT(A, T)
        P_T_given_A = metric_ba_zhao.computeTgivenA(A, T)
        
        num_A = A.shape[1]
        threshold = 1.0 / num_A
        
        print(f"\n[SHORTCOMING 2] Group imbalance scenario:")
        print(f"  A1: 60 T=0, 30 T=1")
        print(f"    P(A1|T=1) = {P_A_given_T[0, 0]:.3f} (threshold: {threshold:.3f})")
        print(f"    P(T=1|A1) = {P_T_given_A[0, 0]:.3f}")
        print(f"    y1 = {bias_mask[0, 0]:.0f} (FLAGGED)")
        print(f"  A2: 10 T=0, 20 T=1")
        print(f"    P(A2|T=1) = {P_A_given_T[1, 0]:.3f} (threshold: {threshold:.3f})")
        print(f"    P(T=1|A2) = {P_T_given_A[1, 0]:.3f} (MORE CORRELATED!)")
        print(f"    y2 = {bias_mask[1, 0]:.0f} (NOT FLAGGED)")
        print(f"\n  PROBLEM: A2 is more correlated (P(T=1|A2)=0.67 > P(T=1|A1)=0.33)")
        print(f"    But BA_Zhao focuses on A1 because it has more examples!")
        
        # Verify A1 is flagged but A2 is not (even though A2 is more correlated)
        assert bias_mask[0, 0] == 1.0, "A1 should be flagged (more examples)"
        assert bias_mask[1, 0] == 0.0, "A2 should NOT be flagged (fewer examples)"
        assert P_T_given_A[1, 0] > P_T_given_A[0, 0], \
            "A2 should be more correlated with T=1 than A1"
        
        # Model that amplifies bias: predicts T=0 for A1, T=1 for A2
        T_pred = T.clone()
        A1_indices = (A[:, 0] == 1)
        A2_indices = (A[:, 1] == 1)
        T_pred[A1_indices] = 0.0  # Always predict T=0 for A1
        T_pred[A2_indices] = 1.0  # Always predict T=1 for A2
        
        # Compute bias amplification
        bias_amp_combined, bias_amp = metric_ba_zhao.computeBiasAmp(A, T, T_pred)
        
        print(f"\n[SHORTCOMING 2] Model predictions:")
        print(f"  A1: Always predicts T=0 (amplifies bias)")
        print(f"  A2: Always predicts T=1 (amplifies bias)")
        print(f"  Bias amplification: {bias_amp_combined:.6f}")
        print(f"  Expected: Negative score (~-0.6) because A1 is flagged but model")
        print(f"    predicts T=0 for A1, reducing P(A1|T=1) in predictions")
        print(f"  PROBLEM: Gets negative score even though bias is amplified!")  
    
# ============================================================================
# DBA TESTS
# ============================================================================

class TestDBA:
    pass