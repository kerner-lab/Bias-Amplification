# Importing Libraries
import torch
from abc import ABC, abstractmethod
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# BASE CO-OCCURRENCE METRIC
# ============================================================================
class BaseCoOccurMetric(ABC):
    """
    Abstract base class for co-occurrence-based bias amplification metrics.
    
    This class provides common functionality for computing probabilities 
    and bias amplification computations.
    """

    def __init__(self):
        pass

    def computePairProbs(self, A: torch.tensor, T: torch.tensor) -> torch.tensor:
        """
        Computes joint probability for given A and T observations.

        Parameters
        ----------
        A : torch.tensor
            Binary tensor of the shape (N x a). a is the number of possible attribute categories. (i.e. 2 for gender {male, female})
        T : torch.tensor
            Binary tensor of the shape (N x t). t is the number of possible task categories.

        Returns
        -------
        probs : torch.tensor
            of the shape (a x t). Represents the joint probability for each A-T pair.
        """
        num_obs = A.shape[0]
        probs = A.T @ T  # (num_A, num_obs) x (num_obs, num_T) = (num_A, num_T)
        probs = probs / num_obs  # Works better if multi-class is possible
        return probs

    def computeProbs(self, vals: torch.tensor) -> torch.tensor:
        """
        Computes observed probability for each category.

        Parameters
        ----------
        vals : torch.tensor
            Binary tensor of the shape (N x v). v is the number of possible categories. (i.e. 2 for gender {male, female})

        Returns
        -------
        probs : torch.tensor
            Float tensor representing probabilities for each category.
        """
        probs = vals.mean(axis=0)
        return probs

    def computeAgivenT(self, A: torch.tensor, T: torch.tensor) -> torch.tensor:
        """
        Computes conditional probability for all Attributes A given T observations. i.e P(A|T)

        Parameters
        ----------
        A : torch.tensor
            Binary tensor of the shape (N x a). a is the number of possible attribute categories. (i.e. 2 for gender {male, female})
        T : torch.tensor
            Binary tensor of the shape (N x t). t is the number of possible task categories.

        Returns
        -------
        probs : torch.tensor
            of the shape (a x t). Represents the conditional probability P(A|T) for each A-T pair.
        """
        probs = A.T @ T  # (num_A, num_obs) x (num_obs, num_T) = (num_A, num_T)
        probs = probs / probs.sum(axis=0).clamp(min=1e-10)
        return probs

    def computeTgivenA(self, A: torch.tensor, T: torch.tensor) -> torch.tensor:
        """
        Computes conditional probability for all Task T given A observations. i.e P(T|A)

        Parameters
        ----------
        A : torch.tensor
            Binary tensor of the shape (N x a). a is the number of possible attribute categories. (i.e. 2 for gender {male, female})
        T : torch.tensor
            Binary tensor of the shape (N x t). t is the number of possible task categories.

        Returns
        -------
        probs : torch.tensor
            of the shape (a x t). Represents the conditional probability P(T|A) for each A-T pair.
        """
        probs = A.T @ T  # (num_A, num_obs) x (num_obs, num_T) = (num_A, num_T)
        probs = probs / probs.sum(axis=1).clamp(min=1e-10)
        return probs

    @abstractmethod
    def computeBiasAmp(
        self, A: torch.tensor, T: torch.tensor, T_pred: torch.tensor
    ) -> torch.tensor:
        """
        Abstract method to compute bias amplification. Subclasses must implement this method to compute the bias amplification for each A-T pair.

        Parameters
        ----------
        A : torch.tensor
            Binary tensor of shape (N x a)
        T : torch.tensor
            Binary tensor of shape (N x t)
        T_pred : torch.tensor
            Binary tensor of shape (N x t)

        Returns
        -------
        bias_amp_combined : torch.tensor
            Scalar representing mean bias amplification across all pairs
        bias_amp : torch.tensor
            Tensor of shape (a x t) representing bias amplification for each A-T pair
        """
        pass


class BA_Zhao(BaseCoOccurMetric):
    """
    Bias Amplification Metric from Zhao et al. (2021).
    This metric computes bias amplification by comparing the conditional 
    probabilities of A given T and A given T_pred.
    """

    def __init__(self):
        super().__init__()

    def check_bias(self, A: torch.tensor, T: torch.tensor) -> torch.tensor:
        """
        Checks if each A-T pair exhibits statistical dependence (positive correlation).
        Uses independence test: P(A,T) > P(A)P(T)

        Parameters
        ----------
        A : torch.tensor
            Binary tensor of shape (N x a)
        T : torch.tensor
            Binary tensor of shape (N x t) - represents ONE attribute combination

        Returns
        -------
        is_biased : torch.tensor
            Binary mask of shape (a x t) indicating positively correlated pairs
        """
        P_A_given_T = self.computeAgivenT(A, T)
        num_A = A.shape[1]
        is_biased = P_A_given_T > (1 / num_A)
        return is_biased.float()

    def computeBiasAmp(
        self, A: torch.tensor, T: torch.tensor, T_pred: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Computes bias amplification by comparing the conditional 
        probabilities of A given T and A given T_pred.

        Parameters
        ----------
        A : torch.tensor
            Binary tensor of shape (N x a)
        T : torch.tensor
            Binary tensor of shape (N x t)
        T_pred : torch.tensor
            Binary tensor of shape (N x t)

        Returns
        -------
        bias_amp_combined : torch.tensor
            Scalar representing mean bias amplification across all pairs
        bias_amp : torch.tensor
            Tensor of shape (a x t) representing bias amplification for each A-T pair
        """
        num_T = T.shape[1]
        A_T_probs = self.computePairProbs(A, T)
        A_Tpred_probs = self.computePairProbs(A, T_pred)
        bias_mask = self.check_bias(A, T)
        bias_amp = (bias_mask * A_Tpred_probs) - (bias_mask * A_T_probs)
        bias_amp = bias_amp / num_T
        bias_amp_combined = torch.sum(bias_amp)
        return bias_amp_combined, bias_amp 


class DBA(BaseCoOccurMetric):
    """
    Bias Amplification Metric from Directional Bias Amplification.
    This metric computes bias amplification that addresses on the shortcomings
    of Zhao's metric by focusing on both positive and negative correlations, 
    and the direction of amplification through comparing the conditional 
    probabilities of A given T and A given T_pred. 
    """

    def __init__(self):
        super().__init__()

    def check_bias(self, A: torch.tensor, T: torch.tensor) -> torch.tensor:
        """
        Checks if each A-T pair exhibits statistical dependence (positive correlation).
        Uses independence test: P(A,T) > P(A)P(T)

        Parameters
        ----------
        A : torch.tensor
            Binary tensor of shape (N x a)
        T : torch.tensor
            Binary tensor of shape (N x t) - represents ONE attribute combination

        Returns
        -------
        y_at : torch.tensor
            Binary mask of shape (a x t) indicating positively correlated pairs
        """
        joint_probs = self.computePairProbs(A, T)
        A_probs = self.computeProbs(A).reshape(-1, 1)
        T_probs = self.computeProbs(T).reshape(-1, 1)
        independent_probs = A_probs.matmul(T_probs.T)
        y_at = joint_probs > independent_probs
        return y_at * 1.0

    def computeBiasAmp(
        self, A: torch.tensor, T: torch.tensor, T_pred: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Computes bias amplification by comparing the conditional 
        probabilities of A given T and A given T_pred.

        Parameters
        ----------
        A : torch.tensor
            Binary tensor of shape (N x a)
        T : torch.tensor
            Binary tensor of shape (N x t)
        T_pred : torch.tensor
            Binary tensor of shape (N x t)

        Returns
        -------
        bias_amp_combined : torch.tensor
            Scalar representing mean bias amplification across all pairs
        bias_amp : torch.tensor
            Tensor of shape (a x t) representing bias amplification for each A-T pair
        """
        num_A = A.shape[1]
        num_T = T.shape[1]
        y_at = self.check_bias(A, T)
        P_T_given_A = self.computeTgivenA(A, T)
        P_Tpred_given_A = self.computeTgivenA(A, T_pred)
        print(f"{P_T_given_A=}")
        print(f"{P_Tpred_given_A=}")
        delta_at = P_Tpred_given_A - P_T_given_A
        bias_amp = (y_at * delta_at) + ((1 - y_at) * (-1 * delta_at))
        bias_amp = bias_amp / (num_A * num_T)
        bias_amp_combined = torch.sum(bias_amp)
        return bias_amp_combined, bias_amp

    def computeBiasAmpBidirectional(
        self,
        A: torch.tensor,
        A_pred: torch.tensor,
        T: torch.tensor,
        T_pred: torch.tensor,
    ) -> dict[str, tuple[torch.tensor, torch.tensor]]:
        """
        Computes bidirectional bias amplification for AtoT and TtoA directions.
        Parameters
        ----------
        A : torch.tensor
            Binary tensor of shape (N x a)
        A_pred : torch.tensor
            Binary tensor of shape (N x a)
        T : torch.tensor
            Binary tensor of shape (N x t)
        T_pred : torch.tensor
            Binary tensor of shape (N x t)

        Returns
        -------
        bias_amp : dict
            Dictionary with keys 'AtoT' and 'TtoA', each containing
            (mean, variance) tuples
        """
        bias_amp_AT = self.computeBiasAmp(A, T, T_pred)
        bias_amp_TA = self.computeBiasAmp(T, A, A_pred)
        bias_amp = {"AtoT": bias_amp_AT, "TtoA": bias_amp_TA}
        return bias_amp


class MDBA(BaseCoOccurMetric):
    """
    Multi-Attribute Directional Bias Amplification Metric.
    This metric computes bias amplification that addresses on the shortcomings
    of DBA by focusing on multi-attribute combinations through comparing the conditional 
    probabilities of A given T and A given T_pred. 
    """

    def __init__(self, min_attr_size: int = 1, max_attr_size: int = None):
        super().__init__()
        self.min_attr_size = min_attr_size
        self.max_attr_size = max_attr_size

    def check_bias(self, A: torch.tensor, T: torch.tensor) -> torch.tensor:
        """
        Checks if each A-T pair exhibits statistical dependence (positive correlation).
        Uses independence test: P(A,T) > P(A)P(T)

        Parameters
        ----------
        A : torch.tensor
            Binary tensor of shape (N x a)
        T : torch.tensor
            Binary tensor of shape (N x t) - represents ONE attribute combination

        Returns
        -------
        y_at : torch.tensor
            Binary mask of shape (a x t) indicating positively correlated pairs
        """
        joint_probs = self.computePairProbs(A, T)
        A_probs = self.computeProbs(A).reshape(-1, 1)
        T_probs = self.computeProbs(T).reshape(-1, 1)
        independent_probs = A_probs.matmul(T_probs.T)
        y_at = joint_probs > independent_probs
        return y_at * 1.0

    def _generateAttributeCombinations(
        self,
        T: torch.tensor,
    ) -> list[tuple[torch.tensor, tuple]]:
        """
        Generate all combinations of attributes for multi-attribute analysis.

        Parameters
        ----------
        T : torch.tensor
            Attribute tensor, shape (N x t)

        Returns
        -------
        combinations : list[tuple[torch.tensor, tuple]]
            List of (tensor, indices) tuples where:
            - tensor: binary mask of shape (N x 1) indicating presence of combination
            - indices: tuple of attribute indices in the combination
        """
        num_T = T.shape[1]

        min_size = self.min_attr_size
        max_size = self.max_attr_size if self.max_attr_size else num_T

        combinations = []

        # Generate all possible combinations of attributes
        from itertools import combinations as iter_combinations

        for size in range(min_size, min(max_size + 1, num_T + 1)):
            for combo in iter_combinations(range(num_T), size):
                # Create binary mask: 1 only if ALL attributes in combo are present
                combo_mask = torch.ones(T.shape[0], dtype=torch.float)
                for attr_idx in combo:
                    combo_mask = combo_mask * T[:, attr_idx]

                # Only include if this combination actually occurs in the data
                if combo_mask.sum() >= 1:  # At least one instance
                    combinations.append((combo_mask.reshape(-1, 1), combo))

        return combinations

    def computeBiasAmp(
        self, A: torch.tensor, T: torch.tensor, T_pred: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Computes Multi-Dimensional Bias Amplification from A to T.

        This implements the Multi-> directional metric from the paper (Equation 3).
        It iterates over ALL combinations of attributes M and computes bias amplification
        for each combination, then aggregates.

        The formula from the paper:
        Multi-> = (mean, variance) where
        mean = (1 / |G||M|) * Σ_g Σ_m |y_gm * Δ_gm + (1 - y_gm) * |-Δ_gm||

        Parameters
        ----------
        A : torch.tensor
            Ground truth group membership, shape (N x a)
        T : torch.tensor
            Ground truth tasks/attributes, shape (N x t)
        T_pred : torch.tensor
            Predicted tasks/attributes, shape (N x t)

        Returns
        -------
        bias_amp_mean : torch.tensor
            Scalar representing mean bias amplification across all combinations
        bias_amp_variance : torch.tensor
            Variance of bias amplification (shows if uniform or concentrated)
        """
        num_A = A.shape[1]

        # Generate ALL attribute combinations
        combinations = self._generateAttributeCombinations(T)
        num_M = len(combinations)

        if num_M == 0:
            return torch.tensor(0.0), torch.tensor(0.0)

        # Store all delta values for variance calculation
        all_deltas = []
        total_bias_amp = 0.0

        # Iterate over all attribute combinations m ∈ M
        for m_tensor, m_indices in combinations:
            # Create corresponding prediction tensor for this combination
            m_pred_tensor = torch.ones(T_pred.shape[0], dtype=torch.float)
            for attr_idx in m_indices:
                m_pred_tensor = m_pred_tensor * T_pred[:, attr_idx]
            m_pred_tensor = m_pred_tensor.reshape(-1, 1)

            # Check which A-m pairs are positively correlated in the data
            y_am = self.check_bias(A, m_tensor)

            # Compute conditional probabilities P(m|A) for data and predictions
            P_m_given_A = self.computeTgivenA(A, m_tensor)
            P_mpred_given_A = self.computeTgivenA(A, m_pred_tensor)

            # Calculate change in conditional probability
            delta_am = P_mpred_given_A - P_m_given_A

            # Weight by bias direction and take absolute value
            # For each group g and attribute combination m:
            # If positively correlated (y_am=1): contribution = |delta|
            # If negatively correlated (y_am=0): contribution = |-delta| = |delta|
            weighted_delta = (y_am * delta_am) + ((1 - y_am) * (-delta_am))
            abs_weighted_delta = torch.abs(weighted_delta)

            # Sum over all groups for this attribute combination
            total_bias_amp += torch.sum(abs_weighted_delta)

            # Store weighted deltas for variance calculation
            all_deltas.append(weighted_delta.flatten())

        # Normalize by number of groups and attribute combinations
        bias_amp_mean = total_bias_amp / (num_A * num_M)

        # Compute variance across all group-attribute pairs
        if len(all_deltas) > 0:
            all_deltas_cat = torch.cat(all_deltas)
            bias_amp_variance = torch.var(all_deltas_cat)
        else:
            bias_amp_variance = torch.tensor(0.0)

        return bias_amp_mean, bias_amp_variance

    def computeBiasAmpBidirectional(
        self,
        A: torch.tensor,
        A_pred: torch.tensor,
        T: torch.tensor,
        T_pred: torch.tensor,
    ) -> dict[str, tuple[torch.tensor, torch.tensor]]:
        """
        Computes bidirectional bias amplification.

        This captures bias amplification in both directions:
        - Multi_A->T (or Multi_G->M): How group membership (A) influences task predictions (T)
        - Multi_T->A (or Multi_M->G): How tasks (T) influence group membership predictions (A)

        Parameters
        ----------
        A : torch.tensor
            Ground truth group membership
        A_pred : torch.tensor
            Predicted group membership
        T : torch.tensor
            Ground truth tasks/attributes
        T_pred : torch.tensor
            Predicted tasks/attributes

        Returns
        -------
        bias_amp : dict
            Dictionary with keys 'AtoT' and 'TtoA', each containing
            (mean, variance) tuples
        """
        # A->T means: given group A, how does it affect prediction of T
        bias_amp_AT = self.computeBiasAmp(A, T, T_pred)
        # T->A means: given task T, how does it affect prediction of A
        bias_amp_TA = self.computeBiasAmp(T, A, A_pred)
        bias_amp = {"AtoT": bias_amp_AT, "TtoA": bias_amp_TA}
        return bias_amp

    def getAttributeCombinationStats(
        self,
        T: torch.tensor,
    ) -> dict:
        """
        Get statistics about attribute combinations in the dataset.
        Useful for understanding the dataset structure.

        Returns
        -------
        stats : dict
            Dictionary containing:
            - 'total_combinations': Total number of attribute combinations
            - 'by_size': Number of combinations for each size
            - 'examples': Example combinations for each size
        """
        combinations = self._generateAttributeCombinations(T)

        stats = {"total_combinations": len(combinations), "by_size": {}, "examples": {}}

        for _, m_indices in combinations:
            size = len(m_indices)
            if size not in stats["by_size"]:
                stats["by_size"][size] = 0
                stats["examples"][size] = []

            stats["by_size"][size] += 1
            if len(stats["examples"][size]) < 5:  # Store up to 5 examples
                stats["examples"][size].append(m_indices)

        return stats


if __name__ == "__main__":
    # Data Initialization
    from utils.datacreator import dataCreator

    P, D, D2, M1, M2 = dataCreator(16384, 0.2, False, 0.05)
    P = torch.tensor(P, dtype=torch.float).reshape(-1, 1)
    P = torch.hstack([P, 1 - P])
    D = torch.tensor(D, dtype=torch.float).reshape(-1, 1)
    D = torch.hstack([D, 1 - D])
    D2 = torch.tensor(D2, dtype=torch.float).reshape(-1, 1)
    D2 = torch.hstack([D2, 1 - D2])
    M1 = torch.tensor(M1, dtype=torch.float).reshape(-1, 1)
    M1 = torch.hstack([M1, 1 - M1])
    M2 = torch.tensor(M2, dtype=torch.float).reshape(-1, 1)
    M2 = torch.hstack([M2, 1 - M2])

    # Calculating Params
    model_1_acc = torch.sum(D == M1) / D.shape[0]
    model_2_acc = torch.sum(D == M2) / D.shape[0]

    # Parameter Initialization
    dpa_obj = DBA()
    dpa_1 = dpa_obj.computeBiasAmp(P, D, M1)
    print(f"DPA for case 1: {dpa_1}")
    print("______________________________________")
    print("______________________________________")
    dpa_2 = dpa_obj.computeBiasAmp(P, D, M2)
    print(f"DPA for case 2: {dpa_2}")
    print("______________________________________")
    print("______________________________________")
    dpa_3 = dpa_obj.computeBiasAmp(P, D2, M1)
    print(f"DPA for case 3: {dpa_3}")
    print("______________________________________")
    print("______________________________________")
    dpa_4 = dpa_obj.computeBiasAmp(P, D2, M2)
    print(f"DPA for case 4: {dpa_4}")
    print("______________________________________")
    print("______________________________________")
