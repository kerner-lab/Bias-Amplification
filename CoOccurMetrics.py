# Importing Libraries
import torch
from abc import ABC, abstractmethod


# TODO: Add Documentation. Add note on thresholding inputs before giving to the models. (Doesn't work for probabilities.)


# Class Definition
class BaseCoOccurMetric(ABC):

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
        # probs = probs/probs.sum()
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
        probs = probs / probs.sum(axis=0)
        # num_A = A.shape[1]
        # num_T = T.shape[1]
        # sum_T = T.sum(axis=0)
        # probs = torch.zeros((num_A, num_T))
        # for T_pos in range(num_T):
        #     curr_A = A[T[:, T_pos] == 1]
        #     curr_A_probs = curr_A.sum(axis=0) / sum_T[T_pos]
        #     probs[:, T_pos] = curr_A_probs
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
        probs = probs / probs.sum(axis=1)
        # num_A = A.shape[1]
        # num_T = T.shape[1]
        # sum_A = A.sum(axis=0)
        # probs = torch.zeros((num_A, num_T))
        # for A_pos in range(num_A):
        #     curr_T = T[A[:, A_pos] == 1]
        #     curr_T_probs = curr_T.sum(axis=0) / sum_A[A_pos]
        #     probs[A_pos] = curr_T_probs
        return probs

    @abstractmethod
    def computeBiasAmp(
        self, A: torch.tensor, T: torch.tensor, T_pred: torch.tensor
    ) -> torch.tensor:
        pass


class BA_Zhao(BaseCoOccurMetric):

    def __init__(self):
        super().__init__()

    def biasCheck(self, A: torch.tensor, T: torch.tensor) -> torch.tensor:
        P_A_given_T = self.computeAgivenT(A, T)
        num_A = A.shape[1]
        is_biased = P_A_given_T > (1 / num_A)
        return is_biased

    def computeBiasAmp(
        self, A: torch.tensor, T: torch.tensor, T_pred: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:
        num_T = T.shape[1]
        A_T_probs = self.computePairProbs(A, T)
        A_Tpred_probs = self.computePairProbs(A, T_pred)
        bias_mask = self.biasCheck(A, T)
        bias_amp = (bias_mask * A_Tpred_probs) - (bias_mask * A_T_probs)
        bias_amp = bias_amp / num_T
        bias_amp_combined = torch.sum(bias_amp)
        return bias_amp_combined, bias_amp


class DBA(BaseCoOccurMetric):

    def __init__(self):
        super().__init__()

    def biasCheck(self, A: torch.tensor, T: torch.tensor) -> torch.tensor:
        joint_probs = self.computePairProbs(A, T)
        A_probs = self.computeProbs(A).reshape(-1, 1)
        T_probs = self.computeProbs(T).reshape(-1, 1)
        independent_probs = A_probs.matmul(T_probs.T)
        y_at = joint_probs > independent_probs
        return y_at * 1.0

    def computeBiasAmp(
        self, A: torch.tensor, T: torch.tensor, T_pred: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:
        num_A = A.shape[1]
        num_T = T.shape[1]
        y_at = self.biasCheck(A, T)
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
        bias_amp_AT = self.computeBiasAmp(A, T, T_pred)
        bias_amp_TA = self.computeBiasAmp(T, A, A_pred)
        bias_amp = {"AtoT": bias_amp_AT, "TtoA": bias_amp_TA}
        return bias_amp


class MDBA(BaseCoOccurMetric):

    def __init__(self):
        pass


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
