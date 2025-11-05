from utils.datacreator import dataCreator
import numpy as np

class TestingBasicFunctionality:

    def test_datacreator_returns_correct_outputs(self):
        result = dataCreator(N=128, error_percent=0.1)
        assert len(result) == 5, f"datacreator should return a tuple of size 5 but returned {len(result)}"

    def test_datacreator_returns_correct_shapes(self):
        N = 128
        P, D, D_bias, M_unbias, M2 = dataCreator(N=N, error_percent=0.1)
        assert len(P) == N, f"P should have length {N}"
        assert len(D) == N, f"D should have length {N}"
        assert len(D_bias) == N, f"D_bias should have length {N}"
        assert len(M_unbias) == N, f"M_unbias should have length {N}"
        assert len(M2) == N, f"M2 should have length {N}"

class TestingBiasValidation:

    def test_datacreator_returns_correct_P(self):
        N = 128
        q2 = N//2
        P, _, _, _, _ = dataCreator(N=N, error_percent=0.1)
        assert np.sum(P[:q2]==0) == q2, f"P should have {q2} 0s in the first half"
        assert np.sum(P[q2:]==1) == q2, f"P should have {q2} 1s in the second half"

    def test_datacreator_returns_correct_D(self):
        N = 128
        q1 = N//4
        q2 = N//2
        q3 = 3*N//4
        _, D, _, _, _ = dataCreator(N=N, error_percent=0.1)
        assert np.sum(D[:q1]==0) == q1, f"D should have {q1} 0s in the first quarter"
        assert np.sum(D[q1:q2]==1) == q1, f"D should have {q1} 1s in the second quarter"
        assert np.sum(D[q2:q3]==0) == q1, f"D should have {q1} 0s in the third quarter"
        assert np.sum(D[q3:]==1) == q1, f"D should have {q1} 1s in the fourth quarter"

    