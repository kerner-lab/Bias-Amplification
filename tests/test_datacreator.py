from utils.datacreator import dataCreator, validate_error_percent
import numpy as np
import pytest


class TestingBasicFunctionality:

    def test_datacreator_returns_correct_outputs(self):
        result = dataCreator(N=128, error_percent=0.1)
        assert (
            len(result) == 5
        ), f"datacreator should return a tuple of size 5 but returned {len(result)}"

    def test_datacreator_returns_correct_shapes(self):
        N = 128
        P, D, D_bias, M_unbias, M2 = dataCreator(N=N, error_percent=0.1)
        assert len(P) == N, f"P should have length {N}"
        assert len(D) == N, f"D should have length {N}"
        assert len(D_bias) == N, f"D_bias should have length {N}"
        assert len(M_unbias) == N, f"M_unbias should have length {N}"
        assert len(M2) == N, f"M2 should have length {N}"

    def test_validate_error_percent_on_correct_values(self):
        try:
            validate_error_percent(0, "error")
            validate_error_percent(0.15, "error")
            validate_error_percent(0.25, "error")
            assert True, "validate_error_percent should not raise an error"
        except ValueError:
            pytest.fail("validate_error_percent() raised Valueerror unexpectedly")

    def test_validate_error_percent_on_negative_values(self):
        with pytest.raises(ValueError):
            validate_error_percent(-0.1, "error")

    def test_validate_error_percent_on_high_values(self):
        with pytest.raises(ValueError):
            validate_error_percent(0.8, "error")


class TestingBiasValidation:

    def test_datacreator_returns_correct_P(self):
        N = 128
        q2 = N // 2
        P, _, _, _, _ = dataCreator(N=N, error_percent=0.1)
        assert np.sum(P[:q2] == 0) == q2, f"P should have {q2} 0s in the first half"
        assert np.sum(P[q2:] == 1) == q2, f"P should have {q2} 1s in the second half"

    def test_datacreator_returns_correct_D(self):
        N = 128
        q1 = N // 4
        mid = N // 2
        q3 = 3 * N // 4
        _, D, _, _, _ = dataCreator(N=N, error_percent=0.1)
        assert np.sum(D[:q1] == 0) == q1, f"D should have {q1} 0s in the first quarter"
        assert np.sum(D[q1:mid] == 1) == q1, f"D should have {q1} 1s in the second quarter"
        assert np.sum(D[mid:q3] == 0) == q1, f"D should have {q1} 0s in the third quarter"
        assert np.sum(D[q3:] == 1) == q1, f"D should have {q1} 1s in the fourth quarter"

    def test_datacreator_returns_correct_D_bias(self):
        N = 16384
        data_err_pct = 0.2
        data_err_cnt_half = int(N * (data_err_pct / 2))
        data_err_cnt = data_err_cnt_half * 2
        _, D, D_bias, _, _ = dataCreator(N=N, data_error_percent=data_err_pct)
        q1 = N // 4
        mid = N // 2
        q3 = 3 * N // 4
        # check the count of errors
        tot_errors_in_D_bias = np.sum(D != D_bias)
        errors_in_quarter_1 = np.sum(D[0:q1] != D_bias[0:q1])
        errors_in_quarter_2 = np.sum(D[q1:mid] != D_bias[q1:mid])
        errors_in_quarter_3 = np.sum(D[mid:q3] != D_bias[mid:q3])
        errors_in_quarter_4 = np.sum(D[q3:] != D_bias[q3:])
        # asserting tests
        assert (
            tot_errors_in_D_bias == data_err_cnt
        ), f"D_bias should have a total of {data_err_cnt} errors"
        assert (
            errors_in_quarter_1 == errors_in_quarter_4 == data_err_cnt_half
        ), f"D_bias should have {data_err_cnt_half} errors in quarter 1 and 4"
        assert (
            errors_in_quarter_2 == errors_in_quarter_3 == 0
        ), f"D_bias should have zero errors in quarter 2 and 3"

    def test_datacreator_returns_correct_M_unbias(self):
        N = 16384
        err_pct = 0.2
        err_cnt_half = int(N * (err_pct / 2))
        tot_err_cnt = err_cnt_half * 2
        _, D, _, M_unbias, _ = dataCreator(N=N, error_percent=err_pct)
        q1 = N // 4
        mid = N // 2
        q3 = 3 * N // 4
        # check the count of errors
        tot_errors_in_M_unbias = np.sum(D != M_unbias)
        errors_in_quarter_1 = np.sum(D[0:q1] != M_unbias[0:q1])
        errors_in_quarter_2 = np.sum(D[q1:mid] != M_unbias[q1:mid])
        errors_in_quarter_3 = np.sum(D[mid:q3] != M_unbias[mid:q3])
        errors_in_quarter_4 = np.sum(D[q3:] != M_unbias[q3:])
        # asserting tests
        assert (
            tot_errors_in_M_unbias == tot_err_cnt
        ), f"M_unbias should have a total of {tot_err_cnt} errors"
        assert (
            errors_in_quarter_1 == errors_in_quarter_2 == err_cnt_half // 2
        ), f"M_unbias should have {err_cnt_half//2} errors in quarter 1 and 2"
        assert (
            errors_in_quarter_3 == errors_in_quarter_4 == err_cnt_half - err_cnt_half // 2
        ), f"M_unbias should have {err_cnt_half - err_cnt_half//2} errors in quarter 3 and 4"

    def test_datacreator_returns_correct_M2(self):
        N = 16384
        err_pct = 0.2
        err_cnt_half = int(N * (err_pct / 2))
        tot_err_cnt = err_cnt_half * 2
        _, D, _, _, M2 = dataCreator(N=N, error_percent=err_pct)
        q1 = N // 4
        mid = N // 2
        q3 = 3 * N // 4
        # check the count of errors
        tot_errors_in_M2 = np.sum(D != M2)
        errors_in_half_1 = np.sum(D[0:mid] != M2[0:mid])
        errors_in_half_2 = np.sum(D[mid:] != M2[mid:])
        errors_in_quarter_1 = np.sum(D[0:q1] != M2[0:q1])
        errors_in_quarter_2 = np.sum(D[q1:mid] != M2[q1:mid])
        errors_in_quarter_3 = np.sum(D[mid:q3] != M2[mid:q3])
        errors_in_quarter_4 = np.sum(D[q3:] != M2[q3:])
        # asserting tests
        assert tot_errors_in_M2 == tot_err_cnt, f"M2 should have a total of {tot_err_cnt} errors"
        assert (
            errors_in_half_1 == errors_in_half_2 == err_cnt_half
        ), f"M2 should have {err_cnt_half} errors in each half"
        assert (
            errors_in_quarter_1 == errors_in_quarter_4 == err_cnt_half
        ), f"M2 should have {err_cnt_half} errors in quarter 1 and 4"
        assert (
            errors_in_quarter_2 == errors_in_quarter_3 == 0
        ), f"M2 should have zero errors in quarter 2 and 3"
