# Importing Libraries
import numpy as np
import pandas as pd


# validator
def validate_error_percent(error_percent, name):
    """
    Checks if the error percentage is valid. otherwise raises an error.
    """
    if 0 <= error_percent <= 0.25:
        pass
    # elif error_percent > 1 and error_percent < 25:
        # error_percent = error_percent / 100
    else:
        raise ValueError(f"{name}_percent needs to be a float value in [0,0.25].")


# dataCreator function
def dataCreator(N=512, error_percent=0.1, shuffle=False, data_error_percent=None):
    """
    This function  a synthetic dataset for the bias amplification analysis.

    Input:
     N: Dataset size
     error_percent: Error percentage of model (must be between 0 and 0.25)
     shuffle: boolean value to shuffle the data
     data_error_percent: Error percentage of data (must be between 0 and 0.25)

    Output:
    A tuple containing:
     P: Protected attribute
     D: Ground truth task label
     D_bias: Biased task data label
     M_unbias: Model prediction without bias
     M2: Model prediction with bias
    """
    if N < 3 or type(N) != int:
        raise ValueError("N must be an integer >= 3. Got {N}")

    error_percent = error_percent / 2
    validate_error_percent(error_percent, "error")

    if data_error_percent == None:
        data_error_percent = error_percent
    else:
        data_error_percent = data_error_percent / 2
        validate_error_percent(data_error_percent, "data_error")

    # Calculating partitions
    q1 = N // 4
    q2 = N // 2
    q3 = 3 * N // 4

    # Initialize protected attribute P
    # First half: group 0, Second half: group 1
    P = np.zeros(N)
    P[q2:] = 1

    # Initialize ground truth task label D
    # First quarter: group 0, Second quarter: group 1, Third quarter: group 0, Fourth quarter: group 1
    D = np.zeros(N)
    D[q1:q2] = 1
    D[q3:] = 1

    # Part A: P=0, D=0
    # Part B: P=0, D=1
    # Part C: P=1, D=0
    # Part D: P=1, D=1

    M_unbias = D.copy() #Model without bias
    M2 = D.copy() #Model with bias
    D_bias = D.copy() #Bias in the data

    num_errors = int(N * error_percent)
    num_data_errors = int(N * data_error_percent)

    # First quarter positions
    A_pos = np.array([i for i in range(0, N // 4)])
    # Third quarter positions
    C_pos = np.array([i for i in range(N // 2, 3 * N // 4)])

    #For M_unbias: introducing balanced error across quarters 1 and 3
    #Randomly choosing num_errors//2 indices from quarter 1 
    swaps_m_unbias_in_A = np.random.choice(A_pos, num_errors // 2, replace=False)
    #Randomly choosing other num_errors // 2 indices from quarter 3
    swaps_m_unbias_in_C = np.random.choice(C_pos, num_errors - num_errors // 2, replace=False)
    #Flip the chosen indices from M_unbias=0 to M_unbias=1 in quarter 1
    M_unbias[swaps_m_unbias_in_A] = 1
    #Flip from M_unbias=1 to M_unbias=0 in corresponding positions in quarter 2
    M_unbias[swaps_m_unbias_in_A + (q1)] = 0
    #Flip the chosen indices from M_unbias=0 to M_unbias=1 in quarter 3
    M_unbias[swaps_m_unbias_in_C] = 1
    #Flip from M_unbias=1 to M_unbias=0 in corresponding positions in quarter 4
    M_unbias[swaps_m_unbias_in_C + (q1)] = 0

    # For M2: introducing all errors in quarter 1 only
    #Randomly choosing num_errors indices from quarter 1
    swaps_m_bias_in_A = np.random.choice(A_pos, num_errors, replace=False)
    #Flip the chosen indices from M2=0 to M2=1 in quarter 1
    M2[swaps_m_bias_in_A] = 1
    #Flip from M2=1 to M2=0 in corresponding positions in quarter 4
    M2[swaps_m_bias_in_A + (q3)] = 0
    
    #For D_bias: introducing bias in quarter 1 only in the data
    swaps_d_bias_in_A = np.random.choice(A_pos, num_data_errors, replace=False)
    #Flip the chosen indices from D_bias=0 to D_bias=1 in quarter 1
    D_bias[swaps_d_bias_in_A] = 1
    #Flip from D_bias=1 to D_bias=0 in corresponding positions in quarter 4
    D_bias[swaps_d_bias_in_A + (q3)] = 0

    if shuffle:
        permut = np.random.permutation(N)
        P = P[permut]
        D = D[permut]
        M_unbias = M_unbias[permut]
        M2 = M2[permut]
    return P, D, D_bias, M_unbias, M2


# Stability Experiment
def StabilityExp(N, data_error_w=0.5, model_error_w=0.2, poly_pow=4, data_range=(0, 1)):
    data_min, data_max = data_range
    A = data_min + (np.random.random(N) * (data_max - data_min))
    coeffs = np.random.randint(10, size=1 + poly_pow)
    polynom = np.poly1d(coeffs)
    error = np.random.random(N)
    D = polynom(A + data_error_w * error)
    M = polynom(A + model_error_w * error)
    return A, D, M


# COMPAS Dataset
COMPAS_SENSITIVE_ATTRS = ["sex", "race", "age"]


def COMPASData(attributes="race"):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    )
    """
    Removing records where charged for alternative reasons. Reference below:
    Jeff Larson, Surya Mattu, Lauren Kirchner, and Julia Angwin. How we analyzed the compas recidivism algorithm. 2016. URL: https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm.
    """
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].reset_index(drop=True)
    if type(attributes) == str:
        attributes = [attributes]
    for item in attributes:
        if not (item in COMPAS_SENSITIVE_ATTRS):
            raise Exception(
                f"{item} not in known sensitive attribute list: {COMPAS_SENSITIVE_ATTRS}"
            )
        A = df[attributes].values
        T = df["is_recid"].values
        T_pred = df["two_year_recid"].values
    return df


if __name__ == "__main__":
    P1, D1, D_bias1, M_unbias1, M2_1 = dataCreator(32, 0.1, False, 0.2)
    print(f"{P1=}")
    print(f"{D1=}")
    print(f"{D_bias1=}")
    print(f"{M_unbias1=}")
    print(f"{M2_1=}")

    df = COMPASData()
    print(f"{df=}")