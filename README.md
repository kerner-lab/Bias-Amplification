# bias-amplification

[![PyPI version](https://img.shields.io/pypi/v/bias-amplification.svg)](https://pypi.org/project/bias-amplification/)
[![Python versions](https://img.shields.io/pypi/pyversions/bias-amplification.svg)](https://pypi.org/project/bias-amplification/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://readthedocs.org/projects/bias-amplification/badge/?version=stable)](https://bias-amplification.readthedocs.io/en/stable/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/161CRQSiy8zF3yBNlCHPJGBgRhZhQ72u9?usp=sharing)

A Python library of state-of-the-art metrics for measuring bias amplification in machine learning models.

## Installation

```bash
pip install bias-amplification
```

## Metrics at a Glance

### Co-occurrence Metrics
Measure bias by analyzing statistical relationships between protected attributes and predictions — no model training required.

| Metric | Class | What it captures | Paper |
|--------|-------|-----------------|-------|
| BA_MALS | `BA_MALS` | Bias amplification in positively correlated attribute–label pairs | Zhao et al., 2021 |
| DBA | `DBA` | Bias amplification in both positively and negatively correlated pairs | — |
| MDBA | `MDBA` | Bias amplification across all attribute combinations (intersectional) | — |

### Predictability Metrics
Measure bias by training an attacker model to predict protected attributes from model outputs.

| Metric | Class | What it captures | Paper |
|--------|-------|-----------------|-------|
| Leakage | `Leakage` | Raw information leakage from predictions to protected attributes | — |
| DPA | `DPA` | Directional leakage: A→T and T→A | [Tokas et al., NeurIPS 2025](https://arxiv.org/abs/2412.11060) |
| DBAC | `DBAC` | Directional bias amplification in image captioning / generation | [Nair et al., WACV 2026](https://arxiv.org/abs/2503.07878) |

## Quick Start

Try the interactive demo: [Open in Colab](https://colab.research.google.com/drive/161CRQSiy8zF3yBNlCHPJGBgRhZhQ72u9?usp=sharing)

Or run your first metric in a few lines:

```python
from bias_amplification.metrics import DBA
import numpy as np

A      = np.array([[1,0],[0,1],[1,0],[0,1]])   # protected attributes (one-hot)
T      = np.array([[1,0],[1,0],[0,1],[0,1]])   # ground truth labels   (one-hot)
T_pred = np.array([[1,0],[0,1],[0,1],[1,0]])   # model predictions     (one-hot)

dba = DBA()
score, matrix = dba.computeBiasAmp(A, T, T_pred)
print(score)   # scalar bias amplification value
```

## Usage

### Co-occurrence Metrics

All co-occurrence metrics accept one-hot encoded arrays: `A` (protected attributes), `T` (ground truth labels), and `T_pred` (model predictions).

**BA_MALS** — measures bias amplification in positively correlated attribute–label pairs only:

```python
from bias_amplification.metrics import BA_MALS

ba_mals = BA_MALS()
score, matrix = ba_mals.computeBiasAmp(A, T, T_pred)
# score : scalar
# matrix: (num_attributes, num_labels) per-pair amplification values
```

**DBA** — extends BA_MALS to capture negatively correlated pairs as well; supports bidirectional analysis:

```python
from bias_amplification.metrics import DBA

dba = DBA()
score, matrix = dba.computeBiasAmp(A, T, T_pred)

# Bidirectional: A→T and T→A simultaneously
results = dba.computeBiasAmpBidirectional(A, A_pred, T, T_pred)
# results: {"AtoT": (score, matrix), "TtoA": (score, matrix)}
```

**MDBA** — measures bias amplification across all combinations of attributes (intersectional fairness):

```python
from bias_amplification.metrics import MDBA

mdba = MDBA(min_attr_size=1, max_attr_size=3)
mean_amp, variance = mdba.computeBiasAmp(A, T, T_pred)

# Inspect which combinations are analyzed
stats = mdba.getAttributeCombinationStats(T)
# {"total_combinations": 7, "by_size": {1: 3, 2: 3, 3: 1}, ...}
```

---

### Predictability Metrics

Predictability metrics train a small neural network (the "attacker") to predict protected attributes from model outputs. You provide the attacker architecture and the known accuracy of your model for quality equalization.

**Leakage** — measures raw information leakage from a model's predictions to protected attributes:

```python
from bias_amplification.metrics import Leakage
from bias_amplification.attacker_models import simpleDenseModel

attacker = simpleDenseModel(
    input_dims=1, output_dims=1, num_layers=1,
    numFirst=1, activations=["sigmoid"]
)

leakage = Leakage(
    attacker_model=attacker,
    train_params={"learning_rate": 0.01, "loss_function": "bce",
                  "epochs": 100, "batch_size": 64},
    model_acc=0.8,        # your model's accuracy (float 0–1, or int 0–100)
    eval_metric="accuracy",
)
result = leakage.computeBiasAmp(A, T, T_pred, num_trials=10)
print(result)  # "0.45 ± 0.02"  (mean ± std over trials)
```

**DPA** — directional leakage; quantifies how much information leaks in each direction (A→T and T→A):

```python
from bias_amplification.metrics import DPA
from bias_amplification.attacker_models import simpleDenseModel

attacker_AtoT = simpleDenseModel(1, 1, 1, numFirst=1, activations=["sigmoid"])
attacker_TtoA = simpleDenseModel(1, 1, 1, numFirst=1, activations=["sigmoid"])

dpa = DPA(
    attacker_AtoT=attacker_AtoT,
    attacker_TtoA=attacker_TtoA,
    train_params={"learning_rate": 0.01, "loss_function": "bce",
                  "epochs": 100, "batch_size": 64},
    model_acc={"AtoT": 0.8, "TtoA": 0.7},   # per-direction accuracy
    eval_metric="accuracy",
)

# Single direction
score_atot = dpa.computeBiasAmp(A, T, T_pred, mode="AtoT", num_trials=10)

# Both directions at once
score_atot, score_ttoa = dpa.computeBiasAmpBidirectional(A, T, A_pred, T_pred)
```

**DBAC** — directional bias amplification in image captions and generative outputs. See the [DBAC documentation](https://bias-amplification.readthedocs.io/en/stable/api_reference/index.html) for usage.

---

## Documentation

Full API reference: [bias-amplification.readthedocs.io](https://bias-amplification.readthedocs.io/en/stable/api_reference/index.html)

---

## Citation

If you use **DPA** or **DBAC** in your research, please cite the relevant paper:

<details>
<summary>DPA — NeurIPS 2025</summary>

```bibtex
@article{tokas2026dpa,
  title={DPA: A one-stop metric to measure bias amplification in classification datasets},
  author={Tokas, Bhanu and Nair, Rahul and Kerner, Hannah},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={150885--150914},
  year={2026}
}
```

</details>

<details>
<summary>DBAC — WACV 2026</summary>

```bibtex
@inproceedings{nair2026woman,
  title={A Woman with a Knife or A Knife with a Woman? Measuring Directional Bias Amplification in Image Captions},
  author={Nair, Rahul and Tokas, Bhanu and Kerner, Hannah},
  booktitle={2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={255--264},
  year={2026},
  organization={IEEE}
}
```

</details>

---

## License

MIT
