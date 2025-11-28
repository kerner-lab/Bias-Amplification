Leakage Module
==============

BasePredictabilityMetric
-------------------------

.. autoclass:: bias_amplification.metrics.PredMetrics.BasePredictabilityMetric
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Leakage
-------

.. autoclass:: bias_amplification.metrics.PredMetrics.Leakage
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   .. code-block:: python

      from bias_amplification.metrics.PredMetrics import Leakage
      from bias_amplification.attacker_models import simpleDenseModel
      import torch

      # Create attacker model
      attacker_model = simpleDenseModel(
          1, 1, 1, numFirst=1,
          activations=["sigmoid", "sigmoid", "sigmoid"]
      )

      # Initialize Leakage metric
      leakage = Leakage(
          attacker_model=attacker_model,
          train_params={
              "learning_rate": 0.01,
              "loss_function": "bce",
              "epochs": 100,
              "batch_size": 64
          },
          model_acc=0.8,
          eval_metric="accuracy"
      )

      # Calculate amortized leakage
      result = leakage.getAmortizedLeakage(
          features, ground_truth, predictions,
          num_trials=10
      )

DPA (Differential Predictability Analysis)
------------------------------------------

.. autoclass:: bias_amplification.metrics.PredMetrics.DPA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   .. code-block:: python

      from bias_amplification.metrics.PredMetrics import DPA
      from bias_amplification.attacker_models import simpleDenseModel

      # Create attacker models
      attacker_AtoT = simpleDenseModel(1, 1, 1, numFirst=1)
      attacker_TtoA = simpleDenseModel(1, 1, 1, numFirst=1)

      # Initialize DPA metric
      dpa = DPA(
          attacker_AtoT=attacker_AtoT,
          attacker_TtoA=attacker_TtoA,
          train_params={...},
          model_acc={"AtoT": 0.8, "TtoA": 0.7},
          eval_metric="accuracy"
      )

      # Calculate bidirectional leakage
      atot_result, ttoa_result = dpa.calcBidirectional(
          protected_attr, target,
          pred_protected, pred_target
      )

