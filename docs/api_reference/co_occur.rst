Co-Occurrence Metrics
=====================

This module provides classes for computing bias amplification metrics based on co-occurrence analysis between protected attributes and task labels.

BaseCoOccurMetric
-----------------

.. autoclass:: bias_amplification.metrics.CoOccurMetrics.BaseCoOccurMetric
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   .. code-block:: python

      from bias_amplification.metrics.CoOccurMetrics import BaseCoOccurMetric
      import torch

      # BaseCoOccurMetric is abstract - use concrete implementations
      # See BA_MALS, DBA, or MDBA below

BA_MALS
-------

Bias Amplification Metric from Zhao et al. (2021). This metric computes bias amplification by comparing conditional probabilities, but only focuses on positive correlations.

.. autoclass:: bias_amplification.metrics.CoOccurMetrics.BA_MALS
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   .. code-block:: python

      from bias_amplification.metrics.CoOccurMetrics import BA_MALS
      import torch

      # Initialize BA_MALS metric
      ba_mals = BA_MALS()

      # Prepare data: A (attributes) and T (tasks) as binary tensors
      # A: shape (N, a) where N is number of observations, a is number of attribute categories
      # T: shape (N, t) where t is number of task categories
      A = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)
      T = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float)
      T_pred = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

      # Check which pairs exhibit bias
      bias_mask = ba_mals.check_bias(A, T)

      # Compute bias amplification
      bias_amp_combined, bias_amp = ba_mals.computeBiasAmp(A, T, T_pred)

DBA (Directional Bias Amplification)
--------------------------------------

Bias Amplification Metric that addresses shortcomings of BA_MALS by focusing on both positive and negative correlations, and the direction of amplification.

.. autoclass:: bias_amplification.metrics.CoOccurMetrics.DBA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   .. code-block:: python

      from bias_amplification.metrics.CoOccurMetrics import DBA
      import torch

      # Initialize DBA metric
      dba = DBA()

      # Prepare data
      A = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)
      T = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float)
      T_pred = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

      # Check statistical dependence (positive correlation)
      dependence_mask = dba.check_bias(A, T)

      # Compute bias amplification (handles both positive and negative correlations)
      bias_amp_combined, bias_amp = dba.computeBiasAmp(A, T, T_pred)

      # Compute bidirectional bias amplification
      A_pred = A.clone()  # Predicted attributes
      bias_amp_bidirectional = dba.computeBiasAmpBidirectional(
          A, A_pred, T, T_pred
      )
      # Returns dict with keys 'AtoT' and 'TtoA'

MDBA (Multi-Attribute Directional Bias Amplification)
--------------------------------------------

Multi-Attribute Directional Bias Amplification Metric that extends DBA to handle multi-dimensional attribute combinations, computing bias amplification across all possible attribute combinations.

.. autoclass:: bias_amplification.metrics.CoOccurMetrics.MDBA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   .. code-block:: python

      from bias_amplification.metrics.CoOccurMetrics import MDBA
      import torch

      # Initialize MDBA metric with attribute size constraints
      mdba = MDBA(min_attr_size=1, max_attr_size=3)

      # Prepare data with multiple attributes
      A = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)
      T = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.float)
      T_pred = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=torch.float)

      # Compute multi-dimensional bias amplification
      # Returns (mean, variance) tuple
      bias_amp_mean, bias_amp_variance = mdba.computeBiasAmp(A, T, T_pred)

      # Get statistics about attribute combinations
      stats = mdba.getAttributeCombinationStats(T)
      # Returns dict with 'total_combinations', 'by_size', 'examples'

      # Compute bidirectional bias amplification
      A_pred = A.clone()
      bias_amp_bidirectional = mdba.computeBiasAmpBidirectional(
          A, A_pred, T, T_pred
      )
      # Returns dict with keys 'AtoT' and 'TtoA', each containing (mean, variance)

