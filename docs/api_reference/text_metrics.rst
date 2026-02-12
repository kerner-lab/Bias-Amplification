Captioning Bias Amplification Metrics
======================================

.. note::
   The text processing modules require optional dependencies.
   Install with: ``pip install bias-amplification[text]``

   Required packages:
   - torchtext: For text tokenization
   - transformers: For BERT embeddings
   - gensim: For GloVe embeddings
   - sentence-transformers: For sentence embeddings
   - nltk: For NLTK tokenizer
   - tqdm: For progress bars

The text processing modules provide functionality for measuring bias amplification
in text-based machine learning models, particularly for image captioning systems.

BiasMetricBase (Base class for bias metrics)
---------------------------------------------

.. autoclass:: bias_amplification.text.metrics.BiasMetricBase
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__


LIC (Leakage In Captioning)
----------------------------------

.. autoclass:: bias_amplification.text.metrics.LIC
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   .. code-block:: python

      from bias_amplification.text.metrics import LIC
      import torch
      import pandas as pd

      # Initialize LIC metric
      lic = LIC(
          model_params={
              "attacker_class": LSTM_ANN_Model,
              "attacker_params": {...}
          },
          train_params={
              "learning_rate": 0.001,
              "loss_function": "mse",
              "epochs": 50,
              "batch_size": 32
          },
          gender_words=["man", "woman", "male", "female"],
          obj_words=["guitar", "ball", "tennis"],
          gender_token="<gender>",
          obj_token="<obj>",
          model_path="path/to/glove.6B.50d.txt",
          model_type="glove"
      )

      # Calculate amortized leakage
      result = lic.getAmortizedLeakage(
          feat=features,
          data=ground_truth,
          pred=predictions,
          num_trials=25
      )

DBAC (Directional Bias Amplification in Captioning)
---------------------------------------------

.. autoclass:: bias_amplification.text.metrics.DBAC
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   .. code-block:: python

      from bias_amplification.text.metrics import DBAC

      # Initialize DBAC metric
      dbac = DBAC(
          model_params={...},
          train_params={...},
          gender_words=[...],
          obj_words=[...],
          gender_token="<gender>",
          obj_token="<obj>",
          glove_path="path/to/glove.6B.50d.txt",
          sub_model="glove"
      )

      # Calculate leakage
      leakage = dbac.calcLeak(
          feat=features,
          data=ground_truth,
          pred=predictions,
          mask_mode="gender"
      )