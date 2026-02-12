Utilities for Captioning Bias Amplification
============================================

CaptionProcessor
----------------

.. autoclass:: bias_amplification.text.utils.text.CaptionProcessor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Examples

   .. code-block:: python

      from bias_amplification.text.utils.text import CaptionProcessor

      # Initialize processor with GloVe
      processor = CaptionProcessor(
          gender_words=["man", "woman"],
          obj_words=["guitar", "ball"],
          glove_path="path/to/glove.6B.50d.txt",
          model_type="glove"
      )

      # Tokenize captions
      tokens = processor.apply_tokenizer(["A man plays guitar"])

      # Build vocabulary
      vocab = processor.build_vocab(captions)

      # Equalize vocabularies
      human_eq, model_eq = processor.equalize_vocab(
          human_captions,
          model_captions,
          similarity_threshold=0.5
      )