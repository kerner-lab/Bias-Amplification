import sys
import os
import pandas as pd
import numpy as np
import torch
import argparse
from typing import Union, Literal
from gensim.models import KeyedVectors
import fasttext
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import numericalize_tokens_from_iterator
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# torchtext.disable_torchtext_deprecation_warning()


# Text Processor Class
class CaptionProcessor:
    def __init__(
        self,
        gender_words,
        obj_words,
        model_path=None,
        gender_token="gender",
        obj_token="obj",
        stopwords=[".", ",", " "],
        tokenizer="basic_english",
        lang="en",
        model_type="glove",
        bert_model="bert-base-uncased",  # for BERT
    ) -> None:
        if tokenizer == "nltk":
            from nltk.tokenize import NLTKWordTokenizer

            self.tokenizer = NLTKWordTokenizer().tokenize
        else:
            self.tokenizer = get_tokenizer(tokenizer, lang)
        self.stopwords = stopwords
        self.gender_words = gender_words
        self.gender_token = gender_token
        self.object_words = obj_words
        self.object_token = obj_token
        self.model_type = model_type
        self.bert_tokenizer = None
        self.bert_model = None
        self.glove_model = None
        self.fasttext_model = None
        if model_type == "glove":
            self.glove_model = self.load_glove_model(model_path) if model_path else None
        elif model_type == "fasttext":
            self.fasttext_model = fasttext.load_model(model_path) if model_path else None
        elif model_type == "bert":
            print(f"Loading BERT model: {bert_model}...")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
            self.bert_model = AutoModel.from_pretrained(bert_model)
            self.bert_model.eval()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    @staticmethod
    def load_glove_model(glove_path):
        return KeyedVectors.load_word2vec_format(glove_path, binary=False)

    def apply_tokenizer(
        self, text_obj: Union[list[str], pd.Series]
    ) -> Union[list[list[str]] | pd.Series]:
        if isinstance(text_obj, pd.Series):
            return text_obj.apply(self.tokenize)
        return [self.tokenize(text) for text in text_obj]

    def build_vocab(self, text_obj: Union[list[str], pd.Series]):
        vocab = build_vocab_from_iterator(self.apply_tokenizer(text_obj))
        return vocab

    def tokenize(self, text: str) -> list[str]:
        tokens = self.tokenizer(text)
        return [token for token in tokens if token not in self.stopwords]

    def tokens_to_numbers(self, vocab, text_obj: Union[list[str], pd.Series], pad_value: int = 0):
        sequence = numericalize_tokens_from_iterator(vocab, self.apply_tokenizer(text_obj))
        token_ids = [list(next(sequence)) for _ in range(len(text_obj))]
        return pad_sequence(
            [torch.tensor(x) for x in token_ids],
            batch_first=True,
            padding_value=pad_value,
        )

    def maskWords(
        self,
        string_list: Union[list[str], pd.Series],
        mode: Literal["gender", "object"] = "gender",
        object_presence_df: pd.DataFrame = None,
        img_id: int = None,
    ) -> Union[list[str], pd.Series]:
        """
        Mask words based on the specified mode:
        - "gender": Masks gender words with self.gender_token.
        - "object": Masks object words with self.object_token if present in object_presence_df.
        """
        if mode not in ["gender", "object"]:
            raise ValueError("Expected mode to be 'gender' or 'object'")

        words_to_mask = self.gender_words if mode == "gender" else self.object_words
        mask_token = self.gender_token if mode == "gender" else self.object_token
        masked_strings = [
            " ".join([mask_token if token in words_to_mask else token for token in self.tokenize(string)])
            for string in string_list
        ]
        return masked_strings

    def get_embedding_dim(self):
        """
        Get the embedding dimension for the current model.
        Returns the dimension size (e.g., 300 for GloVe, 768 for BERT-base).
        """
        if self.model_type == "glove":
            if self.glove_model is not None:
                # Get dimension from GloVe model
                return self.glove_model.vector_size
            else:
                # Fallback: try to get from a sample word
                sample_vec = self.get_token_vector("the", None)
                if sample_vec is not None:
                    return sample_vec.shape[0]
                return 300  # Default GloVe dimension
        elif self.model_type == "fasttext":
            if self.fasttext_model is not None:
                return self.fasttext_model.get_dimension()
            else:
                sample_vec = self.get_token_vector("the", None)
                if sample_vec is not None:
                    return sample_vec.shape[0]
                return 300  # Default FastText dimension
        elif self.model_type == "bert":
            if self.bert_model is not None:
                # Get dimension from BERT model config
                return self.bert_model.config.hidden_size
            else:
                # Fallback: try to get from a sample token
                sample_vec = self.get_token_vector("the", None)
                if sample_vec is not None:
                    return sample_vec.shape[0]
                return 768  # Default BERT-base dimension
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def get_token_vector(self, token, context_sentence=None):
        """
        Return embedding vector for a token.
        - For GloVe: simple lookup
        - For BERT: token in context_sentence (if provided), else static
        """
        if self.model_type == "glove":
            if self.glove_model and token in self.glove_model:
                return torch.tensor(self.glove_model[token])
            return None
        elif self.model_type == "fasttext":
            if self.fasttext_model:
                return torch.tensor(self.fasttext_model.get_word_vector(token))
            return None
        elif self.model_type == "bert":
            # Use context if provided
            if context_sentence:
                inputs = self.bert_tokenizer(context_sentence, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                tokens = self.bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                hidden_states = outputs.last_hidden_state.squeeze(0)
                try:
                    idx = tokens.index(token)
                    return hidden_states[idx]
                except ValueError:
                    return None
            else:
                # Encode token in isolation
                inputs = self.bert_tokenizer(token, return_tensors="pt", add_special_tokens=False)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).squeeze(0)


    def equalize_vocab(
        self,
        human_captions,
        model_captions,
        similarity_threshold=0.5,
        maskType="contextual",
        bidirectional=False,
    ):
        """
        Equalize captions using embeddings (GloVe or BERT).
        Preserves structure of tokenized captions.
        """

        human_tokens = [self.tokenize(caption) for caption in human_captions]
        model_tokens = [self.tokenize(caption) for caption in model_captions]

        # Flatten corpora into sets
        machine_corpus = set([token for tokens in model_tokens for token in tokens])
        human_corpus = set([token for tokens in human_tokens for token in tokens])

        machine_corpus_list = list(machine_corpus)
        human_corpus_list = list(human_corpus)

        # Get embedding dimension before processing
        embedding_dim = self.get_embedding_dim()
        # Create a zero vector template for missing embeddings
        zero_vec_template = torch.zeros(embedding_dim)

        def substitute_token(token, corpus_list, corpus_embeddings, context_sentence=None):
            token = token.lower()
            if token in corpus_list:
                return token

            token_vec = self.get_token_vector(token, context_sentence)
            if token_vec is None:
                return "unk"

            similarities = torch.nn.functional.cosine_similarity(
                token_vec.unsqueeze(0), corpus_embeddings, dim=1
            )
            max_similarity, best_idx = torch.max(similarities, dim=0)

            if max_similarity >= similarity_threshold and maskType == "contextual":
                return corpus_list[best_idx.item()]
            return "unk"

        def equalize_caption(caption_tokens, corpus_list):
            """
            Process a caption: pre-compute corpus embeddings once, then process all tokens.
            """
            context_sentence = " ".join(caption_tokens)
            
            # Pre-compute corpus embeddings ONCE for a caption
            corpus_embeddings = []
            for t in corpus_list:
                vec = self.get_token_vector(t, context_sentence)
                if vec is not None:
                    corpus_embeddings.append(vec.unsqueeze(0))
                else:
                    # Use pre-determined zero vector template
                    corpus_embeddings.append(zero_vec_template.unsqueeze(0))
            corpus_embeddings_tensor = torch.cat(corpus_embeddings, dim=0)

            return " ".join([
                substitute_token(
                    tok,
                    corpus_list,
                    corpus_embeddings_tensor,
                    context_sentence
                )
                for tok in caption_tokens
            ])


        # Equalize human captions
        equalized_human = [
            equalize_caption(human_cap, machine_corpus_list)
            for human_cap in tqdm(human_tokens, desc="Equalizing Human Captions")
        ]

        # Equalize model captions if bidirectional
        if bidirectional:
            equalized_model = [
                equalize_caption(model_cap, human_corpus_list)
                for model_cap in tqdm(model_tokens, desc="Equalizing Model Captions")
            ]
        else:
            equalized_model = [" ".join(cap) for cap in model_tokens]

        return equalized_human, equalized_model


def cmpVocab(vocab1, vocab2):
    set1 = set(vocab1.stoi.keys())
    set2 = set(vocab2.stoi.keys())

    common_tokens = set1 & set2
    only_in_vocab1 = set1 - set2
    only_in_vocab2 = set2 - set1
    print(
        f"Common_tokens : {len(common_tokens)}, vocab_1_exc: {len(only_in_vocab1)}, vocab_2_exc: {len(only_in_vocab2)}"
    )


# CLI
def get_parser():
    parser = argparse.ArgumentParser(description="CaptionProcessor CLI")
    parser.add_argument("--tokenizer", default="nltk", choices=["nltk", "spacy"])
    parser.add_argument("--mode", default="gender", choices=["gender", "object"])
    parser.add_argument("--glove_path", required=True)
    parser.add_argument("--output_folder", default="output")
    parser.add_argument("--similarity_threshold", type=float, default=0.5)
    return parser
