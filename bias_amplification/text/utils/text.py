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
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import json

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
        device=torch.device("cpu"),
    ) -> None:
        """
        Initialize the CaptionProcessor.
        This class is used to pre process the model captions and ground truth captions for bias amplification.
        It provides functionality to tokenize the captions, build the vocabulary, and equalize the vocabularies.
        """
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
        self.default_dim_model = None
        self.sbert_model = None
        self.device = device

        if model_type == "glove":
            self.glove_model = self.load_glove_model(model_path) if model_path else None
            self.default_dim_model = 50
        elif model_type == "fasttext":
            self.fasttext_model = fasttext.load_model(model_path) if model_path else None
            self.default_dim_model = 300
        elif model_type == "bert":
            # bert_model="sentence-transformers/all-MiniLM-L6-v2"
            bert_model="bert-base-uncased"
            print(f"Loading BERT model: {bert_model}...")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
            self.bert_model = AutoModel.from_pretrained(bert_model)
            self.bert_model.eval()
            self.bert_model = self.bert_model.to(self.device)
            self.default_dim_model = 768
        elif model_type == "sbert":
            sbert_model="all-MiniLM-L6-v2"
            print(f"Loading SBERT model: {sbert_model}...")
            self.sbert_model = SentenceTransformer(sbert_model)
            self.sbert_model.eval()
            self.sbert_model = self.sbert_model.to(self.device)
            self.default_dim_model = 384
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    @staticmethod
    def load_glove_model(glove_path):
        """
        Load the GloVe model from the given path.
        """
        return KeyedVectors.load_word2vec_format(glove_path, binary=False)

    def apply_tokenizer(
        self, text_obj: Union[list[str], pd.Series]
    ) -> Union[list[list[str]], pd.Series]:
        """
        Tokenize the text object.
        """
        if isinstance(text_obj, pd.Series):
            return text_obj.apply(self.tokenize)
        return [self.tokenize(text) for text in text_obj]

    def build_vocab(self, text_obj: Union[list[str], pd.Series]):
        """
        Build the vocabulary from the text object.
        """
        vocab = build_vocab_from_iterator(self.apply_tokenizer(text_obj))
        return vocab

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize the text.
        """
        tokens = self.tokenizer(text)
        return [token for token in tokens if token not in self.stopwords]

    def tokens_to_numbers(self, vocab, text_obj: Union[list[str], pd.Series], pad_value: int = 0):
        """
        Convert the tokens to numbers.
        """
        sequence = numericalize_tokens_from_iterator(vocab, self.apply_tokenizer(text_obj))
        token_ids = [list(next(sequence)) for _ in range(len(text_obj))]
        return pad_sequence(
            [torch.tensor(x) for x in token_ids],
            batch_first=True,
            padding_value=pad_value,
        )

    def mask_words(
        self,
        string_list: Union[list[str], pd.Series],
        mode: Literal["gender", "object"] = "gender",
        object_presence_df: pd.DataFrame = None,
        img_id: int = None,
    ) -> Union[list[str], pd.Series]:
        """
        Mask the words in the string list based on the specified mode.
        
        Parameters
        ----------
        string_list : list[str]
            List of strings to mask.
        mode : Literal["gender", "object"], optional
            The mode to mask the words in. Default is "gender".
        object_presence_df : pd.DataFrame, optional
            The dataframe containing the object presence information. Default is None.
        img_id : int, optional
            The image id to use for the object presence information. Default is None.

        Returns
        -------
        list[str] or pd.Series
            The masked strings.
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

    # def get_embedding_dim(self):
    #     """
    #     Get the embedding dimension for the current model.
    #     Returns the dimension size (e.g., 300 for GloVe, 768 for BERT-base).
    #     """
    #     if self.model_type == "glove":
    #         if self.glove_model is not None:
    #             # Get dimension from GloVe model
    #             return self.glove_model.vector_size
    #         else:
    #             # Fallback: try to get from a sample word
    #             sample_vec = self.get_token_vector("the", None)
    #             if sample_vec is not None:
    #                 return sample_vec.shape[0]
    #             return 300  # Default GloVe dimension
    #     elif self.model_type == "fasttext":
    #         if self.fasttext_model is not None:
    #             return self.fasttext_model.get_dimension()
    #         else:
    #             sample_vec = self.get_token_vector("the", None)
    #             if sample_vec is not None:
    #                 return sample_vec.shape[0]
    #             return 300  # Default FastText dimension
    #     elif self.model_type == "bert":
    #         if self.bert_model is not None:
    #             # Get dimension from BERT model config
    #             return self.bert_model.config.hidden_size
    #         else:
    #             # Fallback: try to get from a sample token
    #             sample_vec = self.get_token_vector("the", None)
    #             if sample_vec is not None:
    #                 return sample_vec.shape[0]
    #             return 768  # Default BERT-base dimension
    #     else:
    #         raise ValueError(f"Unknown model_type: {self.model_type}")


    def get_tokenized_vectors(self, tokens):
        """
        Return embedding vector for a token.
        - For GloVe and fasttext: simple lookup
        - For BERT or SBERT: token in isolation
        """
        embeddings = []
        if self.model_type == "glove":
            for token in tokens:
                if self.glove_model and token in self.glove_model:
                    embeddings.append(torch.tensor(self.glove_model[token]))
                else:
                    embeddings.append(torch.zeros(self.default_dim_model))
            embeddings = torch.stack(embeddings)
        elif self.model_type == "fasttext":
            for token in tokens:
                if self.fasttext_model:
                    embeddings.append(torch.tensor(self.fasttext_model.get_word_vector(token)))
                else:
                    embeddings.append(torch.zeros(self.default_dim_model))
            embeddings = torch.stack(embeddings)
        elif self.model_type == "bert":
            cap_inputs = self.bert_tokenizer(tokens, return_tensors="pt", padding=True, add_special_tokens=False)
            cap_inputs = {k: v.to(self.device) for k, v in cap_inputs.items()}
            with torch.no_grad():
                outputs = self.bert_model(**cap_inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        elif self.model_type == "sbert":
            embeddings = self.sbert_model.encode(tokens, convert_to_tensor=True)
        return embeddings

    def get_tokenized_vectors_sbert_contextual(self, sentence_tokens):
        """
        Return embedding vector for a token.
        - For SBERT: token in context_sentence
        """
        sentence = " ".join(sentence_tokens)
        sentence_embeddings = self.sbert_model.encode(sentence, convert_to_tensor=True, output_value="token_embeddings")
        return sentence_embeddings

    def get_tokenized_vectors_bert_contextual(self, corpus_tokens, sentence_tokens):
        """
        Return embedding vector for a token.
        - For BERT: token in context_sentence
        """
        corpus_embeddings = []
        sentence_embeddings = []
        sentence = " ".join(sentence_tokens)
        inputs = self.bert_tokenizer(sentence, return_tensors="pt",padding=True,truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        hidden_states = outputs.last_hidden_state.squeeze(0)
        tokens = self.bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        for token in corpus_tokens:
            if token in tokens:
                corpus_embeddings.append(hidden_states[tokens.index(token)])
            else:
                corpus_embeddings.append(torch.zeros(self.default_dim_model))
        for cap_token in sentence_tokens:
            if cap_token in tokens:
                sentence_embeddings.append(hidden_states[tokens.index(cap_token)])
            else:
                sentence_embeddings.append(torch.zeros(self.default_dim_model))
        corpus_embeddings = torch.stack(corpus_embeddings)
        sentence_embeddings = torch.stack(sentence_embeddings)

        return corpus_embeddings, sentence_embeddings

    def get_token_vector(self, token, context_sentence=None):
        """
        Get the embedding vector for a token.

        Parameters
        ----------
        token : str
            The token to get the embedding vector for.
        context_sentence : str, optional
            The context sentence to use for the BERT model. Default is None.

        Returns
        -------
        torch.tensor or None
            The embedding vector for the token.
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

        Parameters
        ----------
        human_captions : list[str]
            List of human captions.
        model_captions : list[str]
            List of model captions.
        similarity_threshold : float, optional
            Similarity threshold for vocabulary equalization. Default is 0.5.
        maskType : str, optional
            Type of mask to apply. Default is "contextual".
        bidirectional : bool, optional
            Whether to equalize the vocabularies bidirectionally. Default is False.

        Returns
        -------
        tuple[list[list[str]], list[list[str]]]
            Equalized human captions and model captions.
        """

        human_tokens = [self.tokenize(caption) for caption in human_captions]
        model_tokens = [self.tokenize(caption) for caption in model_captions]

        # Flatten corpora into sets
        machine_corpus = set([token for tokens in model_tokens for token in tokens])
        human_corpus = set([token for tokens in human_tokens for token in tokens])

        machine_corpus_list = list(machine_corpus)
        human_corpus_list = list(human_corpus)

        def equalize_caption(caption_tokens, corpus_list, corpus_embeddings=None):

            if corpus_embeddings is None:
                corpus_embeddings, sentence_embeddings = self.get_tokenized_vectors_bert_contextual(corpus_list, caption_tokens)
            elif self.model_type == "sbert" and maskType == "contextual":
                sentence_embeddings = self.get_tokenized_vectors_sbert_contextual(caption_tokens)
            else:
                sentence_embeddings = self.get_tokenized_vectors(caption_tokens)

            cosine_scores = util.cos_sim(sentence_embeddings, corpus_embeddings)
            equalized_sent_toks = []
            for i, token in enumerate(caption_tokens):
                token = token.lower()
                if token in corpus_list:
                    equalized_sent_toks.append(token)
                    continue

                max_similarity, best_idx = torch.max(cosine_scores[i], dim=0)
                sim_threshold = 0.3 if self.model_type == "sbert" else similarity_threshold
                if max_similarity >= sim_threshold:
                    target_word = corpus_list[best_idx.item()]
                    equalized_sent_toks.append(target_word)
                else:
                    equalized_sent_toks.append("unk")
            return " ".join(equalized_sent_toks)

        
        
        corpus_embedding = None
        if maskType != "bert":
            corpus_embedding = self.get_tokenized_vectors(machine_corpus_list)
        equalized_human = [
            equalize_caption(human_cap, machine_corpus_list, corpus_embedding)
            for human_cap in tqdm(human_tokens, desc="Equalizing Human Captions")
        ]
        if bidirectional:
            if maskType != "bert":
                corpus_embedding = self.get_tokenized_vectors(human_corpus_list)
            equalized_model = [
                equalize_caption(model_cap, human_corpus_list, corpus_embedding)
                for model_cap in tqdm(model_tokens, desc="Equalizing Model Captions")
            ]
        else:
            equalized_model = [" ".join(cap) for cap in model_tokens]
        

        return equalized_human, equalized_model


def cmpVocab(vocab1, vocab2):
    """
    Compare the vocabularies of two caption processors.
    """
    # Compatible with both old and new torchtext versions
    set1 = set(vocab1.get_stoi().keys() if hasattr(vocab1, 'get_stoi') else vocab1.stoi.keys())
    set2 = set(vocab2.get_stoi().keys() if hasattr(vocab2, 'get_stoi') else vocab2.stoi.keys())

    common_tokens = set1 & set2
    only_in_vocab1 = set1 - set2
    only_in_vocab2 = set2 - set1
    print(
        f"Common_tokens : {len(common_tokens)}, vocab_1_exc: {len(only_in_vocab1)}, vocab_2_exc: {len(only_in_vocab2)}"
    )


# CLI
def get_parser():
    """
    Get the parser for the CaptionProcessor CLI.
    """
    parser = argparse.ArgumentParser(description="CaptionProcessor CLI")
    parser.add_argument("--tokenizer", default="nltk", choices=["nltk", "spacy"])
    parser.add_argument("--mode", default="gender", choices=["gender", "object"])
    parser.add_argument("--glove_path", required=True)
    parser.add_argument("--output_folder", default="output")
    parser.add_argument("--similarity_threshold", type=float, default=0.5)
    return parser
