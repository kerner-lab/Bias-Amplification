MASCULINE = [
        "man",
        "men",
        "male",
        "father",
        "gentleman",
        "boy",
        "uncle",
        "husband",
        "actor",
        "prince",
        "waiter",
        "he",
        "his",
        "him",
    ]
FEMININE = [
        "woman",
        "women",
        "female",
        "mother",
        "lady",
        "girl",
        "aunt",
        "wife",
        "actress",
        "princess",
        "waitress",
        "she",
        "her",
        "hers",
    ]

DEFAULT_CONFIG = {
    "NUM_TRIALS": 10,
    "METHOD": "mean",
    "NORMALIZED": False,
    "MASK_MODE": "gender",
    "MASK_TYPE": "constant",
    "APPLY_BAYES": False,
    "EVAL_METRIC": "mse",
    "DEVICE": "cpu",
    "MODEL_TYPE": "glove",
    "GENDER_WORDS": MASCULINE + FEMININE,
    "OBJ_WORDS": [],
    "GENDER_TOKEN": "<unk>",
    "OBJ_TOKEN": "<obj>",
}

LIC_CONFIG = {
    "SIMILARITY_THRESHOLD": 1,
    "MASK_TYPE": "contextual",
}

DBAC_CONFIG = {
    "SIMILARITY_THRESHOLD": 0.5,
    "MASK_TYPE": "contextual",
}