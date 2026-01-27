import os
import sys
import pickle
from typing import Union, Literal
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
# from bias_amplification.text.attacker_models import LSTM_ANN_Model
# Type Hints
pathType = Union[str, os.PathLike]
boolNum = Literal[0, 1]


def read_pkl_file(file_path: pathType) -> list[dict]:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data

if __name__ == "__main__":
    # Use absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    HUMAN_ANN_PATH = os.path.join(script_dir, "gender_obj_cap_mw_entries.pkl")
    MODEL_ANN_PATH = os.path.join(script_dir, "gender_val_transformer_cap_mw_entries.pkl")
    human_ann = read_pkl_file(HUMAN_ANN_PATH)
    model_ann = read_pkl_file(MODEL_ANN_PATH)
    human_ann_0 = human_ann
    model_ann_0 = model_ann
    # human_ann_0 = human_ann[0:5]
    # model_ann_0 = model_ann[0:5]
    human_ann_obj = {"img_id": [], "caption": []}
    model_ann_obj = {"img_id": [], "caption": []}
    attribute_data = {"img_id": [], "gender": [], "objects": []}

    for h in human_ann_0:
        human_ann_obj["img_id"].append(h['img_id'])
        human_ann_obj["caption"].append(h['caption_list'])
        attribute_data["img_id"].append(h['img_id'])
        attribute_data["gender"].append(h['bb_gender'])
        attribute_data["objects"].append(h['rmdup_object_list'])
    for m in model_ann_0:
        model_ann_obj["img_id"].append(m['img_id'])
        model_ann_obj["caption"].append(m['pred'])

    human_df = pd.DataFrame(human_ann_obj)
    model_df = pd.DataFrame(model_ann_obj)
    attribute_data_df = pd.DataFrame(attribute_data)
    print("-"*100)
    print("Human DataFrame:")
    print(human_df.head())
    print("-"*100)
    print("Model DataFrame:")
    print(model_df.head())
    print("-"*100)
    print("Attribute Data DataFrame:")
    print(attribute_data_df.head())
    mlb = MultiLabelBinarizer()
    objects = mlb.fit_transform(attribute_data_df['objects'])
    object_presence_df = pd.DataFrame(objects, columns=mlb.classes_, index=attribute_data_df['img_id'])
    print("-"*100)
    print("Object Presence DataFrame:")
    print(object_presence_df.head())

    attribute_data_df["gender"] = (
            1 * attribute_data_df["gender"] == "Male"
        ).astype(
            int
        ) 
    print("-"*100)
    print("Attribute Data DataFrame with 0/1 Gender:")
    print(attribute_data_df.head())

    human_merged =  human_df.merge(
        attribute_data_df.drop("objects", axis=1), on="img_id", how="left"
    )
    model_merged = model_df.merge(
        attribute_data_df.drop("objects", axis=1), on="img_id", how="left"
    )
    print("-"*100)
    print("Human DataFrame with 0/1 Gender:")
    print(human_merged.head())
    print("-"*100)
    print("Model DataFrame with 0/1 Gender:")
    print(model_merged.head())

    combined_df = human_merged.merge(
            model_merged[["img_id", "caption"]],
            on="img_id",
            suffixes=["_human", "_model"],
        )
    print("-"*100)
    print("Combined DataFrame:")
    print(combined_df.head())

