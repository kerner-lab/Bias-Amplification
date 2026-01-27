# Importing Libraries
import os
import pickle
import numpy as np
import pandas as pd
from typing import Union, Literal
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer

# Type Hints
pathType = Union[str, os.PathLike]
boolNum = Literal[0, 1]


# Helper Function
def checkWordPresence(word: str, sentence: str) -> boolNum:
    if word in sentence.split(" "):
        return 1
    return 0


# Data Class
class CaptionGenderDataset:

    def __init__(self, human_ann_file: pathType, model_ann_file: pathType) -> None:
        self.human_ann_path = human_ann_file
        self.model_ann_path = model_ann_file
        print("Reading Annotation Files")
        self.human_data = self.read_pkl_file(self.human_ann_path)
        self.model_data = self.read_pkl_file(self.model_ann_path)
        print("Processing Annotation Data")
        self.processData()
        self.wnl = WordNetLemmatizer()

    @staticmethod
    def read_pkl_file(file_path: pathType) -> list[dict]:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data

    def processData(self) -> None:
        self.mlb = MultiLabelBinarizer()
        self.human_ann = {"img_id": [], "caption": []}
        self.model_ann = {"img_id": [], "caption": []}
        self.attribute_data = {"img_id": [], "gender": [], "objects": []}

        for item in self.human_data:
            img_id = item["img_id"]
            gender = item["bb_gender"]
            objects = item["rmdup_object_list"]
            captions = item["caption_list"]

            self.attribute_data["img_id"].append(img_id)
            self.attribute_data["gender"].append(gender)
            self.attribute_data["objects"].append(objects)

            self.human_ann["img_id"].extend([img_id] * len(captions))
            self.human_ann["caption"].extend(captions)

        for item in self.model_data:
            img_id = item["img_id"]
            caption = item["pred"]

            self.model_ann["img_id"].append(img_id)
            self.model_ann["caption"].append(caption)

        self.human_ann = pd.DataFrame(self.human_ann)
        self.model_ann = pd.DataFrame(self.model_ann)
        self.attribute_data = pd.DataFrame(
            self.attribute_data
        )  # Look for attribute_data instead of object_presence_df

        objs = self.mlb.fit_transform(self.attribute_data["objects"])
        self.object_presence_df = pd.DataFrame(
            objs, columns=self.mlb.classes_, index=self.attribute_data["img_id"]
        )

        self.attribute_data["gender"] = (
            1 * self.attribute_data["gender"] == "Male"
        ).astype(
            int
        )  # 1 represents Male

    def getData(self) -> list[pd.DataFrame]:
        human_merged = self.human_ann.merge(
            self.attribute_data.drop("objects", axis=1), on="img_id", how="left"
        )
        model_merged = self.model_ann.merge(
            self.attribute_data.drop("objects", axis=1), on="img_id", how="left"
        )
        return human_merged, model_merged

    def get_object_presence_df(self):
        # Return the object presence DataFrame
        return self.object_presence_df

    def getDataCombined(self) -> pd.DataFrame:
        human_merged, model_merged = self.getData()
        combined_df = human_merged.merge(
            model_merged[["img_id", "caption"]],
            on="img_id",
            suffixes=["_human", "_model"],
        )
        return combined_df

    def getLabelPresence(self, labels: list[str], captions: pd.Series) -> pd.DataFrame:
        new_labels = [self.wnl.lemmatize(item) for item in labels]
        new_captions = captions.apply(
            lambda x: " ".join([self.wnl.lemmatize(item) for item in x.split(" ")])
        )
        presence_df = pd.DataFrame()
        presence_df["caption"] = captions
        for label in new_labels:
            presence_df[label] = new_captions.apply(
                lambda sentence: checkWordPresence(label, sentence)
            )
        return presence_df


if __name__ == "__main__":
    # Use absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    HUMAN_ANN_PATH = os.path.join(script_dir, "gender_obj_cap_mw_entries.pkl")
    MODEL_ANN_PATH = os.path.join(script_dir, "gender_val_transformer_cap_mw_entries.pkl")
    data_obj = CaptionGenderDataset(HUMAN_ANN_PATH, MODEL_ANN_PATH)
    human_ann, model_ann = data_obj.getData()
    human_ann, model_ann = data_obj.getData()
    object_presence_df = data_obj.get_object_presence_df()

    # Optional - In case if you want to print the sample results
    print("Human Annotations Sample:\n", human_ann.head())
    print("Model Annotations Sample:\n", model_ann.head())
    print("Object Presence DataFrame:\n", object_presence_df.head())


# race_words = ['white', 'caucasian','black', 'african', 'asian', 'latino', 'latina', 'latinx','hispanic', 'native', 'indigenous']