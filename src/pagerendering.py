
""" Creates a sentiment analysis App using Taipy"""
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

import numpy as np
import pandas as pd
from taipy.gui import Gui, notify



MODEL = "sbcBI/sentiment_analysis_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

dataframe = pd.DataFrame(
    {
        "Text": [""],
        "Score Pos": [0.33],
        "Score Neu": [0.33],
        "Score Neg": [0.33],
        "Overall": [0],
    }
)

dataframe2 = dataframe.copy()


def analyze_text(input_text: str) -> dict:
    """
    Runs the sentiment analysis model on the text

    Args:
        - text (str): text to be analyzed

    Returns:
        - dict: dictionary with the scores
    """
    encoded_text = tokenizer(input_text, return_tensors="pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    return {
        "Text": input_text[:50],
        "Score Pos": scores[2],
        "Score Neu": scores[1],
        "Score Neg": scores[0],
        "Overall": scores[2] - scores[0],
    }


def local_callback(state) -> None:
    """
    Analyze the text and updates the dataframe

    Args:
        - state: state of the Taipy App
    """
    notify(state, "Info", f"The text is: {state.text}", True)
    temp = state.dataframe.copy()
    scores = analyze_text(state.text)
    state.dataframe = temp.append(scores, ignore_index=True)
    state.text = ""


path = ""
treatment = 0

# Attendance page
page_file = """

<|Table|expandable|
<|{dataframe2}|table|width=100%|number_format=%.2f|>
|>

<br/>

<|{dataframe2}|chart|type=bar|x=Text|y[1]=Score Pos|y[2]=Score Neu|y[3]=Score Neg|y[4]=Overall|color[1]=green|color[2]=grey|color[3]=red|type[4]=line|height=600px|>

"""


webcam_md = """<|toggle|theme|>

<container|container|part|

# Welcome, look at the camera to **Sign In**{: .color-primary}


<br/>

<card|card p-half|part|
## **Webcam**{: .color-primary} component

<|text-center|part|
<|webcam.Webcam|faces={labeled_faces}|classname=face_detector|id=my_face_detector|on_data_receive=handle_image|sampling_rate=100|>

<|Capture|button|on_action={lambda s: s.assign("capture_image", True)}|>
<|RE-train|button|on_action=button_retrain_clicked|>
>
|card>
|container>


<|{show_capture_dialog}|dialog|labels=Validate;Cancel|on_action=on_action_captured_image|title=Add new training image|
<|{captured_image}|image|width=300px|height=300px|>

<|{captured_label}|input|>
|>
"""


def analyze_file(state) -> None:
    """
    Analyse the lines in a text file

    Args:
        - state: state of the Taipy App
    """
    state.dataframe2 = dataframe2
    state.treatment = 0
    with open(state.path, "r", encoding="utf-8") as f:
        data = f.read()
        print(data)

        file_list = list(data.split("\n"))

    for i, input_text in enumerate(file_list):
        state.treatment = int((i + 1) * 100 / len(file_list))
        temp = state.dataframe2.copy()
        scores = analyze_text(input_text)
        print(scores)
        state.dataframe2 = temp.append(scores, ignore_index=True)

    state.path = None

# intergrating pages urls
pages = {
    "/": "<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
    "Camera": webcam_md,
    "Attendance": page_file,
}

# rending pages with named tab title
Gui(pages=pages).run(title="Facial Recognition Attendance")