
""" Creates a sentiment analysis App using Taipy"""
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

import numpy as np
import pandas as pd
from taipy.gui import Gui, notify

webcam_md = """<|toggle|theme|>

<container|container|part|

# Welcome, look at the camera to **Sign In**{: .color-primary}


<br/>

<card|card p-half|part|
## **Webcam**{: .color-primary} component


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


MODEL = "sbcBI/sentiment_analysis_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

dataframe = pd.DataFrame(
    {
        "Persons": [""],
        "Name": [""],
        "Clearance Level": [""],
        "Attendance": [""],
    }
)





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
<|{dataframe}|table|width=100%|number_format=%.2f|>
|>

"""


def analyze_file(state) -> None:
    """
    Analyse the lines in a text file

    Args:
        - state: state of the Taipy App
    """
    state.dataframe = dataframe
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
        state.dataframe = temp.append(scores, ignore_index=True)

    state.path = None

# intergrating pages urls
pages = {
    "/": "<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
    "Camera": webcam_md,
    "Attendance": page_file,
}

# rending pages with named tab title
Gui(pages=pages).run(title="Facial Recognition Attendance")