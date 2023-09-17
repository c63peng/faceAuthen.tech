import numpy as np
import pandas as pd
from taipy.gui import Gui, notify
import json 

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

with open("src/Attendance.json", 'r') as file:
    attendees = json.load(file)
    names = [] 
    for person in attendees:
        names.append(person["name"])

    attends = []
    for person in attendees:
        attends.append(person["attendance"])

    time_stamp =[]
    for person in attendees:
        time_stamp.append(person["attendanceTime"])

dataframe = pd.DataFrame(
    {
        "Name": names,
        "Attendance": attends,
        "Time-Stamp": time_stamp,
    }
)


path = ""
treatment = 0

# Attendance page
page_file = """

<|Persons|expandable|
<|{dataframe}|table|width=100%|number_format=%.2f|>
|>

"""

pages = {
    "/": "<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
    "Camera": webcam_md,
    "Attendance": page_file,
}

# rending pages with named tab title
Gui(pages=pages).run(title="Facial Recognition Attendance")