# main.py

from taipy.gui import Gui
from pagerendering import webcam_md,dataframe, page_file, names, attends, time_stamp
from facialRecognition import recognize_faces

# Define Taipy components and pages
# ... (Define Taipy components and pages as you did in pagesrendered.py)

# Create the Taipy GUI
pages = {
    "/": "<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
    "Camera": webcam_md,
    "Attendance": page_file,
}

# rending pages with named tab title

gui = Gui(pages=pages)
taipy_state = gui.state

# Define a callback to update the Taipy DataFrame with attendance data
def update_taipy_dataframe():
    taipy_state.dataframe = dataframe

# Add a callback to start the face recognition component
def start_face_recognition():
    recognize_faces()

# Add a callback to update the Taipy DataFrame when the Attendance page is shown
gui.add_callback(update_taipy_dataframe, when="Attendance")

# Add a callback to start the face recognition component when the webcam page is shown
gui.add_callback(start_face_recognition, when="show_capture_dialog")

# Run the Taipy GUI
Gui(pages=pages).run(title="Facial Recognition Attendance")
