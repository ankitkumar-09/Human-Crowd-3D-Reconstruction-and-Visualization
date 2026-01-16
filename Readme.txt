1.Create environment and install libs

Open terminal in your project folder.

Create and activate venv (if not already):

python -m venv venv
venv\Scripts\activate 

2.Install requirements:

pip install ultralytics opencv-python pandas matplotlib

3.Place video and YOLO weights:
Put your crowd video in videos/ and update VIDEO_PATH in crowd_main.py to the correct file path.
Make sure yolov8n.pt is in the project root and the model line is:

model = YOLO("yolov8n.pt")

4.Run the main script
From the project folder:

venv\Scripts\activate
python crowd_main.py

5.(Optional) Run 3D Python plot
python plot_3d.py

Unity Engine:

1.Prepare CSVs for Unity
Copy the generated ground_points.csv and frame_risk.csv from your Python project into your Unity project under: