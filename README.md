# ICC-SML
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

An application of ICanCode - Hackcovy20 submission

SML: Smart eMeeting & eLearning System

# Description
Our project includes 2 main components:
1. AI Service: a Flask application that currently serves following features:
+ Face recognition
+ Harassment prevention by automatically blurring the unknown people (have never registered with face ID to the learning system)
+ Fatigue detection: detect whether the listener is tired, sleepy
+ Attention Analysis: Analyze attention level of viewer by tracking eyes gaze.
+ Emotion Analysis: Analyze the reaction level (positive/negative) of the viewer based on emotions (Happy, Neutral, Sad)

2. Learning Interface: We will publish this as open source later.

# Link model

Get ML model from [link](https://drive.google.com/file/d/1w3saxS8RLuwsheWvOXR2AhMJKP_2v1PW/view?usp=sharing)

# Prepare environment
```
$ virtualenv -p python3.6 python3.6
$ pip install -r requirements.txt
$ source python3.6/bin/active
```
# Run
```
$ python face_api.py
```

# DEMO

[Harassment prevention](https://drive.google.com/file/d/19rC4sli6zkylduSiSHubo3JTn7NdJOb4/view)

[Face recognition](https://drive.google.com/file/d/1qbchRqN5PVrvm40vgwofU5oQn1zxBcPj/view)

[Fatigue detection](https://drive.google.com/file/d/19SgNCdk8IRv8rK-mo9xaG1Hr87PhxI0v/view)

[Attention analysis](https://drive.google.com/file/d/1I8E_1JCE5oD4lr1xc2YijEGMXSUAQHH-/view)

[Emotion analysis](https://drive.google.com/file/d/1UbYNIJh3ZceSDasynRbTI-ItGtn8PvUY/view)
