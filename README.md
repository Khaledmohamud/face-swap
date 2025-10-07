# face-swap
A lightweight face swapping program

## what is it
face-swap is a silly project I made to demonstrate what can be done on a lightweight system that isn't the richest in resources. It is supposed to be a quoute on quote "alternative to deepfake technology". it fundamentally operates in 2 steps:
1. It takes a source image and maps out it's facial features for extraction.
2. it then mapsout the same features on the target face to get facial expressions
from there it rips the features off the source image and plasters them over the target image making one uniform image, even blending skin tone.

## why did I make this

Most logical people would run this project and question why I would invest any time and energy into making such a thing. This project was developed with 2 thoughts in mind, the first of which was pure boredum and the second being a sense of frustration with the way in which modern day software engineering was heading with a tilt towards more and more compute power. Technology is supposed to be the path to a better solution for the world but it needs to be sustainable

## Current status

The program currently supports static image face swaps with accurate blending and tone adjustment. It’s been tested on several different photo pairs and performs well for frontal and slightly tilted faces.

Future updates may include:
- Video-based face-swapping (frame-by-frame)
- Improved lighting and shadow correction
- Live webcam integration for real-time swaps

## Usage
After installing everything from the requirements. simply enter 
"python3 swapper.py *Path_To_Source_Image* *Path_To_Target_Image* *Path_To_Output_Image*"