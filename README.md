# Installation
run
> pip install -r requirements.txt

# Usage
> python face_detector.py --image=(Your Input Image) --feature=(which face feature you want to extract)

> python face_extractor.py --feature=(which face feature you want to extract)

face_extractor runs over every image in the faces folder whild face_detector only does that with a single image

wanted feature string must be from this set:

"head": Shape of head from left temple to right temple

"left eye brow": Left Eye Brow

"right eye brow": Right Eye Brow

"nose": Shape of Nose

"left eye": Left Eye

"right eye": Right eye

"lips": Lips

"all" : get all features

> python eyes.py

You can press the 'x' key at any time to stop the cycling and the 'z' key to exclude an image from the final results