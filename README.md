# Installation

## Windows
Download Anaconda (https://www.anaconda.com/products/individual)

run the following commands

> conda create --name eyes

> conda install -n eyes python=3.7.6 scipy tensorflow=1.14.0

> conda install -c conda-forge tf_object_detection

> conda activate eyes

> pip install opencv-python cmake dlib imutils matplotlib cython pillow

With this, your virtual environment is set up for this project.

Open up faces.rar and unpack it into a folder named 'faces'.

# Usage

This program requires a Dlib shape predictor trained to find landmarks for faces. Included is a 194 landmark detector but any face detector that finds landmarks for the eyes will work but will require manipulating face_detection.py accordingly.

Inside the faces folder, place whatever dataset of faces that will be measured. Included is a set of 596 faces taken from the Chicago Face Database. These faces are labeled with a 3 letter code representing the database source and then a two letter code where the first letter is the racial group the face belongs to (A = Asian, B = Black, L = Latino/Latina, W = White) and the second letter is the gender of the face (F = Female, M = Male).

With the shape predictor and the faces folder, you can now run

> python clusterize.py

and it will process each of the faces and create a folder containing the crops of the eyes and a file called eyes.txt that contains variables for each pair of eyes in the faces folder. This may take a long time depending on your system.

After clusterize finishes, you then will run

> gcc main.cpp

> a.exe

and follow the prompts. The filename will be eyes.txt. The data file name will be whatever name you wish to give to the clustered data. The maximum number of clusters is recommended to be the ceiling of the square root of the number of faces in the faces folder (e.g. if there are 596 faces, then the max number of clusters will be 25). I recommend making the min number of clusters 2. Then choose how many variables you want to optimize around and then choose the desired optimization functions for them. The seed can be any integer of 10 or more digits.

This may take around 10 minutes. When done it will have created a set of clustering groups. To group the eyes into their most common clustering groups, please see https://github.com/Noahkito/AMOSA-Clustering-for-Eyes.
