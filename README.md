# Installation
run
> pip install -r requirements.txt

You need to have a C compiler installed for cmake
# Usage
> python clusterize.py -p={your shape predictor}

This will create a file called eyes.txt that contains information that can be used for a clustering algorithm. This may take a long time depending on the size of the dataset.

AMOSA clustering is done with main.cpp
> gcc main.cpp

This will create an executable file. The filename you give it will be eyes.txt and then what name you want to give the data. The maximum amount of clusters should be the square root of the size of your data set and the minimum is whatever you choose.
The indices are chosen as needed. I do 2 1 3. The seed can be any large number with a minimum of 10 digits.
