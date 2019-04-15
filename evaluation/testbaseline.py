import os
from os import listdir
from os.path import isfile, join
from calculate_fluency_bert import calculate_fluency
#from calculate_fluency_gpt import calculate_fluency

#mypath = '../baseline/yelp'
mypath = '../preprocess/yelp/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))and('train'not in f)]
results = []
for f in files:
    test_path = os.path.join(mypath, f)
    results.append(calculate_fluency(test_path))
for i in range(len(results)):
    print(files[i])
    print(results[i])

