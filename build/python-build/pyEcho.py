#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import sys
import numpy
import g2s

import time

argumentDictionary={};
currentFlag='';
currentInputsArray=[];

for x in sys.argv :
	if(x.startswith('-')): # new input type
		argumentDictionary[currentFlag]=currentInputsArray;
		currentFlag=x[1:]
		currentInputsArray=[];
	else:
		currentInputsArray.append(x);
argumentDictionary[currentFlag]=currentInputsArray;
workId=argumentDictionary['r'][0][5:-4];

log=open(argumentDictionary['r'][0],"w",buffering=1); # open with direct write import to don loose errro in case of crash
log.write(" ".join(sys.argv)+"\n"); # usually start the log file by printing the comamnd line that we are able to run the situation agin in cases of crash

# load the image file 
im,dt=g2s.loadData(argumentDictionary['ti'][0])

time_start = time.time()

for p in range(0,101):
	time.sleep(0.02)
	log.write("whatever %f%%\n" % p);

duration= time.time()- time_start;


log.write("computation time %f s\n" % duration);
log.write("computation time %f ms\n" % (duration*1000));

# witre the final reuslt
g2s.writeData(im,dt,'im_1_'+workId)



log.close(); # close log file