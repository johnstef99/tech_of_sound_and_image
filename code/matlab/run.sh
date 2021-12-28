#!/bin/sh

samples_dir="../../../physionet/samples/all/" # you need to add the '/' at the end

LD_PRELOAD=/usr/lib/libfreetype.so matlab \
  -nodesktop \
  -nosplash \
  -r "addpath('physionet');generateS1Files('$samples_dir','test.csv');exit"
