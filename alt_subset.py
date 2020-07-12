# python-weka-wrapper3
# matplotlib

import sys
import os
import datetime
import math

import weka.core.jvm as jvm
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter
from weka.core.classes import Random

import matplotlib.pyplot as plt
import matplotlib

# also rundata with size=2000?
# note: first 10000 - all goodware

# Start the JVM for Weka
jvm.start()

# Parse command-line arguments, like "file=xxx.arff"
args = {}
for arg in sys.argv:
	if ('=' in arg):
		splitted = arg.split('=', 1)
		if (len(splitted) == 2): # sanity check
			args[splitted[0]] = splitted[1]
			
# Create a unique ID for this run to save models - simply the date/time
RUN_ID = str(datetime.datetime.now()).split(".")[0].replace(":", "꞉")

# Some constants - if no command-line argument present, the initial values serve as the defaults
filename = 'brazilian-malware - processed notext.arff'
if ('filename' in args):
	filename = args['filename']

group_size = 5000
if ('group_size' in args):
	group_size = int(args['group_size'])

train_split = 4
if ('train_split' in args):
	train_split = int(args['train_split'])
	
save = True
if ('save' in args):
	save = (args['save'].lower() != 'false')
	
sortby = 'FirstSeenDate'
if ('sortby' in args):
	sortby = args['sortby']
	
balance = True
if ('balance' in args):
	balance = (args['balance'].lower() != 'false')
	
train_mode = 'start'
if ('train_mode' in args):
	train_mode = args['train_mode']
	
# window_sizes = [5, 10, 100]
# window_sizes = [1] # just for debugging
# window_sizes = [2, 3, 5]
window_sizes = [10]

# Load the data file
print("Loading file", filename + '...')

loader = Loader(classname = 'weka.core.converters.ArffLoader')
data = loader.load_file(filename)

print("File loaded.")

# If specified, sort the dataset by a specific attribute
if (sortby is not None):
	attribute = data.attribute_by_name(sortby)
	data.sort(attribute.index)
	
target = 0
run = 0
cnt = 0
MAX = 100000
SZ = 1000
a = []
for elt in data:
	if (elt.get_value(19) == target):
		run += 1
		if (run <= MAX):
			cnt += 1
			a.append(target)
	else:
		target = elt.get_value(19)
		a.append(target)
		run = 1
		cnt += 1
		
print(cnt)

MAX_IMBALANCE = .90
C = int(SZ * MAX_IMBALANCE)

psum = []
run = 0
for elt in a:
	run += elt
	psum.append(run)

def getsum(left, right):
	sum = psum[right]
	if (left > 0):
		sum -= psum[left - 1]
	return sum

dp = []
for i in range(len(a)):
	cur = 0
	if (i > 0):
		cur = max(cur, dp[i - 1])
	if (i >= SZ):
		val = getsum(i - SZ + 1, i)
		if (max(val, SZ - val) <= C):
			cur = max(cur, 1 + dp[i - SZ])
	dp.append(cur)

print(dp[-1])

cnt = 0
c = [0, 0]
for i in range(len(a)):
	val = int(a[i])
	if (c[val] < C):
		c[val] += 1
	if (c[0] + c[1] >= SZ):
		c = [0, 0]
		cnt += 1
print(cnt)

# Stop the JVM, quit the program
jvm.stop()
sys.exit(0)