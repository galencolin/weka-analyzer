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
	
window_sizes = [5, 10, 100]
# window_sizes = [1] # just for debugging

# Load the data file
print("Loading file", filename + '...')

loader = Loader(classname = 'weka.core.converters.ArffLoader')
data = loader.load_file(filename)

print("File loaded.")

# If specified, sort the dataset by a specific attribute
if (sortby is not None):
	attribute = data.attribute_by_name(sortby)
	data.sort(attribute.index)

# Split the dataset into groups of [interval] instances each.
# strict_size means that we only return groups with size of exactly [interval]. 
# For example, with 50,123 instances splitting by 5,000, this means we ignore the last group with 123 instances.
def split_by_index(data, interval, strict_size = True):
	labels = []
	splitted = []
	length = len(data)
	for index in range(0, length, interval):
		# If we're at the end of the file, make sure we don't go out of bounds
		cur_size = interval
		if (index + cur_size > length):
			cur_size = length - index
			
		if (not strict_size or cur_size == interval):
			label = str(index + 1) + "-" + str(index + cur_size)
			labels.append(label)
			splitted.append(data.subset(row_range = label))
	return labels, splitted
	
# Split the dataset by a certain column's values (for example, in ranges of dates)
# todo

# Apply the balancing filter to a dataset
def apply_balance(data):
	if (not balance):
		return data
	balancer = Filter(classname = "weka.filters.supervised.instance.Resample", options = ["-B", "1"])
	balancer.inputformat(data)
	return balancer.filter(data)

# A list of classifiers
classifier_list = [
"weka.classifiers.trees.RandomTree",
"weka.classifiers.trees.J48",
"weka.classifiers.trees.RandomForest",
"weka.classifiers.rules.JRip",
"weka.classifiers.rules.PART",
"weka.classifiers.lazy.IBk"
]

# Builds a list of classifiers without evaluating them
def build_classifiers(classifier_list, data):
	classifier_objects = []
	for classifier in classifier_list:
		print("Building classifier", classifier + "...")
		object = Classifier(classname = classifier)
		object.build_classifier(data)
		classifier_objects.append(object)
	return classifier_objects

# Train a list of classifiers - by default, 10-fold cross validation
def train_classifiers(classifier_list, data):
	classifier_objects = []
	results = []
	
	print("Training results:\n")
	for classifier in classifier_list:
		print("Training classifier", classifier + "...")
		object = Classifier(classname = classifier)
		evaluation = Evaluation(data)
		evaluation.crossvalidate_model(object, data, 10, Random(2233))
		object.build_classifier(data)
		
		acc = evaluation.percent_correct
		auc = evaluation.area_under_roc(1)
		rec = evaluation.recall(1)
		if (math.isnan(acc) or math.isnan(auc) or math.isnan(rec) or rec < 0.02):
			acc = float('nan')
			auc = float('nan')
			rec = float('nan')
		
		print("Result for", classifier)
		print("Accuracy:", acc)
		print("AUC:", auc)
		print("Recall:", rec)
		print()
		
		classifier_objects.append(object)
		results.append([acc, auc, rec])
	return classifier_objects, results

# Test a list of classifiers
def test_classifiers(classifiers, classifier_names, data, label):
	results = []
	print("Testing results (label:", label + "):\n")
	
	for i in range(len(classifiers)):
		print("Testing classifier", classifier_names[i] + "...")
		evaluation = Evaluation(data)
		evaluation.test_model(classifiers[i], data)
		
		acc = evaluation.percent_correct
		auc = evaluation.area_under_roc(1)
		rec = evaluation.recall(1)
		if (math.isnan(acc) or math.isnan(auc) or math.isnan(rec) or rec < 0.02):
			acc = float('nan')
			auc = float('nan')
			rec = float('nan')
		
		print("Result for", classifier_names[i])
		print("Accuracy:", acc)
		print("AUC:", auc)
		print("Recall:", rec)
		print()
		
		results.append([acc, auc, rec])
	return results

# Split data, collect the training data, also preprocess a bit
labels, splitted = split_by_index(data, group_size)

for i in range(len(splitted)):
	splitted[i].delete_attribute(splitted[i].attribute_by_name('FirstSeenDate').index)
	splitted[i].delete_attribute(splitted[i].attribute_by_name('TimeDateStamp').index)
	splitted[i].class_is_last()
	splitted[i] = apply_balance(splitted[i])

def run_slide(classifier, data, train_split):
	label_list = []
	result_list = []

	start_pos = 1
	if (train_mode == 'relative_strict'):
		start_pos = train_split

	# Run the evaluation
	if (start_pos < len(splitted)):
		for i in range(start_pos, len(splitted)):
			# manually exclude first 10,000
			num = int(labels[i].split("-")[-1])
			if (num <= 10000):
				continue
			
			print("Evaluating group", labels[i])
			
			train_data = splitted[i - 1]
			for next in range(1, train_split):
				pos = i - 1 - next
				if (pos < 0):
					break
				train_data = train_data.append_instances(train_data, splitted[pos])
			
			classifiers = build_classifiers([classifier], train_data)
			result = test_classifiers(classifiers, [classifier], splitted[i], labels[i])
			label_list.append(labels[i])
			result_list.append(result)
	return label_list, result_list

# Extract the model's name from its Weka filepath (example: weka.classifiers.trees.J48 -> J48)
def nice_title(title):
	return title.split('.')[-1]

parent_dir = os.getcwd()

if (save):
	os.mkdir(os.path.join(parent_dir, RUN_ID))
	with open(RUN_ID + "/" + "log.txt", "w") as f:
		cmd = 'python '
		for arg in sys.argv:
			if (' ' in arg):
				arg = "\"" + arg + "\""
			cmd += arg + ' '
		f.write(cmd + '\n')

for classifier in classifier_list:
	print("Running model", classifier)
	label_list = []
	result_list = []
	for winsize in window_sizes:
		label_results, results = run_slide(classifier, data, winsize)
		label_list.append(label_results)
		result_list.append(results)
		
	results_accuracy = []
	results_auc = []
	results_recall = []
	for i in range(len(window_sizes)):
		result_acc = []
		result_auc = []
		result_rec = []
		for j in range(len(result_list[i])):
			result_acc.append(result_list[i][j][0][0])
			result_auc.append(result_list[i][j][0][1])
			result_rec.append(result_list[i][j][0][2])
		results_accuracy.append(result_acc)
		results_auc.append(result_auc)
		results_recall.append(result_rec)
		
	# Plot the data
	plt.rcParams['xtick.labelsize'] = 7

	title = nice_title(classifier)
	
	plt.figure(figsize = (13, 7))
	plt.title("Accuracy (" + title + ")")
	plt.xlabel("Test group")
	for i in range(len(window_sizes)):
		winsize = window_sizes[i]
		if (winsize >= len(splitted)):
			winsize = "All previous"
		plt.plot(label_list[i], results_accuracy[i], label = str(winsize) + " groups")
	plt.xticks(label_list[-1], rotation=45, ha='right')
	plt.legend()
			
	if (save):
		plt.savefig(RUN_ID + "/" + "Accuracy - " + title + " (" + RUN_ID + ")" + ".png", bbox_inches='tight')
		
	plt.clf()

	plt.figure(figsize = (13, 7))
	plt.title("AUC (" + title + ")")
	plt.xlabel("Test group (" + title + ")")
	for i in range(len(window_sizes)):
		winsize = window_sizes[i]
		if (winsize >= len(splitted)):
			winsize = "All previous"
		plt.plot(label_list[i], results_auc[i], label = str(winsize) + " groups")
	plt.xticks(label_list[-1], rotation=45, ha='right')
	plt.legend()
	
	if (save):
		plt.savefig(RUN_ID + "/" + "AUC - " + title + " (" + RUN_ID + ")" + ".png", bbox_inches='tight')
	
	plt.clf()
	
	plt.figure(figsize = (13, 7))
	plt.title("Recall (" + title + ")")
	plt.xlabel("Test group (" + title + ")")
	for i in range(len(window_sizes)):
		winsize = window_sizes[i]
		if (winsize >= len(splitted)):
			winsize = "All previous"
		plt.plot(label_list[i], results_recall[i], label = str(winsize) + " groups")
	plt.xticks(label_list[-1], rotation=45, ha='right')
	plt.legend()
	
	if (save):
		plt.savefig(RUN_ID + "/" + "Recall - " + title + " (" + RUN_ID + ")" + ".png", bbox_inches='tight')
		
	plt.clf()
	
	if (save):
		with open(RUN_ID + "/" + title + " (" + RUN_ID + ")" + ".txt", "w") as f:
			f.write(str(window_sizes) + '\n')
			f.write(str(label_list) + '\n')
			f.write(str(results_accuracy) + '\n')
			f.write(str(results_auc) + '\n')
			f.write(str(results_recall) + '\n')
	
	print("Model", classifier, "complete.")

# Save models, if applicable
# print("Run ID:", RUN_ID)

# parent_dir = os.getcwd()
# if (save):
	# os.mkdir(os.path.join(parent_dir, RUN_ID))
	
	# plt.savefig(RUN_ID + "/" + "Results" + " (" + RUN_ID + ")" + ".png", bbox_inches='tight')
	
	# saver = Saver(classname = "weka.core.converters.ArffSaver")
	# if (train_mode == 'start'):
		# saver.save_file(train_data, RUN_ID + "/" + "train_" + train_label + ".arff")
	# for i in range(len(splitted)):
		# saver.save_file(splitted[i], RUN_ID + "/" + "test_" + labels[i] + ".arff")

# Display the plot
# plt.show()	

# Stop the JVM, quit the program
jvm.stop()
sys.exit(0)