# THINGS TO USE PIP FOR
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
RUN_ID = str(datetime.datetime.now()).split(".")[0].replace(":", "꞉") + " (rundata)"

# Some constants - if no command-line argument present, the initial values serve as the defaults
filename = 'brazilian-malware - processed notext.arff'
if ('filename' in args):
	filename = args['filename']

group_size = 5000
if ('group_size' in args):
	group_size = int(args['group_size'])

train_split = 3
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
	
splitby = 'count'
if ('splitby' in args):
	splitby = args['splitby'].lower()

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

one_day = 1000 * 60 * 60 * 24
month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
month_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def get_time_val(month, year):
	passed_days = 0
	for i in range(month):
		passed_days += month_days[i]
		if (i == 1 and year % 4 == 0): # leap year
			passed_days += 1
			
	return ((year - 1970) * 365 + passed_days) * one_day

# Split the dataset by months, by FirstSeenDate
def split_by_month(data, interval, strict_size = True):
	labels = []
	splitted = []
	length = len(data)
	pointer = 0
	current_month = 0
	current_year = 1970
	date_index = data.attribute_by_name("FirstSeenDate").index
	while (pointer < length):
		last_pointer = pointer
		
		next_month = current_month + interval
		next_year = current_year + next_month // 12
		next_month %= 12
		
		interval_start = get_time_val(current_month, current_year)
		interval_end = get_time_val(next_month, next_year)
		
		while (pointer < length):
			date = data.get_instance(pointer).get_value(date_index)
			if (interval_start <= date and date < interval_end):
				pointer += 1
			else:
				break
		
		if (pointer > last_pointer):
			month_label = month_abbr[current_month] + " " + str(current_year)
			if (interval > 1):
				before_next_month = next_month - 1
				before_next_year = next_year
				if (before_next_month < 0):
					before_next_month += 12
					before_next_year -= 1
				month_label += " - " + month_abbr[before_next_month] + " " + str(before_next_year)
			labels.append(month_label)
			range_label = str(last_pointer + 1) + "-" + str(pointer)
			splitted.append(data.subset(row_range = range_label))
		
		current_month = next_month
		current_year = next_year
		
	return labels, splitted

# Apply the balancing filter to a dataset
def apply_balance(data):
	if (not balance):
		return data
	balancer = Filter(classname = "weka.filters.supervised.instance.Resample", options = ["-B", "1"])
	balancer.inputformat(data)
	return balancer.filter(data)

# A list of classifiers
classifier_list = [
# ["weka.classifiers.meta.Vote", ["-S", "1", "-B", "weka.classifiers.trees.RandomForest", "-B", "weka.classifiers.trees.RandomTree", "-R", "AVG"]],
# ["weka.classifiers.meta.RandomCommittee", ["-W", "weka.classifiers.trees.RandomForest"]],
# ["weka.classifiers.trees.J48", ""],
# ["weka.classifiers.trees.RandomTree", ""],
["weka.classifiers.meta.AdaBoostM1", ["-W", "weka.classifiers.trees.RandomForest"]],
["weka.classifiers.trees.RandomForest", ["-P", "100", "-I", "1000", "-K", "4", "-M", "1.0", "-V", "0.001", "-S", "1", "-depth", "10"]], 
["weka.classifiers.trees.RandomForest", ""],
# ["weka.classifiers.rules.JRip", ""],
# ["weka.classifiers.rules.PART", ""],
# ["weka.classifiers.lazy.IBk", ""],
]

# Builds a list of classifiers without evaluating them
def build_classifiers(classifier_list, data):
	classifier_objects = []
	for classifier in classifier_list:
		print("Building classifier", classifier[0] + "...")
		object = Classifier(classname = classifier[0], options = classifier[1])
		object.build_classifier(data)
		classifier_objects.append(object)
	return classifier_objects

# Train a list of classifiers - by default, 10-fold cross validation
def train_classifiers(classifier_list, data):
	classifier_objects = []
	results = []
	
	print("Training results:\n")
	for classifier in classifier_list:
		print("Training classifier", classifier[0] + "...")
		object = Classifier(classname = classifier[0], options = classifier[1])
		evaluation = Evaluation(data)
		evaluation.crossvalidate_model(object, data, 10, Random(2233))
		object.build_classifier(data)
		
		acc = evaluation.percent_correct
		auc = evaluation.area_under_roc(1)
		rec = evaluation.recall(1)
		pre = evaluation.precision(1)
		if (math.isnan(acc) or math.isnan(auc) or math.isnan(rec) or math.isnan(pre) or rec < 0.02):
			acc = float('nan')
			auc = float('nan')
			rec = float('nan')
			pre = float('nan')
		
		print("Result for", classifier[0])
		print("Accuracy:", acc)
		print("AUC:", auc)
		print("Recall:", rec)
		print("Precision:", pre)
		print()
		
		classifier_objects.append(object)
		results.append([acc, auc, rec, pre])
	return classifier_objects, results

# Test a list of classifiers
def test_classifiers(classifiers, classifier_names, data, label):
	results = []
	print("Testing results (label:", label + "):\n")
	
	for i in range(len(classifiers)):
		print("Testing classifier", classifier_names[i][0] + "...")
		evaluation = Evaluation(data)
		evaluation.test_model(classifiers[i], data)
		
		acc = evaluation.percent_correct
		auc = evaluation.area_under_roc(1)
		rec = evaluation.recall(1)
		pre = evaluation.precision(1)
		if (math.isnan(acc) or math.isnan(auc) or math.isnan(rec) or math.isnan(pre) or rec < 0.02):
			acc = float('nan')
			auc = float('nan')
			rec = float('nan')
			pre = float('nan')
		
		print("Result for", classifier_names[i][0])
		print("Accuracy:", acc)
		print("AUC:", auc)
		print("Recall:", rec)
		print("Precision:", pre)
		print()
		
		results.append([acc, auc, rec, pre])
	return results

# Split data, collect the training data, also preprocess a bit
if (splitby == 'count'):
	labels, splitted = split_by_index(data, group_size)
else:
	labels, splitted = split_by_month(data, group_size)

for i in range(len(splitted)):
	splitted[i].delete_attribute(splitted[i].attribute_by_name('FirstSeenDate').index)
	splitted[i].delete_attribute(splitted[i].attribute_by_name('TimeDateStamp').index)
	splitted[i].class_is_last()

train_data = splitted[0]
for next in range(1, train_split):
	train_data = train_data.append_instances(train_data, splitted[next])
train_data = apply_balance(train_data)
train_label = "Training set"

for i in range(len(splitted)):
	splitted[i] = apply_balance(splitted[i])

# Train the models
label_list = [train_label]
classifiers, train_results = train_classifiers(classifier_list, train_data)
result_list = [train_results]

# Test the models
for i in range(train_split, len(splitted)):
	result = test_classifiers(classifiers, classifier_list, splitted[i], labels[i])
	label_list.append(labels[i])
	result_list.append(result)

# Gather the data into a plot-friendly format (dimensions: model, result, type (accuracy or recall))
model_accuracy = []
model_auc = []
model_recall = []
model_precision = []
for i in range(len(classifier_list)):
	results_accuracy = []
	results_auc = []
	results_recall = []
	results_precision = []
	for j in range(len(result_list)):
		results_accuracy.append(result_list[j][i][0])
		results_auc.append(result_list[j][i][1])
		results_recall.append(result_list[j][i][2])
		results_precision.append(result_list[j][i][3])
	model_accuracy.append(results_accuracy)
	model_auc.append(results_auc)
	model_recall.append(results_recall)
	model_precision.append(results_precision)

# Extract the model's name from its Weka filepath (example: weka.classifiers.trees.J48 -> J48)
def nice_title(title, index):
	if (title[1] == [] or title[1] == ""):
		return title[0].split('.')[-1] + " (" + str(index) + ")"
	else:
		return title[0].split('.')[-1] + " (modified)" + " (" + str(index) + ")"

# Save models, if applicable
print("Run ID:", RUN_ID)

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
	
	for i in range(len(classifier_list)):
		classifiers[i].serialize(RUN_ID + "/" + nice_title(classifier_list[i], i) + " (" + RUN_ID + ")" + ".model", train_data)
		
	saver = Saver(classname = "weka.core.converters.ArffSaver")
	saver.save_file(train_data, RUN_ID + "/" + "train_" + train_label + ".arff")
	for i in range(len(splitted)):
		saver.save_file(splitted[i], RUN_ID + "/" + "test_" + labels[i] + ".arff")
	
	with open(RUN_ID + "/" + "results.txt", "w") as f:
		f.write(str(label_list) + '\n')
		for i in range(len(classifier_list)):
			f.write(nice_title(classifier_list[i], i) + " (accuracy): " + str(model_accuracy[i]) + '\n')
		for i in range(len(classifier_list)):
			f.write(nice_title(classifier_list[i], i) + " (auc): " + str(model_auc[i]) + '\n')
		for i in range(len(classifier_list)):
			f.write(nice_title(classifier_list[i], i) + " (recall): " + str(model_recall[i]) + '\n')
		for i in range(len(classifier_list)):
			f.write(nice_title(classifier_list[i], i) + " (precision): " + str(model_precision[i]) + '\n')

# Plot the data
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['font.size'] = 25
plt.rcParams['legend.fontsize'] = 18.75

plt.figure(figsize = (13, 7))
plt.title("Accuracy")
plt.xlabel("Test group")
for i in range(len(model_accuracy)):
	plt.plot(label_list, model_accuracy[i], label = nice_title(classifier_list[i], i))
plt.xticks(label_list, rotation=45, ha='right')
plt.legend()
# plt.tight_layout()
			
if (save):
	plt.savefig(RUN_ID + "/" + "Accuracy (" + RUN_ID + ")" + ".png", bbox_inches='tight')
	
plt.clf()

plt.figure(figsize = (13, 7))
plt.title("AUC")
plt.xlabel("Test group")
for i in range(len(model_recall)):
	plt.plot(label_list, model_auc[i], label = nice_title(classifier_list[i], i))
plt.xticks(label_list, rotation=45, ha='right')
plt.legend()
# plt.tight_layout()
			
if (save):
	plt.savefig(RUN_ID + "/" + "AUC (" + RUN_ID + ")" + ".png", bbox_inches='tight')
	
plt.clf()

plt.figure(figsize = (13, 7))
plt.title("Recall")
plt.xlabel("Test group")
for i in range(len(model_recall)):
	plt.plot(label_list, model_recall[i], label = nice_title(classifier_list[i], i))
plt.xticks(label_list, rotation=45, ha='right')
plt.legend()
# plt.tight_layout()
			
if (save):
	plt.savefig(RUN_ID + "/" + "Recall (" + RUN_ID + ")" + ".png", bbox_inches='tight')
	
plt.clf()

plt.figure(figsize = (13, 7))
plt.title("Precision")
plt.xlabel("Test group")
for i in range(len(model_precision)):
	plt.plot(label_list, model_precision[i], label = nice_title(classifier_list[i], i))
plt.xticks(label_list, rotation=45, ha='right')
plt.legend()
# plt.tight_layout()
			
if (save):
	plt.savefig(RUN_ID + "/" + "Precision (" + RUN_ID + ")" + ".png", bbox_inches='tight')
	
plt.clf()

# Stop the JVM, quit the program
jvm.stop()
sys.exit(0)