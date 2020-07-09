# python-weka-wrapper3
# matplotlib

import sys
import os
import datetime

import weka.core.jvm as jvm
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter
from weka.core.classes import Random

import matplotlib.pyplot as plt
import matplotlib

# do stuff per model
# show sliding window vs. full past
# multiple sliding windows

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
"weka.classifiers.trees.J48",
"weka.classifiers.trees.RandomTree",
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
		
		print("Result for", classifier)
		print("Accuracy:", evaluation.percent_correct)
		print("AUC:", evaluation.area_under_roc(1))
		print("Recall:", evaluation.recall(1))
		print()
		
		classifier_objects.append(object)
		results.append([evaluation.percent_correct, evaluation.area_under_roc(1), evaluation.recall(1)])
	return classifier_objects, results

# Test a list of classifiers
def test_classifiers(classifiers, classifier_names, data, label):
	results = []
	print("Testing results (label:", label + "):\n")
	
	for i in range(len(classifiers)):
		print("Testing classifier", classifier_names[i] + "...")
		evaluation = Evaluation(data)
		evaluation.test_model(classifiers[i], data)
		
		print("Result for", classifier_names[i])
		print("Accuracy:", evaluation.percent_correct)
		print("AUC:", evaluation.area_under_roc(1))
		print("Recall:", evaluation.recall(1))
		print()
		
		results.append([evaluation.percent_correct, evaluation.area_under_roc(1), evaluation.recall(1)])
	return results

# Split data, collect the training data, also preprocess a bit
labels, splitted = split_by_index(data, group_size)

for i in range(len(splitted)):
	splitted[i].delete_attribute(splitted[i].attribute_by_name('FirstSeenDate').index)
	splitted[i].delete_attribute(splitted[i].attribute_by_name('TimeDateStamp').index)
	splitted[i].class_is_last()

if (train_mode == 'start'):
	train_data = splitted[0]
	for next in range(1, train_split):
		train_data = train_data.append_instances(train_data, splitted[next])
	train_data = apply_balance(train_data)
	train_label = "1-" + str(train_split * group_size)

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
else:
	label_list = []
	result_list = []
	
	start_pos = 1
	if (train_mode == 'relative_strict'):
		start_pos = train_split
	
	for i in range(start_pos, len(splitted)):
		print("Evaluating group", labels[i])
		
		train_data = splitted[i - 1]
		for next in range(1, train_split):
			pos = i - 1 - next
			if (pos < 0):
				break
			train_data = train_data.append_instances(train_data, splitted[pos])
		
		classifiers = build_classifiers(classifier_list, train_data)
		result = test_classifiers(classifiers, classifier_list, splitted[i], labels[i])
		label_list.append(labels[i])
		result_list.append(result)

# Gather the data into a plot-friendly format (dimensions: model, result, type (accuracy or recall))
model_accuracy = []
model_auc = []
model_recall = []
for i in range(len(classifier_list)):
	results_accuracy = []
	results_auc = []
	results_recall = []
	for j in range(len(result_list)):
		results_accuracy.append(result_list[j][i][0])
		results_auc.append(result_list[j][i][1])
		results_recall.append(result_list[j][i][2])
	model_accuracy.append(results_accuracy)
	model_auc.append(results_auc)
	model_recall.append(results_recall)

# Extract the model's name from its Weka filepath (example: weka.classifiers.trees.J48 -> J48)
def nice_title(title):
	return title.split('.')[-1]

# Plot the data
plt.rcParams['xtick.labelsize'] = 7
figure, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (13.0, 7.0))

ax1.set_title("Accuracy")
ax1.set_xlabel("Test group")
for i in range(len(model_accuracy)):
	ax1.plot(label_list, model_accuracy[i], label = nice_title(classifier_list[i]))
ax1.set_xticklabels(label_list, rotation=45, ha='right')
ax1.legend()

ax2.set_title("AUC")
ax2.set_xlabel("Test group")
for i in range(len(model_recall)):
	ax2.plot(label_list, model_auc[i], label = nice_title(classifier_list[i]))
ax2.set_xticklabels(label_list, rotation=45, ha='right')
ax2.legend()

ax3.set_title("Recall")
ax3.set_xlabel("Test group")
for i in range(len(model_recall)):
	ax3.plot(label_list, model_recall[i], label = nice_title(classifier_list[i]))
ax3.set_xticklabels(label_list, rotation=45, ha='right')
ax3.legend()

# Save models, if applicable
print("Run ID:", RUN_ID)

parent_dir = os.getcwd()
if (save):
	os.mkdir(os.path.join(parent_dir, RUN_ID))
	for i in range(len(classifier_list)):
		classifiers[i].serialize(RUN_ID + "/" + nice_title(classifier_list[i]) + " (" + RUN_ID + ")" + ".model", train_data)
	
	plt.savefig(RUN_ID + "/" + "Results" + " (" + RUN_ID + ")" + ".png", bbox_inches='tight')
	
	saver = Saver(classname = "weka.core.converters.ArffSaver")
	if (train_mode == 'start'):
		saver.save_file(train_data, RUN_ID + "/" + "train_" + train_label + ".arff")
	for i in range(len(splitted)):
		saver.save_file(splitted[i], RUN_ID + "/" + "test_" + labels[i] + ".arff")

# Display the plot
plt.show()	

# Stop the JVM, quit the program
jvm.stop()
sys.exit(0)