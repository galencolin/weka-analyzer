# Some information about the command-line arguments (for rundata.py)

**balance**

Default value: true

A boolean indicator for whether the dataset should be balanced using the filter you talked about. 
To make it false, the value must be exactly equal to "false" (non-case-sensitive), otherwise, it will be true.

**filename**

Default value: 'brazilian-malware - processed notext.arff', a file I used locally

The .arff file to load. It's probably most convenient to stick the script into the same directory as the directory with the data, 
otherwise, file paths are necessary (I think they should work fine, but I don't know). 
If a filename has spaces, enter the argument as "filename=a b.arff", with the quotes around it.

**group_size**

Default value: 5000

The size of the groups that the data should be split into.

**train_split**

Default value: 4

The number of groups that should be used as training data. For a training set size of 20,000, this value should be 20,000/5,000 = 4.

**save**

Default value: true

A boolean indicator for whether everything should be saved or not. 
If true, the models, new datasets created, and the result graph will be saved to a folder named after the time that the program was run. 
To make it false, the value must be exactly equal to "false" (non-case-sensitive), otherwise, it will be true.

**sortby**

Default value: FirstSeenDate

Specifies the name of an attribute to order the dataset by. For example, TimeDateStamp or FirstSeenDate.