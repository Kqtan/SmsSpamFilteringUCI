# SmsSpamFilteringUCI
Everything's compiled into a single file TeeHee

Given I am told to record all my experience in Python.

Therefore, this is one of them.

This project is not my own work but it is done by some fine lads (including me ofc :P)

----------------------------
# 1.0 Installation
----------------------------

The source code is somewhat complicated and requires some installation on various libraries

If you find any error on the installation issue, please do not hesitate and open up your cmd to install the packages.

Package manager [pip] will assist in install the packages
Command example: pip install sklearn


-----------------------------
# 2.0 Reading the raw data
-----------------------------

raw data from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Option 1: Download the data and place them in your working directory, get your working directory via the method below.

Command:
```
import os

cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
```

Option 2: Manually get your working directory after you've downloaded your dataset
- make sure the string portion of the command below is the path to the dataset
- make sure the path you've copied from your file directory is modified using "forward slash" 
  instead of the default back slash ie. "/"

Line 43 in the source code "PM ME program code":
```
sms = pd.read_csv("C:/Users/user/SMSSpamCollection", sep = "\\t", header = None)
```


----------------------------
# 3.0 Content
----------------------------

The source code consists of 4 main sections

1. Preliminaries
- that includes importing of necessary packages
- reading of the raw data
- checking on basic information of the data

2. Processing
- includes further processing
- Exploratory Data Analysis
- NTLK pre-processing

3. Supervised model
- the process of training the model
- the result of each model

4. Unsupervised model
- process of utilising some common unsupervised model
- plotting of the result of each model
