# ---------------------------------
# SMS Spamming Classification
# ---------------------------------
# import basic libraries
import pandas as pd
import numpy as np

# import plotting libraries
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import string

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Import Scikit Learn Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score 
from sklearn.metrics import f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef, log_loss
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.naive_bayes import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import ComplementNB, BernoulliNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# -----------------------
# Load the data
# -----------------------
sms = pd.read_csv("C:/Users/teohr/OneDrive/Desktop/UTAR 202206 - Radahn/SMSSpamCollection", sep = "\\t", header = None)
sms = sms.rename({0:"Class", 1: "Text"}, axis=1)

# ------------------------------------
# 1. Data Understanding
# Exploratory Data Analysis
# ------------------------------------
# Under this section, we are looking forward to understand our data better via visualisations and preprocessing. The insights drawn from this section will be vital in the decision on modelling and evaluating the outcome.
# Display the first 5 rows of our dataset
display(sms.head())

pd.set_option('display.max_colwidth', 1000)
display(sms[sms["Class"]=="spam"].head(5))
pd.set_option('display.max_colwidth', 100)


# checking the uniqueness of data
display(sms["Class"].unique())


# checking for missing values or abnormal data
print("Dimension of the dataset:")
display(sms.shape)
print("--------------------------------------")
display(sms.info())


# check for missing values
display(sms.isna().sum())

# visualise the missing values
sns.heatmap(sms.isnull(), cbar=False, cmap="viridis")
plt.show()


# check for duplicated values
display(sms.duplicated().sum())


# Remove the duplicates
sms = sms.drop_duplicates(keep="first")

# Verify the number of duplicates
display(sms.duplicated().sum())


sms.info()


# -------------------------------------------------
# Process the data by inserting extra parameters
# -------------------------------------------------
# include binary label for classes ham and spam
sms["Label"] = sms["Class"].map({"ham": 0, "spam": 1})

# introduce few parameters to accommodate the analysis
# number of characters
sms["Num_char"] = sms["Text"].apply(len)

# number of words
sms["Num_words"] = sms["Text"].apply(lambda x: len(nltk.word_tokenize(x)))

# number of sentences
sms["Num_sentences"] = sms["Text"].apply(lambda x: len(nltk.sent_tokenize(x)))

display(sms.sample(5))
 

# Basic Statistics on the improved dataframe
# overall
display(sms.describe())


# Analyse the statistics for both classes
# ham
print("----------------------------------------")
print("Basic statistics for ham messages:")
display(sms[ sms["Label"] == 0 ][["Num_char", "Num_words", "Num_sentences"]].describe())


# spam
print("----------------------------------------")
print("Basic statistics for spam messages:")
display(sms[ sms["Label"] == 1 ][["Num_char", "Num_words", "Num_sentences"]].describe())


# -------------------------------
# Data Visualisation
# -------------------------------
# bar plot to look at the frequency of the label
plt.bar(sms["Class"].unique(), sms["Label"].value_counts().array)
plt.show()


# pie chart to have a better visualisation
sms["Class"].value_counts().plot(kind="pie", labels=["Ham", "Spam"], colors=["#99ee80", "red"], 
                                 autopct="%0.2f%%", explode=(0, 0.15), shadow=True)
plt.title("The percentage of Ham vs Spam")
plt.show()


# checking whether data is imbalance
class_ratio = sum(sms["Label"]) / len(sms["Label"])
print("Class Ratio: ",round(class_ratio, 5))


# Using histogram to visualise the distribution of data with respect to each parameter
# number of characters
plt.figure(figsize = (10, 8))
sms_ham = sms[ sms["Class"] == "ham" ]["Num_char"]
sms_spam = sms[ sms["Class"] == "spam" ]["Num_char"]

sms_ham.plot(bins=70, kind="hist", color="green", 
             label="ham messages", alpha=0.9, edgecolor="black")
sms_spam.plot(kind="hist", color="red", 
              label="spam messages", alpha=0.7, edgecolor="black")

plt.legend()
plt.xlabel("Number of characters")
plt.show()


# number of words
plt.figure(figsize = (10, 8))

sms[ sms["Class"] == "ham" ]["Num_words"].plot(bins=50, kind="hist", color="green", label="ham messages", 
                                              alpha=0.9, edgecolor="black")
sms[ sms["Class"] == "spam" ]["Num_words"].plot(kind="hist", color="red", label="spam messages", 
                                               alpha=0.7, edgecolor="black")

plt.legend()
plt.xlabel("Number of words")
plt.show()


# number of sentences
plt.figure(figsize = (10, 8))

sms[ sms["Class"] == "ham" ]["Num_sentences"].plot(bins=60, kind="hist", color="green", label="ham messages", 
                                                   alpha=0.9, edgecolor="black")
sms[ sms["Class"] == "spam" ]["Num_sentences"].plot(bins=15, kind="hist", color="red", label="spam messages", 
                                                    alpha=0.7, edgecolor="black")
plt.legend()
plt.xlabel("Number of sentences")
plt.show()
print("-----------------------------------")


# boxplot to view the distribution and outliers involved
fig, ax = plt.subplots(1, 1)
plt.figure(figsize=(16, 12))

sns.boxplot(x=sms["Num_char"], y=sms["Class"], data=sms, hue="Class", ax=ax)
ax.set(xlabel="Number of characters")

plt.show()


fig, ax = plt.subplots(1, 1)
plt.figure(figsize=(16, 12))

sns.boxplot(x=sms["Num_words"], y=sms["Class"], data=sms, hue="Class", ax=ax)
ax.set(xlabel="Number of words")

plt.show()


# To have a better understanding on data
sns.pairplot(sms, hue="Label")

plt.show()


# Correlation to view the relationship between parameters

sns.heatmap(sms.corr(), annot=True, cmap="Blues")
plt.show()


# -------------------------------
# 2. Data Preprocessing
# -------------------------------
display(sms.shape)


# necessary tools for NLP
punctuation = string.punctuation
stop_words = set(stopwords.words("english"))

# lemmatization operator
lemmatizer = WordNetLemmatizer()

# function to process the text
def text_cleaner(text):
    x = []
    remove_punct = "".join([char for char in text if char not in punctuation])
    lowercase = word_tokenize(remove_punct.lower())
    
    for w in lowercase:
        if w not in stop_words:
            x.append(lemmatizer.lemmatize(w))
    return " ".join(x)

sms["Text"]=sms["Text"].apply(text_cleaner)
display(sms.sample(5))


from wordcloud import WordCloud
wc = WordCloud(width=1500, height=1000, 
               min_font_size=12, background_color="white")

spam_wc = wc.generate(sms[sms["Label"]==1]["Text"].str.cat(sep=" "))
plt.figure("figure" , (16, 9))
plt.imshow(spam_wc, interpolation="bilinear")
plt.show()


ham_wc = wc.generate(sms[sms["Label"]==0]["Text"].str.cat(sep=" "))

plt.figure("figure" , (16, 9))
plt.imshow(ham_wc, interpolation="bilinear")
plt.show()


# -----------------------------------------
# SPAM messages analysis
# -------------------------------------------
# creating a list to further analyse spam messages

spam = []

for m in sms[sms["Label"] == 1]["Text"].tolist():
    for w in m.split():
        spam.append(w)
           
display(len(spam))


spam_df = pd.DataFrame(spam)
spam_top20 = pd.DataFrame(spam_df.value_counts()[:20], columns = ["freq"])
spam_top20.reset_index(inplace=True)
display(spam_top20)


plt.figure(figsize=(20, 6))
plt.xticks(rotation="vertical")
sns.barplot(x=spam_top20[0], y=spam_top20["freq"])
plt.show()


# ---------------------------------
# HAM messages analysis
# ----------------------------------
# creating a list to further analyse ham messages

ham = []

for m in sms[sms["Label"] == 0]["Text"].tolist():
    for w in m.split():
        ham.append(w)
display(len(ham))


ham_df = pd.DataFrame(ham)
ham_top20 = pd.DataFrame(ham_df.value_counts()[:20], columns = ["freq"])
ham_top20.reset_index(inplace=True)
display(ham_top20)


plt.figure(figsize=(20, 6))
plt.xticks(rotation="vertical")
sns.barplot(x = ham_top20[0], y = ham_top20["freq"])
plt.show()


# -----------------------------------
# 3. Modelling
# a. Supervised Learning
# ----------------------------------
def check_train_test():
    print("Shape of Training input set:", X_train.shape)
    print("Shape of Traning output set:", y_train.shape)
    print("\n")
    print("Shape of Testing input set:", X_test.shape)
    print("Shape of Testing output set:", y_test.shape)
    
    
# k-fold cross validation, using k = 10
def skfold(model, name):
    # vectoring the text
    tfidf_vec = TfidfVectorizer()
    X = tfidf_vec.fit_transform(sms["Text"])
    y = np.array(sms["Label"])
    np.random.seed(2022)
    test_score = []
    train_score = []
    prec = []
    recall = []
    f1 = []
    lg_loss = []
    ham_loss =[]
    kappa = []
    matthews = []
   
    # 10-fold cross validation
    skf = StratifiedKFold(n_splits = 10, shuffle = True)
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_pred = model.predict(X_train)
        pred_proba = model.predict_proba(X_test)
        
        test_score.append(accuracy_score(y_test, y_pred))
        train_score.append(accuracy_score(y_train, train_pred))
        prec.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
        lg_loss.append(log_loss(y_test, pred_proba))
        ham_loss.append(hamming_loss(y_test, y_pred))
        kappa.append(cohen_kappa_score(y_test, y_pred))
        matthews.append(matthews_corrcoef(y_test, y_pred))
    
        
    print(f"Training Set Accuracy : {round(sum(train_score) / len(train_score) * 100, 5)} %\n")
    print(f"Test Set Accuracy : {round(sum(test_score) / len(test_score) * 100, 5)} % \n")
    print(f"Precision : {round(sum(prec) / len(prec) * 100, 5)} % \n")
    print(f"Recall : {round(sum(recall) / len(recall) * 100, 5)} % \n")
    print(f"F1 score : {round(sum(f1) / len(f1), 5)}  \n")
    print(f"Hamming Loss : {round(sum(ham_loss) / len(ham_loss) * 100, 5)} % \n")
    print(f"Cross Entropy Loss : {round(sum(lg_loss) / len(lg_loss), 5)} \n")
    print(f"Kappa Statistics : {round(sum(kappa) / len(kappa), 5)} \n")
    print(f"Matthews Correlation : {round(sum(matthews) / len(matthews), 5)} \n\n")
 
    
def plot_performance(model, name):
    ax = plt.subplot()
    y_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    pred_proba = model.predict_proba(X_test)
    
    print(f"Training Set Accuracy : {round(accuracy_score(y_train, train_pred) * 100, 5)} %\n")
    print(f"Test Set Accuracy : {round(accuracy_score(y_test, y_pred) * 100, 5)} % \n")
    print(f"Precision : {round(precision_score(y_test, y_pred) * 100, 5)} % \n")
    print(f"Recall : {round(recall_score(y_test, y_pred) * 100, 5)} % \n")
    print(f"F1 score : {round(f1_score(y_test, y_pred), 5)}  \n")
    print(f"Hamming Loss : {round(hamming_loss(y_test, y_pred) * 100, 5)} % \n")
    print(f"Cross Entropy Loss : {round(log_loss(y_test, pred_proba), 5)} \n")
    print(f"Kappa Statistics : {round(cohen_kappa_score(y_test, y_pred), 5)} \n")
    print(f"Matthews Correlation : {round(matthews_corrcoef(y_test, y_pred), 5)} \n\n")
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="g", cmap = "Blues")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("Actual labels")
    ax.set_title("Confusion Matrix of "+ name + " on Test set")
    ax.xaxis.set_ticklabels(["Negative", "Positive"]); ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    plt.show()
    
    clr = classification_report(y_test, y_pred)
    print(name + ": Classification Report on Test Set:\n-------------------------------------------------------\n", clr)
    
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label = name + "(area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    
    plt.show()
    
    
## Model 1: Logistic Regression
print(" Logistic Regression Model")
lr = LogisticRegression(solver="liblinear", penalty="l1")
# liblinear — Library for Large Linear Classification.
# L1 regularization adds an L1 penalty equal to the absolute value of the magnitude of coefficients.
# It limits the size of the coefficients.

# default step
# vectorizer - convert a collection of text documents to a vector of term/token counts (count words)
tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(sms["Text"]).toarray()
X = pd.DataFrame(X, columns = tfidf_vec.get_feature_names_out())
print(X)
y = sms["Label"]

print(" Linear sampling ")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
check_train_test()


# train and test the model
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
train_pred = lr.predict(X_train)
pred_proba = lr.predict_proba(X_test)

# show and evaluate the performance
plot_performance(lr, "Logistic Regression")


tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(sms["Text"])


feature_importance = pd.DataFrame({
    "feature": tfidf_vec.get_feature_names_out(), 
    "feature_importance": lr.coef_[0],
    "absolute value feature_importance": np.abs(lr.coef_[0])})
display(feature_importance.sort_values("absolute value feature_importance", ascending=False).head(10))
display(feature_importance.sort_values("absolute value feature_importance", ascending=False).tail(10))
print("Number of features:", feature_importance.shape[0])
print("Number of features equal to 0:", feature_importance[feature_importance['absolute value feature_importance'] == 0].shape[0])

# stratified sampling
print(" Stratified Sampling ")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
check_train_test()


#Using stratified sampling
# train and test the model
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
train_pred = lr.predict(X_train)
pred_proba = lr.predict_proba(X_test)

# show the performance
plot_performance(lr, "Logistic Regression")


print(" K-fold ")
skfold(lr, "Logistic Regression")


print("Test Logistic Regression with number of characters")
# ---------------------------------------------------------------
# Train the logistic regression model with number of characters
# ---------------------------------------------------------------
# basically to test the potential of num of characters in building the model
# however, the result is a little disappointing
# the variables may not be suitable in building a spam filter model
X = np.array(sms["Num_char"]).reshape(-1, 1)
y = sms["Label"]

np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
check_train_test()

# train and test the model
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
train_pred = lr.predict(X_train)
pred_proba = lr.predict_proba(X_test)

# show and evaluate the performance
plot_performance(lr, "Logistic Regression")


# ----------------------------------
# Model 2: Multinomial Naive Bayes
# ----------------------------------
print(" Multinomial Naive Bayes")
mnb = MultinomialNB()


tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(sms["Text"])
y = sms["Label"]

# linear sampling
print(" Linear Sampling")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
check_train_test()

# train and test the model
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
train_pred = mnb.predict(X_train)
pred_proba = mnb.predict_proba(X_test)

# show the performance
plot_performance(mnb, "Multinomial NB")


# stratified sampling
print(" Stratified Sampling")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
check_train_test()


# train and test the model
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
train_pred = mnb.predict(X_train)
pred_proba = mnb.predict_proba(X_test)

# show the performance
plot_performance(mnb, "Multinomial NB")


print(" K-Fold")
skfold(mnb, "Multinomial NB")

# -------------------------
# Model 3: Complement NB
# -------------------------
print("Complement NB")
cnb = ComplementNB()


tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(sms["Text"])
y = sms["Label"]

# linear sampling
print(" Linear Sampling ")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
check_train_test()

# train and test the model
cnb.fit(X_train, y_train)
y_pred = cnb.predict(X_test)
train_pred = cnb.predict(X_train)
pred_proba = cnb.predict_proba(X_test)

# show the performance
plot_performance(cnb, "Complement NB")


# stratified sampling
print(" Stratified Sampling")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
check_train_test()


# train and test the model
cnb.fit(X_train, y_train)
y_pred = cnb.predict(X_test)
train_pred = cnb.predict(X_train)
pred_proba = cnb.predict_proba(X_test)

# show the performance
plot_performance(cnb, "Complement NB")


print(" K-Fold")
skfold(cnb, "Complement NB")


# ----------------------------------
# Model 4: Bernoulli Naive Bayes
# ----------------------------------
print(" Bernoulli NB")
bnb = BernoulliNB()


tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(sms["Text"]).toarray()
X = pd.DataFrame(X, columns = tfidf_vec.get_feature_names_out())
y = sms["Label"]

# linear sampling
print(" Linear Sampling ")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
check_train_test()


# train and test the model
bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)
train_pred = bnb.predict(X_train)
pred_proba = bnb.predict_proba(X_test)

# show and evaluate the performance
plot_performance(bnb, "Bernoulli NB")


# stratified sampling
print(" Stratified Sampling")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
check_train_test()


# train and test the model
bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)
train_pred = bnb.predict(X_train)
pred_proba = bnb.predict_proba(X_test)

# show and evaluate the performance
plot_performance(bnb, "Bernoulli NB")


print(" K-Fold")
skfold(bnb, "Bernoulli NB")

# ----------------------------------
# Model 5: k-Neighbors Classifier
# ----------------------------------
print(" k-Neighbors Classifier")
knc = KNeighborsClassifier(n_neighbors=100)


tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(sms["Text"]).toarray()
X = pd.DataFrame(X, columns = tfidf_vec.get_feature_names_out())
y = sms["Label"]

# linear sampling
print(" Linear Sampling ")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
check_train_test()


# train and test the model
knc.fit(X_train, y_train)
y_pred = knc.predict(X_test)
train_pred = knc.predict(X_train)
pred_proba = knc.predict_proba(X_test)

# show and evaluate the performance
plot_performance(knc, "K-Neighbors Classifier")


# stratified sampling
print(" Stratified Sampling")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
check_train_test()


# train and test the model
knc.fit(X_train, y_train)
y_pred = knc.predict(X_test)
train_pred = knc.predict(X_train)
pred_proba = knc.predict_proba(X_test)

# show and evaluate the performance
plot_performance(knc, "K-Neighbors Classifier")


print(" K-Fold")
skfold(knc, "K-Neighbors Classifier")

# ----------------------------
# Model 6: Decision Trees
# ----------------------------
dt = DecisionTreeClassifier(max_depth=5)


# linear sampling
print(" Linear Sampling ")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
check_train_test()


# train and test the model
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
train_pred = dt.predict(X_train)
pred_proba = dt.predict_proba(X_test)

#show the performance
plot_performance(dt, "Decision Trees")


print(" Stratified Sampling")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
check_train_test()


# train and test the model
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
train_pred = dt.predict(X_train)
pred_proba = dt.predict_proba(X_test)

plot_performance(dt, "Decision Tree")


print(" K-Fold")
skfold(dt, "Decision Trees")

# Plotting the Decision Tree
fig = plt.figure(figsize=(16,9))
fig= tree.plot_tree(dt, feature_names = tfidf_vec.get_feature_names_out())

#---------------------------------------------
# Model 7: Random Forest Classifiers
# --------------------------------------------
rf = RandomForestClassifier(n_estimators = 50, max_depth=50, max_features = 500)
tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(sms["Text"])
y = sms["Label"]

# linear sampling
print( "Linear Sampling")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
check_train_test()

# train and test the model
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
train_pred = rf.predict(X_train)
pred_proba = rf.predict_proba(X_test)

# show and evaluate the performance
plot_performance(rf, "Random Forest Classifier")

# stratified sampling
print(" Stratified Sampling")
np.random.seed(2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
check_train_test()

# train and test the model
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
train_pred = rf.predict(X_train)
pred_proba = rf.predict_proba(X_test)

# show and evaluate the performance
plot_performance(rf, "Random Forest Classifier")

print(" K-Fold")
skfold(rf, "Random Forest Classifier")


## b. Unsupervised Learning

#PCA
def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-0.5,0.5)
    plt.ylim(-0.5,0.5)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.show()
pcamodel = PCA(2)
tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(sms["Text"]).toarray()
X = pd.DataFrame(X, columns = tfidf_vec.get_feature_names_out())
pca = pcamodel.fit_transform(X)

pca.shape


myplot(pca[:,0:2],np.transpose(pcamodel.components_[0:2,:]))
#######################
### PCA clustering
#######################
tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(sms["Text"]).toarray()
X = pd.DataFrame(X, columns = tfidf_vec.get_feature_names_out())


from sklearn.decomposition import PCA
np.random.seed(2022)

# scaling not needed since the variance is not large and the difference after scaling w/o std is similar (done experiment)
# reduce the dimensionality to 2 features
pca = PCA(n_components = 2)
df_pca = pca.fit_transform(X)
df_pca = pd.DataFrame(df_pca, columns = ["PC1", "PC2"])  
print(df_pca.shape)


# the tentative plot of the reduced dataset with their labels
sns.scatterplot(x = "PC1", y = "PC2", hue = sms["Class"].tolist(),
                palette = sns.color_palette("hls", 2),
                data = df_pca).set(title="SMS spam projection under PCA") 
plt.show()


# ----------
## K-means
# ----------
from sklearn.cluster import KMeans
# inertia is the sum of squared distances of samples to their closest cluster center
inertias = []
np.random.seed(2022)

for i in range(1, 11):
    
    # initialise the model with higher iterations and faster convergence
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 500)
    kmeans.fit(df_pca[["PC1", "PC2"]])
    
    inertias.append(kmeans.inertia_)


# plot the eblow method graph
plt.plot(range(1, 11), inertias, "bx-")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("The Elbow Method using Inertia")
plt.show()

# to prevent the accumulation of item within the list
del inertias[:]


# from elbow method above, we suspect num_of_clusters to be around
# 2, 3, 4
# further verify using silhouette method
# in simple words, silhouette method calculates the diff b/w (
# the avg distance b/w center and data within clusters
# AND the avg dist b/w center and data outside the cluster)

from sklearn.metrics import silhouette_score
range_cluster = [2, 3, 4, 5]
silhouette_avg = []
for i in range_cluster:
    
    # initialise the model with higher iterations and faster convergence
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 500)
    clus_labels = kmeans.fit_predict(df_pca[["PC1", "PC2"]])
    
    # silhouette score calculation
    silhouette_avg.append(silhouette_score(df_pca[["PC1", "PC2"]], clus_labels))
    
plt.plot(range_cluster, silhouette_avg, "bx-")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.title("Silhouette method illustration")

plt.show()

# to prevent the accumulation of item within the list
del silhouette_avg[:]


# 2 clusters has the highest silhouette score
# but the elbow shows the clearer winner
# using elbow and silhouette method, we select 3 clusters

np.random.seed(2022)
km = KMeans(n_clusters = 3, init = "k-means++", max_iter = 500)

# predict the label of each data
label = km.fit_predict(df_pca[["PC1", "PC2"]])
df_pca["Label"] = label
print(label)


# giving labels to the plot
labels = {
    0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3"
}

df_pca["Label"] = df_pca["Label"].map(labels)

# plotting
sns.scatterplot(data = df_pca, x = "PC1", y = "PC2", hue = df_pca["Label"], palette = "rainbow")
    
# plotting the Centroids
centroids = km.cluster_centers_
plt.scatter(centroids[:, 0] , centroids[:, 1] , s = 50, color = "k", label = "Centroids")
plt.title("The K-Means plot for SMS spam data using PCA n = 3")

plt.legend()
plt.show()


# we also try with 2 clusters
np.random.seed(2022)
km = KMeans(n_clusters = 2, init = "k-means++", max_iter = 500)

# predicting the labels
label = km.fit_predict(df_pca[["PC1", "PC2"]])
df_pca["Label"] = label

# giving the labels
labels = {
    0: "Cluster 1", 1: "Cluster 2"
}

df_pca["Label"] = df_pca["Label"].map(labels)

# plotting
sns.scatterplot(data = df_pca, x = "PC1", y = "PC2", hue = df_pca["Label"], palette = "rainbow")
    
# plotting the Centroids
centroids = km.cluster_centers_
plt.scatter(centroids[:, 0] , centroids[:, 1] , s = 50, color = "k", label = "Centroids")
plt.title("The K-Means plot for SMS spam data using PCA n = 2")

plt.legend()
plt.show()


# --------------------------
## Hierarchical Clustering
# --------------------------
import scipy.cluster.hierarchy as hc
from sklearn.cluster import AgglomerativeClustering

# slice only the components but not the label from previous part
# since the label is not useful here
df_pca = df_pca.iloc[:, 0:2]

# Plot the Dendrogram
plt.figure(figsize = (16, 9))
plt.title("Raw Text Dendrogram (PCA)")

# Ward method is the Minimal Increase of Sum-of-Squares
# it serves to minimise sum of square errors (SSE), to minimise error in overall
# lower p (default = 30) and truncate is to condense the dendrogram
# to have a better view of it
d = hc.dendrogram((hc.linkage(df_pca, method = "ward", metric = "euclidean")), 
                  p = 10, truncate_mode="level", leaf_rotation = 90)

plt.show()


# Create cluster using Agglomerative hierarchical clustering
# by default, the linkage is ward and metric euclidean
agc = AgglomerativeClustering(n_clusters = 3)
agc.fit(df_pca)

# plot the cluster
sns.scatterplot(data = df_pca, x = "PC1", y = "PC2", hue = agc.labels_, palette = "rainbow")
plt.title("Agglomerative Hierarchical Clusters with n = 3 (PCA)")

plt.show()


##################
### t-SNE
##################
tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(sms["Text"]).toarray()
X = pd.DataFrame(X, columns = tfidf_vec.get_feature_names_out())


from sklearn.manifold import TSNE

np.random.seed(2022)
# reduce the dimensionality to 2 features
# perplexity measures the effective number of neighbors
# trying to obtain a more defined structure
tsne = TSNE(n_components = 2, perplexity = 50, verbose = 1, learning_rate = "auto") 
z = tsne.fit_transform(X)

z_df = pd.DataFrame(z, columns = ["comp-1", "comp-2"])

# the tentative plot of the reduced dataset with their labels
sns.scatterplot(x = "comp-1", y = "comp-2", hue = sms["Class"].tolist(),
                palette = sns.color_palette("hls", 2),
                data = z_df).set(title="SMS spam T-SNE projection") 
plt.show()


# ----------
## K-means
# ----------
from sklearn.cluster import KMeans
# inertia is the sum of squared distances of samples to their closest cluster center
inertias = []
np.random.seed(2022)

for i in range(1, 11):
    
    # initialise the model with higher iterations and faster convergence
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 500)
    kmeans.fit(z_df)
    
    inertias.append(kmeans.inertia_)


# plot the eblow method illustration
plt.plot(range(1, 11), inertias, "bx-")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("The Elbow Method using Inertia")
plt.show()

# to prevent the accumulation of item within the list
del inertias[:]


# from elbow method above, we suspect num_of_clusters to be around
# 2, 3, 4, 5
# further verify using silhouette method
# in simple words, silhouette method calculates the diff b/w (
# the avg distance b/w center and data within clusters
# AND the avg dist b/w center and data outside the cluster)

from sklearn.metrics import silhouette_score
range_cluster = [2, 3, 4, 5, 6, 7]
silhouette_avg = []
np.random.seed(2022)
for i in range_cluster:
    
    # initialise the model with higher iterations and faster convergence
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 500)
    clus_labels = kmeans.fit_predict(z_df)
    
    # silhouette score calculation
    silhouette_avg.append(silhouette_score(z_df, clus_labels))

# plotting the score
plt.plot(range_cluster, silhouette_avg, "bx-")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.title("Silhouette method illustration")

plt.show()

# to prevent the accumulation of item within the list
del silhouette_avg[:]


# given the highest silhouette score when number = 2
np.random.seed(2022)
km = KMeans(n_clusters = 2, init = "k-means++", max_iter = 500)

# predict the label using k-means
label = km.fit_predict(z_df)

# inserting the label into the data frame
z_df["Label"] = label
print(label)


labels = {
    0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5", 5: "Cluster 6"
}

# map the label into strings
z_df["Label"] = z_df["Label"].map(labels)

sns.scatterplot(data = z_df, x = "comp-1", y = "comp-2", hue = "Label", palette = "rainbow")
    
# plotting the Centroids
centroids = km.cluster_centers_
plt.scatter(centroids[:, 0] , centroids[:, 1] , s = 50, color = "k", label = "Centroids")
plt.title("The K-Means plot for SMS spam data using t-SNE")

# to make sure the legend is not blocking the plot
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.show()


# plot the leftover possible clusters besides 3
# since the score is super low
plt.figure(figsize = (10, 15))

for i in range(4, 7):
    # given the highest silhouette score when number = 2
    np.random.seed(2022)
    km = KMeans(n_clusters = i, init = "k-means++", max_iter = 500)

    # removing the label
    z_df = z_df.iloc[:, 0:2]

    # predict the label using k-means
    label = km.fit_predict(z_df)

    # inserting the label into the data frame
    z_df["Label"] = label

    labels = {
        0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5", 5: "Cluster 6"
    }

    # map the label into strings
    z_df["Label"] = z_df["Label"].map(labels)
    
    
    plt.subplot(3, 1, i - 3)
    sns.scatterplot(data = z_df, x = "comp-1", y = "comp-2", hue = "Label", palette = "rainbow")

    # plotting the Centroids
    centroids = km.cluster_centers_
    plt.scatter(centroids[:, 0] , centroids[:, 1] , s = 50, color = "k", label = "Centroids")
    plt.title("The K-Means plot for SMS spam data using t-SNE")

    # to make sure the legend is not blocking the plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.show()

# --------------------------
## Hierarchical Clustering
# --------------------------
import scipy.cluster.hierarchy as hc
from sklearn.cluster import AgglomerativeClustering

# slice only the components but not the label from previous part
# since the label is not useful here
z_df = z_df.iloc[:, 0:2]

# Plot the Dendrogram
plt.figure(figsize = (16, 9))
plt.title("Raw Text Dendrogram (t-SNE)")
hc.set_link_color_palette(["r", "g", "b", "c", "m", "y"])

# Ward method is the Minimal Increase of Sum-of-Squares
# it serves to minimise sum of square errors (SSE), to minimise error in overall
# lower p (default = 30) and truncate is to condense the dendrogram
# to have a better view of it
d = hc.dendrogram((hc.linkage(z_df, method = "ward", metric = "euclidean")),  
                  p = 10, truncate_mode="level", leaf_rotation = 90)

plt.show()


# Create cluster using Agglomerative hierarchical clustering
# by default, the linkage is ward and metric euclidean
agc = AgglomerativeClustering(n_clusters = 2)
agc.fit(z_df)

# plot the figure
sns.scatterplot(data = z_df, x = "comp-1", y = "comp-2", hue = agc.labels_, palette = "rainbow")

plt.title("Agglomerative Hierarchical Clusters with n = 2 (t-SNE)")

# to prevent the legend from blocking the plot
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.show()
