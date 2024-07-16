import re
import string
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

# Load the training and test datasets.

dataset = pd.read_csv('mediaeval-2015-trainingset.txt', sep='\t')
dataset_test = pd.read_csv('mediaeval-2015-testset.txt', sep='\t')

#Download resources for vocabulary
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')

#Remove tweet duplicates for dataset
def remove_duplicates(dataframe):
    dataframe = dataframe.sort_values('tweetText')
    dataframe = dataframe.drop_duplicates(subset='tweetText', keep='first')
    return dataframe

# Apply the function to the dataset
dataset = remove_duplicates(dataset)
dataset_test = remove_duplicates(dataset_test)

# Remove hashtags, user mentions, and URLs from the text.
def preprocess_text(text):
    new_text = []
    for t in text.split(" "):
        t = '' if (t.startswith('@')
                   or
                   t.startswith('http')
                   or
                   t.startswith('#')) else t
        if t != '':
            new_text.append(t)
    return " ".join(new_text)


dataset['CleanData'] = dataset['tweetText'].apply(preprocess_text)
dataset_test['CleanData'] = dataset_test['tweetText'].apply(preprocess_text)

# Preprocess the tweets in the training and test datasets by removing certain types of words.
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
english_punctuations = string.punctuation
punctuations_list = english_punctuations
translator = str.maketrans('', '', punctuations_list)


# Clean the text by removing stop words, lemmatizing, and removing punctuation marks.
def CleanData(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)  # Removing punctuation marks
    text = re.sub(r'(?<=[^0-9])/(?=[^0-9])', ' ', text)  # Replacing slashes surrounded by non-digits with a space
    text = re.sub("\t+", " ", text)  # converting multiple tabs and spaces ito a single tab or space
    text = re.sub(" +", " ", text)
    text = re.sub("\.\.+", "", text)  # these were the common noises in our data, depends on data
    text = re.sub("\A ?", "", text)  # Removing leading space if it exists
    text = text.lower()
    text = text.split()
    text = ' '.join([w for w in text if len(w) > 1])
    text = " ".join([word.lower() for word in str(text).split() if word not in stop_words])
    text = text.translate(translator)
    return text

dataset['CleanData'] = dataset['CleanData'].apply(lambda text: CleanData(text))
dataset_test['CleanData'] = dataset_test['CleanData'].apply(lambda text: CleanData(text))

#Remove emojies
def emoji_remover(text):
    emojis = re.compile("["
        u"\U0001F600-\U0001F64f"  # emoticons
        u"\U0001F300-\U0001F5ff"  # symbols & pictographs
        u"\U0001F680-\U0001F6ff"  # transport & map symbols
        u"\U0001F1e0-\U0001F1ff"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emojis.sub(r'', text)

dataset['CleanData'] = dataset['CleanData'].apply(lambda text: emoji_remover(text))
dataset_test['CleanData'] = dataset_test['CleanData'].apply(lambda text: emoji_remover(text))

# Split the cleaned text from traning and testing data  into a list of words
text_cleaned_splitted = [text.split() for text in dataset['CleanData']]
text_cleaned_splitted_test = [text.split() for text in dataset_test['CleanData']]

# Concatenating the cleaned text from both the training and test datasets
CleanData = dataset['CleanData']
CleanData = dataset_test['CleanData']


# Convert the labels in the training and test datasets from strings to integers
def labels_to_numbers(label):
    if label[0] == 'r':
        return 0
    else:
        return 1

dataset['label'] = dataset['label'].apply(labels_to_numbers)
dataset_test['label'] = dataset_test['label'].apply(labels_to_numbers)

# Drop unnecessary columns from the training and test datasets.
data = dataset.drop(columns=['timestamp', 'username', 'userId', 'tweetText'])
data_test = dataset_test.drop(columns=['timestamp', 'username', 'userId', 'tweetText'])

# Bag-of-Words Vectorizer
cv = CountVectorizer(max_features=1000)
X_train = cv.fit_transform(data.CleanData).toarray()
y_train = np.array(data.label.values)
X_test = cv.fit_transform(data_test.CleanData).toarray()
y_test = np.array(data_test.label.values)

# TFIDF Vectorizer
tfidf = TfidfVectorizer(max_features=1000, stop_words='english', max_df=0.2)
Xt_train = tfidf.fit_transform(data.CleanData).toarray()
Xt_test = tfidf.fit_transform(data_test.CleanData).toarray()

# Model Training
lr = LogisticRegression()
sc = svm.SVC(kernel='poly', max_iter=300)
dtc = DecisionTreeClassifier(criterion="entropy", random_state=43)
rfc = RandomForestClassifier(n_estimators=500)
gb = GradientBoostingClassifier(max_depth=2, n_estimators=120)

# Traing Models on vectors of Bag-of-Words
lr = lr.fit(X_train, y_train)
sc = sc.fit(X_train, y_train)
dtc = dtc.fit(X_train, y_train)
rfc = rfc.fit(X_train, y_train)
gb = gb.fit(X_train, y_train)

ypl = lr.predict(X_test)
yps = sc.predict(X_test)
ypd = dtc.predict(X_test)
ypr = rfc.predict(X_test)
ypg = gb.predict(X_test)

print("\nF1-Score of Various Models - Bag-of-Words")
print("LogisticRegression = ", f1_score(ypl, y_test))
print("DecisionTree = ", f1_score(ypd, y_test))
print("Gradient Boosting = ", f1_score(ypg, y_test))
print("SupportVector = ", f1_score(yps, y_test))
print("RandomForest = ", f1_score(ypr, y_test))

# Traing Models on vectors of TFIDF Vectorizer
lrt = lr.fit(Xt_train, y_train)
sct = sc.fit(Xt_train, y_train)
dtct = dtc.fit(Xt_train, y_train)
rfct = rfc.fit(Xt_train, y_train)
gbt = gb.fit(Xt_train, y_train)

yplt = lrt.predict(Xt_test)
ypst = sct.predict(Xt_test)
ypdt = dtct.predict(Xt_test)
yprt = rfct.predict(Xt_test)
ypgt = gbt.predict(Xt_test)

print("\nF1-Score of Various Models - TFIDF ")
print("LogisticRegression = ", f1_score(yplt, y_test))
print("DecisionTree = ", f1_score(ypdt, y_test))
print("Gradient Boosting = ", f1_score(ypgt, y_test))
print("SupportVector = ", f1_score(ypst, y_test))
print("RandomForest = ", f1_score(yprt, y_test))

