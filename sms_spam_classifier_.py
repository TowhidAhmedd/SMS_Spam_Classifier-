
import numpy as np
import pandas as pd
# from google.colab import drive
# d=drive.mount('/content/drive')
from google.colab import files
uploaded = files.upload()
ds = pd.read_csv("spam (1).csv", encoding='latin-1')
ds.head(10)

# 1. Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model Building
# 5. Evaluation
# 6. Impovement
# 7. Website
# 8. Deploy

"""## 1. Data **cleaning** *"""

ds.info()

# drop last 3 columns
ds.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
ds.sample(5)

ds.rename(columns={'v1':'target','v2':'text'}, inplace=True) # rename the column
ds.sample(5)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
ds['target'] = encoder.fit_transform(ds['target']) # Encoding for prepare data
ds.head(4)

ds.isnull().sum() # check missing value

ds.duplicated().sum() # check duplicated value



ds = ds.drop_duplicates(keep='first') # remove dupicates without first duplicated value
print(ds.duplicated().sum())
print(ds.shape)

"""# ***2. EDA***"""

print("Dataset:", ds.head(5))

print("Unique_Value: ", ds['target'].value_counts())

print("Value:", ds.value_counts())

import matplotlib.pyplot as plt  # piechar for better visulization
plt.pie(ds['target'].value_counts(), labels=['ham','spam'],autopct="%0.3f")
plt.show()

"""# Here data is imbalanced"""

import nltk
nltk.download('punkt_tab')
ds['num_characters'] = ds['text'].apply(len) # total character
ds['num_words'] =     ds['text'].apply(lambda x:len(nltk.word_tokenize(x))) # total word
ds['num_sentances'] = ds['text'].apply(lambda x:len(nltk.sent_tokenize(x))) # total sentance
ds.head(5)

ds.describe()

# only ham
ham_ds = ds[ds['target'] == 0]
ham_ds.describe()

# only spam
spam_ds = ds[ds['target'] == 1]
spam_ds.describe()

import seaborn as sns
plt.figure(figsize=(12,6))
sns.histplot(ds[ds['target']==0] ['num_characters'])
sns.histplot(ds[ds['target']==1] ['num_characters'],color='red')

sns.pairplot(ds,hue='target')

sns.heatmap(ds[['target', 'num_characters', 'num_words', 'num_sentances']].corr(), annot=True) # Correlation visualization

"""3. Data Preprocessing

      -> Lower Case

      -> Tokenization

      -> Removing Special Characters
      
      -> Stemming
"""

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import string

import nltk
# nltk.download('punkt') # Already downloaded in the previous cell
# nltk.download('stopwords') # Already downloaded in the previous cell
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_test(test):
  text = test.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text: # character and number
    if i.isalnum():
      y.append(i)
  text = y[:]
  y.clear()

  for i in text: # loop stopwards
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text: # stemming
    y.append(ps.stem(i))

  text = y[:]
  y.clear()

  return " ".join(text)

transform_test(' H did you like dancing m presentation on ML and Dl ## % 6 ^ ')

ds['transformed_text'] = ds['text'].apply(transform_test)
ds.head()

from wordcloud import WordCloud # frequent work show bigger size
wc = WordCloud(width=500, height=500, min_font_size=10,
               background_color='white')

spam_wc = wc.generate(ds[ds['target']==1]['transformed_text'].str.cat(sep=" "))

plt.imshow(spam_wc)

spam_corpus=[] # store all word from spam transformed_text
for msg in ds[ds['target']==1]['transformed_text'].tolist():
  for word in msg.split():
    spam_corpus.append(word)

from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(20))[0],
            y=pd.DataFrame(Counter(spam_corpus).most_common(20))[1])
plt.xticks(rotation='vertical')
plt.show()
 # count word frequency



ham_corpus=[] # store all word from spam transformed_text
for msg in ds[ds['target']==0]['transformed_text'].tolist():
  for word in msg.split():
    ham_corpus.append(word)

from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(20))[0],
            y=pd.DataFrame(Counter(ham_corpus).most_common(20))[1])
plt.xticks(rotation='vertical')
plt.show()



"""# ***4. Model Building***"""

from re import T
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
Tfidf = TfidfVectorizer(max_features=3000)
# X = cv.fit_transform(ds['transformed_text']).toarray()
X = Tfidf.fit_transform(ds['transformed_text']).toarray()
X.shape

# from sklearn.preprocessing import MinMaxScaler # decrease model performance
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# X

# appending the num_character col to X
X = np.hstack((X,ds['num_characters'].values.reshape(-1,1)))
# X = np.hstack((X,ds['num_words'].values.reshape(-1,1)))
X

y = ds['target'].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)

from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(x_train,y_train)
y_pred1 = gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

mnb.fit(x_train,y_train)
y_pred2 = mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

bnb.fit(x_train,y_train)
y_pred3 = bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))

# tfidf -> mnb

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)

abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs ={
    'SVC' : svc,
    'KN' : knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'xgb': xgb
}

def train_classifier(clf,x_train,y_train,x_test,y_test):
  clf.fit(x_train,y_train)
  y_pred = clf.predict(x_test)
  accuracy = accuracy_score(y_test,y_pred)
  precision = precision_score(y_test,y_pred)

  return accuracy, precision

train_classifier(svc,x_train,y_train,x_test,y_test)

accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
  current_accuracy,current_precision = train_classifier(clf,x_train,y_train,x_test,y_test)

  print("For ",name)
  print("Accuracy - ",current_accuracy)
  print("Precision - ",current_precision)

  accuracy_scores.append(current_accuracy)
  precision_scores.append(current_precision)

performance_df = pd.DataFrame({'Algorithm':clfs.keys(),
'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',
ascending=False)
performance_df

performance_df1 = pd.melt(performance_df,id_vars='Algorithm')
performance_df1

performance_df_melted = performance_df.melt(id_vars='Algorithm', var_name='Metric', value_vars=['Accuracy', 'Precision'], value_name='Value')
sns.catplot(x='Algorithm', y='Value', hue='Metric', data=performance_df_melted, kind='bar', height=5)
plt.xticks(rotation='vertical')
plt.ylim(.5, 1.0)
plt.show()

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)

new_df = performance_df.merge(temp_df,on='Algorithm')

new_df_scaled = new_df.merge(temp_df,on='Algorithm')

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)

new_df_scaled.merge(temp_df,on='Algorithm')

performance_df.merge(temp_df,on='Algorithm')

# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')

voting.fit(x_train,y_train)

VotingClassifier(estimators=[('svm',
                              SVC(gamma=1.0, kernel='sigmoid',
                                  probability=True)),
                             ('nb', MultinomialNB()),
                             ('et',
                              ExtraTreesClassifier(n_estimators=50,
                                                   random_state=2))],
                 voting='soft')

y_pred = voting.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))

# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()

from sklearn.ensemble import StackingClassifier

clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))



import pickle
pickle.dump(Tfidf,open('vectorizer.pkl','wb'))
from google.colab import files
files.download('vectorizer.pkl')

pickle.dump(mnb,open('model.pkl','wb'))
from google.colab import files
files.download('model.pkl')

# pickle.dump(mnb,open('model.pkl','wb'))
# from google.colab import files
# files.download('model.pkl')



