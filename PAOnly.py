import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv('DatasetsCSV/charliehebdo.csv')

labels=df.type

x_train,x_test,y_train,y_test=train_test_split(df['tweet'], labels, test_size=0.2, random_state=7)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

pa= PassiveAggressiveClassifier()
pa.fit(tfidf_train, y_train)

y_pred=pa.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print('Accuracy: '+str(round(score*100,2))+'%')

print(confusion_matrix(y_test,y_pred, labels=['rumor','non_rumor']))


cnf_matrix = confusion_matrix(y_test,y_pred, labels=['rumor','non_rumor'])

class_names=[0,1]


fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)


sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
