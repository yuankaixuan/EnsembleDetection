import json
import os
import pandas as pd

nr_path ='Datasets/phemernrdataset/pheme-rnr-dataset/charliehebdo/non-rumours'
r_path = 'Datasets/phemernrdataset/pheme-rnr-dataset/charliehebdo/rumours'

nr = os.listdir(nr_path)
r = os.listdir(r_path)

tweet_texts=[]
type = []

for tweets in nr:
  source = os.listdir(nr_path+'/'+tweets+'/source-tweet')
  with open(nr_path+'/'+tweets+'/source-tweet/'+source[0]) as f:
    source_tweet = json.load(f)
    print(source_tweet['text']+',non_rumor')
    tweet_texts +=[source_tweet['text']]
    type+=['non_rumor']

for tweets in r:
  source = os.listdir(r_path + '/' + tweets + '/source-tweet')
  with open(r_path + '/' + tweets + '/source-tweet/' + source[0]) as f:
    source_tweet = json.load(f)
    print(source_tweet['text']+',rumor')
    tweet_texts +=[source_tweet['text']]
    type+=['rumor']

tweets_and_type = zip(tweet_texts,type)

df = pd.DataFrame(tweets_and_type, columns=['tweet','type'])

out = df.to_csv()

df.to_csv("DatasetsCSV/charliehebdo.csv")