__author__ = 'mirko'

from sklearn.externals import joblib
from Tweet import make_tweet
import os.path
import pymysql
import config as cfg

class Database_manager(object):

    db=None
    cur=None

    def __init__(self):

        self.db = pymysql.connect(host=cfg.mysql['host'],
                 user=cfg.mysql['user'],
                 passwd=cfg.mysql['passwd'],
                 db=cfg.mysql['db'],
                 charset='utf8')
        self.cur = self.db.cursor()
        self.cur.execute('SET NAMES utf8mb4')
        self.cur.execute("SET CHARACTER SET utf8mb4")
        self.cur.execute("SET character_set_connection=utf8mb4")
        self.db.commit()

    def return_test_tweets(self, filter=None):


        if os.path.isfile('tweets_test'+filter+'.pkl') :
            tweets= joblib.load('tweets_test'+filter+'.pkl')
            return tweets


        tweets=[]
        if filter != None:
            self.cur.execute("SELECT `ID`, `Target`, `Tweet`, `Stance`,`Opinion Towards`, `Sentiment` FROM `data_test` where target=%s  order by id",(filter))
        else:
            self.cur.execute("SELECT `ID`, `Target`, `Tweet`, `Stance`,`Opinion Towards`, `Sentiment` FROM `data_test`  order by id")


        for tweet in self.cur.fetchall():


                sentiment = return_labeled_sentiment(tweet[5])
                opiniontowards= return_labeld_opinion_toward(tweet[4])

                this_tweet=make_tweet(tweet[0], tweet[2],tweet[1], opiniontowards,sentiment, stance= tweet[3])

                tweets.append(this_tweet)

        joblib.dump(tweets, 'tweets_test'+filter+'.pkl')

        return tweets

    def return_train_tweets(self,filter=None):


        if os.path.isfile('tweets_train'+filter+'.pkl') :
            tweets= joblib.load('tweets_train'+filter+'.pkl')
            return tweets


        tweets=[]
        if filter != None:
            self.cur.execute("SELECT `ID`, `Target`, `Tweet`, `Stance`,`Opinion Towards`, `Sentiment` FROM `data_training` where target=%s  order by id",(filter))
        else:
            self.cur.execute("SELECT `ID`, `Target`, `Tweet`, `Stance`,`Opinion Towards`, `Sentiment` FROM `data_training`  order by id")

        for tweet in self.cur.fetchall():

                sentiment= return_labeled_sentiment(tweet[5])
                opiniontowards= return_labeld_opinion_toward(tweet[4])


                this_tweet=make_tweet(tweet[0], tweet[2],tweet[1],opiniontowards,sentiment,stance= tweet[3])

                tweets.append(this_tweet)


        joblib.dump(tweets, 'tweets_train'+filter+'.pkl')

        return tweets

    def return_tweets_irony(self,filter=None):


        if os.path.isfile('tweets_irony'+filter+'.pkl') :
            tweets= joblib.load('tweets_irony'+filter+'.pkl')
            return tweets


        tweets=[]
        if filter != None:
            self.cur.execute("SELECT `ID`, `Target`, `Tweet`, `Irony` FROM `data_ironic` where target=%s order by id",(filter))
        else:
            self.cur.execute("SELECT `ID`, `Target`, `Tweet`, `Irony` FROM `data_ironic` order by id")

        for tweet in self.cur.fetchall():


                irony=tweet[3]
                truesentiment  = None
                opiniontowards = None

                this_tweet=make_tweet(tweet[0], tweet[2], tweet[1], opiniontowards, truesentiment,irony=irony)

                tweets.append(this_tweet)


        joblib.dump(tweets, 'tweets_irony'+filter+'.pkl')

        return tweets




def return_labeled_sentiment(sentiment):

    labeledsentiment=0
    if sentiment=="neg":
        labeledsentiment=-1
    elif sentiment=="pos":
        labeledsentiment=1
    elif sentiment=="other":
        labeledsentiment=0

    return labeledsentiment

def return_labeld_opinion_toward(opiniontoward):
    labeledopiniontowards=0
    if opiniontoward=="1.  The tweet explicitly expresses opinion about the target, a part of the target, or an aspect of the target.":
        labeledopiniontowards=1
    elif opiniontoward== "2. The tweet does NOT expresses opinion about the target but it HAS opinion about something or someone other than the target.":
        labeledopiniontowards=2
    elif opiniontoward== "3.  The tweet is not explicitly expressing opinion. (For example, the tweet is simply giving information.)":
        labeledopiniontowards=3

    return labeledopiniontowards




def make_database_manager():
    database_manager = Database_manager()

    return database_manager




