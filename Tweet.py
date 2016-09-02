__author__ = 'mirko'

from Linguistic_resource_LIWC import LIWC
from Linguistic_resource_AFINN import AFINN
from Linguistic_resource_DAL import DAL
from Linguistic_resource_HL import HL
from Linguistic_resource_wordlist import WORDLIST

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()


liwc = LIWC()
afinn = AFINN()
dal = DAL()
hl = HL()
wordlist = WORDLIST()


class Tweet(object):

    tweet_id=''
    text=''
    target=''
    stance=0

    def __init__(self, tweet_id, text, target, opiniontowards, sentiment, stance=None, irony=None):


        self.tweet_id=tweet_id

        self.text=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' URL ', text.replace("#SemST",""))
        self.text_no_stop_word=" ".join([word for word in self.text if word not in stopwords.words('english')])
        self.text_hashtag_and_screen_name_slitted=wordlist.ParseSentence(self.text)

        self.tokens = nltk.word_tokenize(self.text_hashtag_and_screen_name_slitted)
        self.tokens_sentiment = hl.get_HL_sentiment_tokens(self.tokens)
        self.lemmas = [ porter_stemmer.stem(token) for token in self.tokens]
        self.pos = [ token[1] for token in nltk.pos_tag(self.tokens)]

        self.sentimentafinn=afinn.get_afinn_sentiment(text)
        self.sentimentdal=dal.get_dal_sentiment(text)
        self.sentimenthl=hl.get_HL_sentiment(text)
        self.sentimentliwc=liwc.get_liwc_sentiment(text)

        self.liwc_text=liwc.get_liwc_text(self.tokens)
        self.liwc_function=liwc.get_liwc_functions(self.tokens)

        self.irony = irony

        self.target=target
        self.stance=stance

        self.opiniontowards=opiniontowards
        self.sentiment=sentiment



def make_tweet(tweet_id, text, target, labeledopiniontowards,labeledsentiment, stance=None, irony=None):

    tweet = Tweet(tweet_id, text, target, labeledopiniontowards,labeledsentiment, stance=stance, irony=irony)

    return tweet



