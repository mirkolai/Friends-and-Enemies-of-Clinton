__author__ = 'mirko'

import codecs
import re

class HL(object):


    def __init__(self):
        self.hl = {}
        #https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
        file=codecs.open('resources/HL/HL-negatives.txt', encoding='UTF-8')
        for line in file:
            word = line.strip()
            self.hl[word] = int(-1)
        file.close()
        file=codecs.open('resources/HL/HL-positives.txt', encoding='UTF-8')
        for line in file:
            word = line.strip()
            self.hl[word] = int(1)
        file.close()


        self.pattern_split = re.compile(r"\W+")

        return

    def get_HL_sentiment(self,text):
        sentiments=0
        words = self.pattern_split.split(text.lower())
        for word in words:
            if word in self.hl:
                sentiments+=self.hl[word]

        return sentiments

    def get_HL_sentiment_tokens(self,tokens):

        sentiment_token=[]
        for word in tokens:
            sentiment_word=0
            if word.lower() in self.hl:
                sentiment_word=self.hl[word.lower()]
            sentiment_token.append(sentiment_word)
        return sentiment_token


if __name__ == '__main__':
    hl = HL()
    sentiment=hl.get_HL_sentiment("@tedcruz And, #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #SemST")
    print(sentiment)

