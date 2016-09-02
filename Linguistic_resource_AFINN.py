__author__ = 'mirko'
import codecs
import re

class AFINN(object):

    afinn={}

    def __init__(self):
        self.afinn = {}
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('resources/AFINN/AFINN-111.txt', encoding='UTF-8')
        for line in file:
            word, score = line.strip().split('\t')
            self.afinn[word] = int(score)

        self.pattern_split = re.compile(r"\W+")

        return

    def get_afinn_sentiment(self,text):

        sentiments=0
        words = self.pattern_split.split(text.lower())
        for word in words:
            if word in self.afinn:
                sentiments+=self.afinn[word]


        return sentiments


if __name__ == '__main__':
    afinn = AFINN()
    sentiment=afinn.get_afinn_sentiment("@tedcruz And, #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #SemST")
    print(sentiment)

