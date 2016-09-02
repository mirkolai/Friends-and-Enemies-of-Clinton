__author__ = 'mirko'

import csv
import re

class LIWC(object):

    liwc={}
    liwicstartwith={}

    def __init__(self):
	#http://liwc.wpengine.com/
        csvfile = open('resources/LIWC/LIWC.csv', 'r')
        lines = csv.reader(csvfile)
        self.pattern_split = re.compile(r"\W+")
        for l in lines:
            key = l[0]
            value = l[1]
            if "*" in key:
                self.liwicstartwith.setdefault(key, [])
                self.liwicstartwith[key].append(value)
            else:
                self.liwc.setdefault(key, [])
                self.liwc[key].append(value)

        return

    def get_liwc_functions(self,tokens):

        liwc_list=[]

        for word in tokens:
            this_word=[]
            if word.lower() in self.liwc:

                for v in self.liwc[word.lower()]:
                    this_word.append(v)

            else:

                for key, val in self.liwicstartwith.items():
                    if word.lower().startswith(key):
                        for v in val:
                            this_word.append(v)
                        break
            liwc_list.append(this_word)

        return liwc_list

    def get_liwc_text(self,tokens):

        liwc_text=""

        for word in tokens:

            if word.lower() in self.liwc:

                for v in self.liwc[word.lower()]:
                    liwc_text=liwc_text+" "+("feature_liwc_function_"+v)

            else:

                for key, val in self.liwicstartwith.items():
                    if word.lower().startswith(key):
                        for v in val:
                            liwc_text=liwc_text+" "+("feature_liwc_function_"+v)
                        break


        return liwc_text



    def get_liwc_sentiment(self, text):
        liwc_text=""
        words = self.pattern_split.split(text.lower())
        sentiment=0
        for word in words:

            if word.lower() in self.liwc:

                for v in self.liwc[word.lower()]:
                    if v =="posemo":
                        sentiment=+1
                    if v =="negemo":
                        sentiment=-1

            else:

                for key, val in self.liwicstartwith.items():
                    if word.lower().startswith(key):
                        for v in val:
                            if v =="posemo":
                                sentiment=+1
                            if v =="negemo":
                                sentiment=-1
                        break


        return sentiment

if __name__ == '__main__':
    liwc = LIWC()
    sentiment=liwc.get_liwc_sentiment("@tedcruz And, #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #SemST")
    print(sentiment)
