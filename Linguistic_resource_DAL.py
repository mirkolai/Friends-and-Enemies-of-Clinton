__author__ = 'mirko'

import csv
import re
import numpy

class DAL(object):

    dal={}
    dalstartwith={}

    def __init__(self):
        #http://www.cs.columbia.edu/~julia/papers/dict_of_affect/DictionaryofAffect
        csvfile = open('resources/DAL/DAL.txt', 'r')
        lines = csv.reader(csvfile,delimiter="\t")
        self.pattern_split = re.compile(r"\W+")
        for l in lines:
            key = l[0]
            value = float(l[1]),float(l[2]),float(l[3])
            if "*" in key:
                self.dalstartwith.setdefault(key, [])
                self.dalstartwith[key].append(value)
            else:
                self.dal.setdefault(key, [])
                self.dal[key].append(value)

        return


    def get_dal_sentiment(self,text):
        #ee=pleasantness, aa=activation, ii=imagery

        tokens= self.pattern_split.split(text.lower())
        ee=[0]
        aa=[0]
        ii=[0]
        for word in tokens:

            if word.lower() in self.dal:
                ee.append(self.dal[word.lower()][0][0])
                aa.append(self.dal[word.lower()][0][1])
                ii.append(self.dal[word.lower()][0][2])

            else:

                for key, val in self.dalstartwith.items():
                    if word.lower().startswith(key):
                        ee.append(val[0])
                        aa.append(val[1])
                        ii.append(val[2])
                        break

        return numpy.mean(ee),numpy.mean(aa),numpy.mean(ii),numpy.sum(ee),numpy.sum(aa),numpy.sum(ii),


if __name__ == '__main__':
    dal = DAL()
    pleasantness, activation, imagery,pleasantness_sum, activation_sum, imagery_sum=dal.get_dal_sentiment("@tedcruz And, #HandOverTheServer she wiped clean + 30k deleted emails, explains dereliction of duty/lies re #Benghazi,etc #tcot #SemST")
    print(pleasantness, activation, imagery,pleasantness_sum, activation_sum, imagery_sum)

