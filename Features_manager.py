__author__ = 'mirko'

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import re
from scipy.sparse import csr_matrix, hstack



class Features_manager(object):

    def __init__(self):


        return

    def get_stance(self,tweets):

        stance  = []

        for tweet in tweets:
            stance.append(tweet.stance)


        return stance

    def get_opinion_towards(self,tweets):

        opiniontowards  = []

        for tweet in tweets:
            opiniontowards.append(tweet.opiniontowards)

        return opiniontowards

    def get_sentiment(self,tweets):

        sentiment  = []

        for tweet in tweets:
            sentiment.append(tweet.sentiment)

        return sentiment



    #features extractor
    def create_feature_space(self,tweets,featureset=None,
                             train_tweets=None,
                             target_alias=None,target_alias_pronoums=None,target_alias_party=None,target_alias_opposite_in_party=None,target_alias_opposite_out_party=None,
                             target_alias_train=None,target_alias_pronoums_train=None,target_alias_party_train=None,target_alias_opposite_in_party_train=None,target_alias_opposite_out_party_train=None):


        if target_alias_train==None or\
                        target_alias_opposite_in_party_train==None or\
                        target_alias_opposite_out_party_train==None or\
                        target_alias_pronoums_train==None or\
                        target_alias_party_train==None:
            target_alias_train=target_alias
            target_alias_opposite_in_party_train=target_alias_opposite_in_party
            target_alias_opposite_out_party_train=target_alias_opposite_out_party
            target_alias_pronoums_train=target_alias_pronoums
            target_alias_party_train=target_alias_party


        global_featureset={
            "BoW"  : self.get_BoW_features(tweets,train_tweets),
            "hashtag" : self.get_hashtag_features(tweets,train_tweets),
            "numhashtag" : self.get_numhashtag_features(tweets,train_tweets),
            "mention"  : self.get_mention_features(tweets,train_tweets),
            "nummention"  : self.get_nummention_features(tweets,train_tweets),
            "BoWinHashtag" :  self.get_BoWinHashtag_features(tweets,train_tweets,
                                                                  target_alias,target_alias_pronoums,target_alias_party,target_alias_opposite_in_party,target_alias_opposite_out_party,
                                                                  target_alias_train,target_alias_pronoums_train,target_alias_party_train,target_alias_opposite_in_party_train,target_alias_opposite_out_party_train),
            "BoWinMention" : self.get_BoWinMention_features(tweets,train_tweets),
            "sentiment_labeled" : self.get_sentiment_labeled_features(tweets,train_tweets),

            "sentiment_afinn" : self.get_sentiment_afinn_features(tweets,train_tweets),
            "sentiment_liwc" : self.get_sentiment_LIWC_features(tweets,train_tweets),
            "sentiment_hl" : self.get_sentiment_HL_features(tweets,train_tweets),
            "sentiment_dal" : self.get_sentiment_DAL_features(tweets,train_tweets),
            "target_in_tweet_labeled" : self.get_target_in_tweet_labeled_features(tweets,train_tweets),
            "target_in_tweet" : self.get_target_in_tweet_features(tweets,train_tweets,
                                                                  target_alias,target_alias_pronoums,target_alias_party,target_alias_opposite_in_party,target_alias_opposite_out_party,
                                                                  target_alias_train,target_alias_pronoums_train,target_alias_party_train,target_alias_opposite_in_party_train,target_alias_opposite_out_party_train),
            "target_in_tweet_alternative" : self.get_target_in_tweet_alternative_features(tweets,train_tweets,
                                                                  target_alias,target_alias_pronoums,target_alias_party,target_alias_opposite_in_party,target_alias_opposite_out_party,
                                                                  target_alias_train,target_alias_pronoums_train,target_alias_party_train,target_alias_opposite_in_party_train,target_alias_opposite_out_party_train),
            "presence_of_not" : self.get_presence_of_not_features(tweets,train_tweets),
            "target_and_no_in_hashtag"  : self.get_target_and_no_in_hashtag_features(tweets,train_tweets,target_alias,target_alias_opposite_in_party,target_alias_train,target_alias_opposite_in_party_train),
            "target_position" : self.get_target_position_features(tweets,train_tweets,
                                                                  target_alias,target_alias_pronoums,target_alias_party,target_alias_opposite_in_party,target_alias_opposite_out_party,
                                                                  target_alias_train,target_alias_pronoums_train,target_alias_party_train,target_alias_opposite_in_party_train,target_alias_opposite_out_party_train),
            "liwc" : self.get_liwc_features(tweets,train_tweets),
            "BoTargetPOS" : self.get_BoTargetPOS_features(tweets,train_tweets,
                                                                  target_alias,target_alias_pronoums,target_alias_party,target_alias_opposite_in_party,target_alias_opposite_out_party,
                                                                  target_alias_train,target_alias_pronoums_train,target_alias_party_train,target_alias_opposite_in_party_train,target_alias_opposite_out_party_train),
            "BoTargetNegatePOS" : self.get_BoTargetNegatePOS_features(tweets,train_tweets,
                                                                  target_alias,target_alias_pronoums,target_alias_party,target_alias_opposite_in_party,target_alias_opposite_out_party,
                                                                  target_alias_train,target_alias_pronoums_train,target_alias_party_train,target_alias_opposite_in_party_train,target_alias_opposite_out_party_train),
            "conditional": self.get_conditional_features(tweets,train_tweets),
            "punctuation_marks": self.get_puntuaction_marks_features(tweets,train_tweets),
            "irony": self.get_irony_features(tweets,train_tweets),

        }

        all_feature_names=[]
        all_X=[]
        all_Y=[]

        for key in featureset:
            X,Y,feature_names=global_featureset[key]
            all_feature_names=np.concatenate((all_feature_names,feature_names))
            if all_X!=[]:
                all_X=csr_matrix(hstack((all_X,X)))
                all_Y=csr_matrix(hstack((all_Y,Y)))
            else:
                all_X=X
                all_Y=Y

        return all_X,all_Y, all_feature_names

    def get_BoW_features(self, tweets, train_tweets=None):

        tfidfVectorizer = CountVectorizer(ngram_range=(1,3),
                                          stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        feature  = []
        for tweet in tweets:

            feature.append(tweet.text)

        if train_tweets!=None:

            feature_train  = []
            for tweet in train_tweets:

                feature_train.append(tweet.text)

            tfidfVectorizer = tfidfVectorizer.fit(feature_train)
            X_train = tfidfVectorizer.transform(feature_train)
            X_test = tfidfVectorizer.transform(feature)
            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

        else:
            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()

            return X, feature_names

    def get_hashtag_features(self, tweets, train_tweets=None):


        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          #stop_words="english",
                                          lowercase=False, #true 0.507 false 0.51
                                          binary=True,
                                          max_features=500000)

        feature  = []
        for tweet in tweets:
            feature.append(' '.join(re.findall(r"#(\w+)", tweet.text)))


        if train_tweets!=None:

            feature_train  = []
            for tweet in train_tweets:
                feature_train.append(' '.join(re.findall(r"#(\w+)", tweet.text)))

            tfidfVectorizer = tfidfVectorizer.fit(feature_train)
            X_train = tfidfVectorizer.transform(feature_train)
            X_test = tfidfVectorizer.transform(feature)
            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

        else:
            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()


            return X, feature_names

    def get_numhashtag_features(self, tweets, train_tweets=None):

        feature  = []

        for tweet in tweets:
            feature.append(len(re.findall(r"#(\w+)", tweet.text)))

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                feature_train.append(len(re.findall(r"#(\w+)", tweet.text)))

            return csr_matrix(np.vstack(feature_train)),csr_matrix(np.vstack(feature)),["feature_numhashtag"]

        else:


            return csr_matrix(np.vstack(feature)),["feature_numhashtag"]

    def get_mention_features(self, tweets, train_tweets=None):


        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          #stop_words="english",
                                          lowercase=False, #true 0.507 false 0.51
                                          binary=True,
                                          max_features=500000)

        feature  = []
        for tweet in tweets:
            feature.append(' '.join(re.findall(r"@(\w+)", tweet.text)))


        if train_tweets!=None:

            feature_train  = []
            for tweet in train_tweets:
                feature_train.append(' '.join(re.findall(r"#(\w+)", tweet.text)))

            tfidfVectorizer = tfidfVectorizer.fit(feature_train)
            X_train = tfidfVectorizer.transform(feature_train)
            X_test = tfidfVectorizer.transform(feature)
            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

        else:
            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()


            return X, feature_names

    def get_nummention_features(self, tweets, train_tweets=None):

        feature  = []

        for tweet in tweets:
            feature.append(len(re.findall(r"@(\w+)", tweet.text)))

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                feature_train.append(len(re.findall(r"@(\w+)", tweet.text)))

            return csr_matrix(np.vstack(feature_train)),csr_matrix(np.vstack(feature)),["feature_nummention"]

        else:


            return csr_matrix(np.vstack(feature)),["feature_nummention"]

    def get_BoWinHashtag_features(self,tweets, train_tweets=None,
                                     target=None,target_pronoums=None,target_party=None,target_opposite_in_party=None,target_opposite_out_party=None,
                                     target_train=None,target_pronoums_train=None,target_party_train=None,target_opposite_in_party_train=None,target_opposite_out_party_train=None):


        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          #stop_words="english",
                                          lowercase=True, #true 4389  false 0.4356
                                          binary=True,
                                          max_features=500000)

        feature  = []
        for tweet in tweets:
            text_tag=' '.join([' '.join(re.findall('[A-Z](?:[a-z]{1,}|[A-Z]*)', tag)) for tag in re.findall(r"#(\w+)", tweet.text)])
            text_tag=text_tag.lower()
            for t in target:
                text_tag=text_tag.replace(t.lower()," feature_bow_target_TARGET ")
            for t in target_pronoums:
                text_tag=text_tag.replace(t.lower()," feature_bow_target_TARGET_PRONOUN ")
            for t in target_party:
                text_tag=text_tag.replace(t.lower()," feature_bow_target_PARTY ")
            for t in target_opposite_in_party:
                text_tag=text_tag.replace(t.lower()," feature_bow_target_IN_PARTY ")
            for t in target_opposite_out_party:
                text_tag=text_tag.replace(t.lower()," feature_bow_target_out_party ")

            feature.append(text_tag)

        if train_tweets!=None:

            feature_train  = []
            for tweet in train_tweets:
                text_tag=' '.join([' '.join(re.findall('[A-Z](?:[a-z]{1,}|[A-Z]*)', tag)) for tag in re.findall(r"#(\w+)", tweet.text)])
                text_tag=text_tag.lower()
                for t in target_train:
                    text_tag=text_tag.replace(t.lower()," feature_bow_target_TARGET ")
                for t in target_pronoums_train:
                    text_tag=text_tag.replace(t.lower()," feature_bow_target_TARGET_PRONOUN ")
                for t in target_party_train:
                    text_tag=text_tag.replace(t.lower()," feature_bow_target_PARTY ")
                for t in target_opposite_in_party_train:
                    text_tag=text_tag.replace(t.lower()," feature_bow_target_IN_PARTY ")
                for t in target_opposite_out_party_train:
                    text_tag=text_tag.replace(t.lower()," feature_bow_target_out_party ")

                feature_train.append(text_tag)

            tfidfVectorizer = tfidfVectorizer.fit(feature_train)
            X_train = tfidfVectorizer.transform(feature_train)
            X_test = tfidfVectorizer.transform(feature)
            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

        else:
            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()


            return X, feature_names

    def get_BoWinMention_features(self, tweets, train_tweets=None):


        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          #stop_words="english",
                                          lowercase=True, #true 4389  false 0.4356
                                          binary=True,
                                          max_features=500000)

        feature  = []
        for tweet in tweets:

            feature.append(' '.join([' '.join(re.findall('[A-Z](?:[a-z]{1,}|[A-Z]*)', tag))
                 for tag in re.findall(r"@(\w+)", tweet.text)]))


        if train_tweets!=None:

            feature_train  = []
            for tweet in train_tweets:


                feature_train.append(' '.join([' '.join(re.findall('[A-Z](?:[a-z]{1,}|[A-Z]*)', tag))
                 for tag in re.findall(r"#(\w+)", tweet.text)]))

            tfidfVectorizer = tfidfVectorizer.fit(feature_train)
            X_train = tfidfVectorizer.transform(feature_train)
            X_test = tfidfVectorizer.transform(feature)
            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

        else:
            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()


            return X, feature_names

    def get_irony_features(self,tweets, train_tweets=None):
        #'NOTIRONIC' or 'IRONIC'
        irony  = []

        for tweet in tweets:
            irony.append( 0 if tweet.irony=="NOTIRONIC" else 1)

        if train_tweets!=None:
            irony_train  = []
            for tweet in train_tweets:
                irony_train.append( 0 if tweet.irony=="NOTIRONIC" else 1)

            return csr_matrix(np.vstack(irony_train)),csr_matrix(np.vstack(irony)),["feature_irony"]

        else:


            return csr_matrix(np.vstack(irony)),["feature_irony"]

    def get_sentiment_labeled_features(self,tweets, train_tweets=None):

        sentiment  = []

        for tweet in tweets:
            sentiment.append(tweet.sentiment)

        if train_tweets!=None:
            sentiment_train  = []
            for tweet in train_tweets:
                sentiment_train.append(tweet.sentiment)

            return csr_matrix(np.vstack(sentiment_train)),csr_matrix(np.vstack(sentiment)),["feature_sentiment_labeled"]

        else:


            return csr_matrix(np.vstack(sentiment)),["feature_sentiment_labeled"]

    def get_sentiment_afinn_features(self,tweets, train_tweets=None):

        sentiment  = []

        for tweet in tweets:

            sentiment.append(tweet.sentimentafinn)

        if train_tweets!=None:
            sentiment_train  = []
            for tweet in train_tweets:
                sentiment_train.append(tweet.sentimentafinn)

            return csr_matrix(np.vstack(sentiment_train)),csr_matrix(np.vstack(sentiment)),["feature_sentiment_afinn"]

        else:


            return csr_matrix(np.vstack(sentiment)),["feature_sentiment_afinn"]

    def get_sentiment_LIWC_features(self,tweets, train_tweets=None):

        sentiment  = []

        for tweet in tweets:

            sentiment.append(tweet.sentimentliwc)

        if train_tweets!=None:
            sentiment_train  = []
            for tweet in train_tweets:
                sentiment_train.append(tweet.sentimentliwc)

            return csr_matrix(np.vstack(sentiment_train)),csr_matrix(np.vstack(sentiment)),["feature_sentiment_liwc"]

        else:


            return csr_matrix(np.vstack(sentiment)),["feature_sentiment_liwc"]

    def get_sentiment_HL_features(self,tweets, train_tweets=None):

        sentiment  = []

        for tweet in tweets:

            sentiment.append(tweet.sentimenthl)

        if train_tweets!=None:
            sentiment_train  = []
            for tweet in train_tweets:
                sentiment_train.append(tweet.sentimenthl)

            return csr_matrix(np.vstack(sentiment_train)),csr_matrix(np.vstack(sentiment)),["feature_sentiment_HL"]

        else:


            return csr_matrix(np.vstack(sentiment)),["feature_sentiment_HL"]

    def get_sentiment_DAL_features(self,tweets, train_tweets=None):

        sentiment  = []

        for tweet in tweets:

            sentiment.append(tweet.sentimentdal)

        if train_tweets!=None:
            sentiment_train  = []
            for tweet in train_tweets:
                sentiment_train.append(tweet.sentimentdal)
            return csr_matrix(np.vstack(sentiment_train)),csr_matrix(np.vstack(sentiment)),["feature_sentiment_dal_pleasantness", "feature_sentiment_dal_activation", "feature_sentiment_dal_imagery","feature_sentiment_dal_pleasantness_sum", "feature_sentiment_dal_activation_sum", "feature_sentiment_dal_imagery_sum"]

        else:


            return csr_matrix(np.vstack(sentiment)),["feature_sentiment_dal_pleasantness", "feature_sentiment_dal_activation", "feature_sentiment_dal_imagery"]

    def get_target_in_tweet_features(self,tweets, train_tweets=None,
                                     target=None,target_pronoums=None,target_party=None,target_opposite_in_party=None,target_opposite_out_party=None,
                                     target_train=None,target_pronoums_train=None,target_party_train=None,target_opposite_in_party_train=None,target_opposite_out_party_train=None):

        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          binary=False,
                                          max_features=500000)

        feature  = []
        for tweet in tweets:
            text=""
            text_tweet=re.sub(r'[^\w\s]',' ',tweet.text_hashtag_and_screen_name_slitted)
            text_tweet=text_tweet.split(" ")

            for t in target:
                if t.lower() in text_tweet:
                    text+=" feature_target_in_tweet_xTARGETx "

            for t in target_pronoums:
                if t.lower() in text_tweet:
                    text+=" feature_target_in_tweet_xPRONOUMSx "

            for t in target_party:
                if t.lower() in text_tweet:
                    text+=" feature_target_in_tweet_xPARTYx "

            for t in target_opposite_in_party:
                if t.lower() in text_tweet:
                    text+=" feature_target_in_tweet_xOPPOSITEINPARTYTARGETx "

            for t in target_opposite_out_party:
                if t.lower() in text_tweet:
                    text+=" feature_target_in_tweet_xOPPOSITEOUTPARTYTARGETx "

            if text=="":
                text+=" feature_target_in_tweet_xNOTHINGx "

            feature.append(text)


        if train_tweets!=None:

            feature_train  = []
            for tweet in train_tweets:
                text=""
                text_tweet=re.sub(r'[^\w\s]',' ',tweet.text_hashtag_and_screen_name_slitted)
                text_tweet=text_tweet.split(" ")
                for t in target_train:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xTARGETx "

                for t in target_pronoums_train:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xPRONOUMSx "

                for t in target_party_train:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xPARTYx "

                for t in target_opposite_in_party_train:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xOPPOSITEINPARTYTARGETx "

                for t in target_opposite_out_party_train:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xOPPOSITEOUTPARTYTARGETx "

                if text=="":
                    text+=" feature_target_in_tweet_xNOTHINGx "

                feature_train.append(text)

            tfidfVectorizer = tfidfVectorizer.fit(feature_train)
            X_train = tfidfVectorizer.transform(feature_train)
            X_test = tfidfVectorizer.transform(feature)
            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

        else:
            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()


            return X, feature_names

    def get_target_in_tweet_alternative_features(self,tweets, train_tweets=None,
                                     target=None,target_pronoums=None,target_party=None,target_opposite_in_party=None,target_opposite_out_party=None,
                                     target_train=None,target_pronoums_train=None,target_party_train=None,target_opposite_in_party_train=None,target_opposite_out_party_train=None):

        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          binary=False,
                                          max_features=500000)

        feature  = []
        for tweet in tweets:
            text=""
            text_tweet=re.sub(r'[^\w\s]',' ',tweet.text_hashtag_and_screen_name_slitted)
            text_tweet=text_tweet.split(" ")

            for t in target:
                if t.lower() in text_tweet:
                    text+=" feature_target_in_tweet_xTARGETx "

            for t in target_pronoums:
                if t.lower() in text_tweet:
                    text+=" feature_target_in_tweet_xTARGETx "

            for t in target_party:
                if t.lower() in text_tweet:
                    text+=" feature_target_in_tweet_xTARGETx "

            for t in target_opposite_in_party:
                if t.lower() in text_tweet:
                    text+=" feature_target_in_tweet_xOPPOSITEPARTYTARGETx "

            for t in target_opposite_out_party:
                if t.lower() in text_tweet:
                    text+=" feature_target_in_tweet_xOPPOSITEPARTYTARGETx "

            if text=="":
                text+=" feature_target_in_tweet_xNOTHINGx "

            feature.append(text)


        if train_tweets!=None:

            feature_train  = []
            for tweet in train_tweets:
                text=""
                text_tweet=re.sub(r'[^\w\s]',' ',tweet.text_hashtag_and_screen_name_slitted)
                text_tweet=text_tweet.split(" ")
                for t in target_train:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xTARGETx "

                for t in target_pronoums_train:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xTARGETx "

                for t in target_party_train:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xTARGETx "

                for t in target_opposite_in_party_train:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xOPPOSITEPARTYTARGETx "

                for t in target_opposite_out_party_train:
                    if t.lower() in text_tweet:
                        text+=" feature_target_in_tweet_xOPPOSITEPARTYTARGETx "

                if text=="":
                    text+=" feature_target_in_tweet_xNOTHINGx "

                feature_train.append(text)

            tfidfVectorizer = tfidfVectorizer.fit(feature_train)
            X_train = tfidfVectorizer.transform(feature_train)
            X_test = tfidfVectorizer.transform(feature)
            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

        else:
            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()


            return X, feature_names

    def get_target_in_tweet_labeled_features(self,tweets, train_tweets=None):

        opiniontowards  = []

        for tweet in tweets:
            opiniontowards.append(tweet.opiniontowards)

        if train_tweets!=None:
            opiniontowards_train  = []
            for tweet in train_tweets:
                opiniontowards_train.append(tweet.opiniontowards)

            return csr_matrix(np.vstack(opiniontowards_train)),csr_matrix(np.vstack(opiniontowards)),["feature_target_in_tweet_labeled"]

        else:


            return csr_matrix(np.vstack(opiniontowards)),["feature_target_in_tweet_labeled"]

    def get_target_position_features(self,tweets, train_tweets=None,
                                     target=None,target_pronoums=None,target_party=None,target_opposite_in_party=None,target_opposite_out_party=None,
                                     target_train=None,target_pronoums_train=None,target_party_train=None,target_opposite_in_party_train=None,target_opposite_out_party_train=None):

        target_position  = []

        for tweet in tweets:
            position_target=-1
            for t in target:
                if t.lower() in tweet.lemmas:
                    position_target=tweet.lemmas.index(t.lower())/(len(tweet.lemmas)-1)
                    break

            position_pronouns=-1
            for t in target_pronoums:
                if t.lower() in tweet.lemmas:
                    position_pronouns=tweet.lemmas.index(t.lower())/(len(tweet.lemmas)-1)
                    break

            position_party=-1
            for t in target_party:
                if t.lower() in tweet.lemmas:
                    position_party=tweet.lemmas.index(t.lower())/(len(tweet.lemmas)-1)
                    break

            position_opposite_in_the_party=-1
            for t in target_opposite_in_party:
                if t.lower() in tweet.lemmas:
                    position_opposite_in_the_party=tweet.lemmas.index(t.lower())/(len(tweet.lemmas)-1)
                    break
            position_opposite_out_the_party=-1
            for t in target_opposite_out_party:
                if t.lower() in tweet.lemmas:
                    position_opposite_out_the_party=tweet.lemmas.index(t.lower())/(len(tweet.lemmas)-1)
                    break

            target_position.append([position_target,position_pronouns,position_party,position_opposite_in_the_party,position_opposite_out_the_party])

        if train_tweets!=None:
            target_position_train  = []
            for tweet in train_tweets:
                position_target=-1
                for t in target_train:
                    if t.lower() in tweet.lemmas:
                        position_target=tweet.lemmas.index(t.lower())/(len(tweet.lemmas)-1)
                        break

                position_pronouns=-1
                for t in target_pronoums_train:
                    if t.lower() in tweet.lemmas:
                        position_pronouns=tweet.lemmas.index(t.lower())/(len(tweet.lemmas)-1)
                        break

                position_party=-1
                for t in target_party_train:
                    if t.lower() in tweet.lemmas:
                        position_party=tweet.lemmas.index(t.lower())/(len(tweet.lemmas)-1)
                        break

                position_opposite_in_the_party=-1
                for t in target_opposite_in_party_train:
                    if t.lower() in tweet.lemmas:
                        position_opposite_in_the_party=tweet.lemmas.index(t.lower())/(len(tweet.lemmas)-1)
                        break
                position_opposite_out_the_party=-1
                for t in target_opposite_out_party_train:
                    if t.lower() in tweet.lemmas:
                        position_opposite_out_the_party=tweet.lemmas.index(t.lower())/(len(tweet.lemmas)-1)
                        break

                target_position_train.append([position_target,position_pronouns,position_party,position_opposite_in_the_party,position_opposite_out_the_party])


            return csr_matrix(np.vstack(target_position_train)), csr_matrix(np.vstack(target_position)), ["feature_position_target","feature_position_pronouns","feature_position_party","feature_position_opposite_in_the_party","feature_position_opposite_out_the_party"]

        else:


            return csr_matrix(np.vstack(target_position)), ["feature_position_target","feature_position_pronouns","feature_position_party","feature_position_opposite_in_the_party","feature_position_opposite_out_the_party"]

    def get_presence_of_not_features(self,tweets, train_tweets=None):

        not_presence  = []

        for tweet in tweets:
            not_flag=0
            for liwc_func in tweet.liwc_function:
                if "negate" in liwc_func:
                    not_flag=1
            not_presence.append(not_flag)

        if train_tweets!=None:
            not_presence_train=[]
            for tweet in train_tweets:
                not_flag=0
                for liwc_func in tweet.liwc_function:
                    if "negate" in liwc_func:
                        not_flag=1

                not_presence_train.append(not_flag)

            return csr_matrix(np.vstack(not_presence_train)),csr_matrix(np.vstack(not_presence)),["feature_not_presence"]

        else:


            return csr_matrix(np.vstack(not_presence)),["feature_not_presence"]

    def get_target_and_no_in_hashtag_features(self, tweets, train_tweets=None,target=None,target_opposite=None,target_train=None,target_opposite_train=None):


        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          #stop_words="english",
                                          lowercase=False, #true 0.507 false 0.51
                                          binary=True,
                                          max_features=500000)

        feature  = []
        for tweet in tweets:
            text=""
            for tag in re.findall(r"#(\w+)", tweet.text.lower()):
                if "no" in tag or "against" in tag or "stop" in  tag or "anti" in tag:
                    for t in target:
                        if t.lower() in tag:

                            text+=" feature_target_and_no_in_hashtag_xTARGETx "

                    for t in target_opposite:
                        if t.lower() in tag:
                            text+=" feature_target_and_no_in_hashtag_xOPPOSITETARGETx "
            feature.append(text)




        if train_tweets!=None:

            feature_train  = []
            for tweet in train_tweets:
                text=""
                for tag in re.findall(r"#(\w+)", tweet.text.lower()):

                    if "no" in tag:
                        for t in target_train:
                            if t.lower() in tag:

                                text+=" feature_target_and_no_in_hashtag_xTARGETx "

                        for t in target_opposite_train:
                            if t.lower() in tag:
                                text+=" feature_target_and_no_in_hashtag_xOPPOSITETARGETx "
                feature_train.append(text)


            tfidfVectorizer = tfidfVectorizer.fit(feature_train)
            X_train = tfidfVectorizer.transform(feature_train)
            X_test = tfidfVectorizer.transform(feature)
            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

        else:
            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()


            return X, feature_names

    def get_liwc_features(self,tweets,train_tweets=None):


        tfidfVectorizer = TfidfVectorizer(ngram_range=(1,1),
                                          binary=False,
                                          max_features=500000)


        feature  = []

        for tweet in tweets:
            feature.append(tweet.liwc_text)

        if train_tweets!=None:

            feature_train  = []

            for tweet in train_tweets:
                feature_train.append(tweet.liwc_text)



            tfidfVectorizer = tfidfVectorizer.fit(feature_train)
            X_test = tfidfVectorizer.transform(feature)
            X_train = tfidfVectorizer.transform(feature_train)

            feature_names=tfidfVectorizer.get_feature_names()

            return X_train,X_test,feature_names


        else:

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()

            return X,feature_names

    def get_BoTargetPOS_features(self,tweets, train_tweets=None,
                                     target=None,target_pronoums=None,target_party=None,target_opposite_in_party=None,target_opposite_out_party=None,
                                     target_train=None,target_pronoums_train=None,target_party_train=None,target_opposite_in_party_train=None,target_opposite_out_party_train=None):

        tfidfVectorizer = CountVectorizer(ngram_range=(3,5),
                                          #stop_words="english",
                                          lowercase=True,
                                          binary=False,
                                          max_features=500000)

        feature  = []
        for tweet in tweets:
            text=""
            for i in range(0,len(tweet.pos)):
                if len([t in tweet.lemmas[i] for t in target])>0:
                    text+=" feature_BoTargetNEgatePOS_xTARGETx "
                elif len([target in tweet.lemmas[i] for target in target_pronoums])>0:
                    text+=" feature_BoTargetNEgatePOS_xpronoumsTARGETx "
                elif len([target in tweet.lemmas[i] for target in target_party])>0:
                    text+=" feature_BoTargetNEgatePOS_xPARTYTARGETx "
                elif len([target in tweet.lemmas[i] for target in target_opposite_in_party])>0:
                    text+=" feature_BoTargetNEgatePOS_xOPPOSITEINPARTYTARGETx "
                elif len([target in tweet.lemmas[i] for target in target_opposite_out_party])>0:
                    text+=" feature_BoTargetNEgatePOS_xOPPOSITEOUTPARTYTARGETx "
                elif tweet.pos[i] not in ["$","''",".",":",",","-RRB-","-LRB-","``","\""]:
                    text+=" feature_BoTargetNEgatePOS_x"+tweet.pos[i][0:2]+" "

            feature.append(text)


        if train_tweets!=None:

            feature_train  = []
            for tweet in train_tweets:
                text=""
                for i in range(0,len(tweet.pos)):
                    if len([target in tweet.lemmas[i] for target in target_train])>0:
                        text+=" feature_BoTargetNEgatePOS_xTARGETx "
                    elif len([target in tweet.lemmas[i] for target in target_pronoums_train])>0:
                        text+=" feature_BoTargetNEgatePOS_xpronoumsTARGETx "
                    elif len([target in tweet.lemmas[i] for target in target_party_train])>0:
                        text+=" feature_BoTargetNEgatePOS_xPARTYTARGETx "
                    elif len([target in tweet.lemmas[i] for target in target_opposite_in_party_train])>0:
                        text+=" feature_BoTargetNEgatePOS_xOPPOSITEINPARTYTARGETx "
                    elif len([target in tweet.lemmas[i] for target in target_opposite_out_party_train])>0:
                        text+=" feature_BoTargetNEgatePOS_xOPPOSITEOUTPARTYTARGETx "
                    elif tweet.pos[i] not in ["$","''",".",":",",","-RRB-","-LRB-","``","\""]:
                        text+=" feature_BoTargetNEgatePOS_x"+tweet.pos[i][0:2]+" "

                feature_train.append(text)

            tfidfVectorizer = tfidfVectorizer.fit(feature_train)
            X_train = tfidfVectorizer.transform(feature_train)
            X_test = tfidfVectorizer.transform(feature)
            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

        else:
            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()


            return X, feature_names

    def get_BoTargetNegatePOS_features(self, tweets, train_tweets=None,
                                     target=None,target_pronoums=None,target_party=None,target_opposite_in_party=None,target_opposite_out_party=None,
                                     target_train=None,target_pronoums_train=None,target_party_train=None,target_opposite_in_party_train=None,target_opposite_out_party_train=None):

        tfidfVectorizer = CountVectorizer(ngram_range=(3,5),
                                          #stop_words="english",
                                          lowercase=True,
                                          binary=False,
                                          max_features=500000)

        feature  = []
        for tweet in tweets:
            text=""
            for i in range(0,len(tweet.pos)):
                if len([t in tweet.lemmas[i] for t in target])>0:
                    text+=" feature_BoTargetNEgatePOS_xTARGETx "
                elif len([target in tweet.lemmas[i] for target in target_pronoums])>0:
                    text+=" feature_BoTargetNEgatePOS_xpronoumsTARGETx "
                elif len([target in tweet.lemmas[i] for target in target_party])>0:
                    text+=" feature_BoTargetNEgatePOS_xPARTYTARGETx "
                elif len([target in tweet.lemmas[i] for target in target_opposite_in_party])>0:
                    text+=" feature_BoTargetNEgatePOS_xOPPOSITEINPARTYTARGETx "
                elif len([target in tweet.lemmas[i] for target in target_opposite_out_party])>0:
                    text+=" feature_BoTargetNEgatePOS_xOPPOSITEOUTPARTYTARGETx "
                elif "negate" in tweet.liwc_function[i]:
                    text+=" feature_BoTargetNEgatePOS_xNEGATEx "
                elif  tweet.token_hl[i]==1:
                    text+=" feature_BoTargetNEgatePOS_xPOSx "
                elif  tweet.token_hl[i]==-1:
                    text+=" feature_BoTargetNEgatePOS_xNEGx "
                elif tweet.pos[i] not in ["$","''",".",":",",","-RRB-","-LRB-","``","\""]:
                    text+=" feature_BoTargetNEgatePOS_x"+tweet.pos[i][0:2]+" "

            feature.append(text)


        if train_tweets!=None:

            feature_train  = []
            for tweet in train_tweets:
                text=""
                for i in range(0,len(tweet.pos)):
                    if len([target in tweet.lemmas[i] for target in target_train])>0:
                        text+=" feature_BoTargetNEgatePOS_xTARGETx "
                    elif len([target in tweet.lemmas[i] for target in target_pronoums_train])>0:
                        text+=" feature_BoTargetNEgatePOS_xpronoumsTARGETx "
                    elif len([target in tweet.lemmas[i] for target in target_party_train])>0:
                        text+=" feature_BoTargetNEgatePOS_xPARTYTARGETx "
                    elif len([target in tweet.lemmas[i] for target in target_opposite_in_party_train])>0:
                        text+=" feature_BoTargetNEgatePOS_xOPPOSITEINPARTYTARGETx "
                    elif len([target in tweet.lemmas[i] for target in target_opposite_out_party_train])>0:
                        text+=" feature_BoTargetNEgatePOS_xOPPOSITEOUTPARTYTARGETx "
                    elif "negate" in tweet.liwc_function[i]:
                        text+=" feature_BoTargetNEgatePOS_xNEGATEx "
                    elif  tweet.token_hl[i]==1:
                        text+=" feature_BoTargetNEgatePOS_xPOSx "
                    elif  tweet.token_hl[i]==-1:
                        text+=" feature_BoTargetNEgatePOS_xNEGx "
                    elif tweet.pos[i] not in ["$","''",".",":",",","-RRB-","-LRB-","``","\""]:
                        text+=" feature_BoTargetNEgatePOS_x"+tweet.pos[i][0:2]+" "

                feature_train.append(text)


            tfidfVectorizer = tfidfVectorizer.fit(feature_train)
            X_train = tfidfVectorizer.transform(feature_train)
            X_test = tfidfVectorizer.transform(feature)
            feature_names=tfidfVectorizer.get_feature_names()
            return X_train, X_test, feature_names

        else:
            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()


            return X, feature_names

    def get_conditional_features(self,tweets, train_tweets=None):

        feature  = []
        """
            VB  Verb, base form
            VBD Verb, past tense
            VBG Verb, gerund or present participle
            VBN Verb, past participle
            VBP Verb, non-3rd person singular present
            VBZ Verb, 3rd person singular present

        """

        for tweet in tweets:
            tipo0=0
            tipo1=0
            tipo2_3=0
            for i_first_clause in range(0,len(tweet.lemmas)):
                if tweet.lemmas[i_first_clause]=="if":
                    tipo0=1
                    for i_second_clause in range(i_first_clause,len(tweet.lemmas)):
                        if tweet.lemmas[i_second_clause] in ["will"]:
                            tipo1=1
                            tipo0=0
                            break
                        if tweet.lemmas[i_second_clause] in ["would"]:
                            tipo2_3=1
                            tipo0=0
                            break

            feature.append([tipo0,tipo1,tipo2_3])

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                tipo0=0
                tipo1=0
                tipo2_3=0
                for i_first_clause in range(0,len(tweet.lemmas)):
                    if tweet.lemmas[i_first_clause]=="if":
                        tipo0=1
                        for i_second_clause in range(i_first_clause,len(tweet.lemmas)):
                            if tweet.lemmas[i_second_clause] in ["will"]:
                                tipo1=1
                                tipo0=0
                                break
                            if tweet.lemmas[i_second_clause] in ["would"]:
                                tipo2_3=1
                                tipo0=0
                                break




                feature_train.append([tipo0,tipo1,tipo2_3])

            return csr_matrix(np.vstack(feature_train)),csr_matrix(np.vstack(feature)),["feature_conditional_0","feature_conditional_1","feature_conditional_2_3"]

        else:


            return csr_matrix(np.vstack(feature)),["feature_conditional_0","feature_conditional_1","feature_conditional_2_3"]

    def get_puntuaction_marks_features(self,tweets, train_tweets=None):

        feature  = []

        for tweet in tweets:
            feature.append([
                len(re.findall(r"[!]", tweet.text)),
                len(re.findall(r"[?]", tweet.text)),
                len(re.findall(r"[.]", tweet.text)),
                len(re.findall(r"[,]", tweet.text)),
                len(re.findall(r"[;]", tweet.text)),
                len(re.findall(r"[!?.,;]", tweet.text)),
                ]

            )

        if train_tweets!=None:
            feature_train  = []
            for tweet in train_tweets:
                feature_train.append([
                len(re.findall(r"[!]", tweet.text)),
                len(re.findall(r"[?]", tweet.text)),
                len(re.findall(r"[.]", tweet.text)),
                len(re.findall(r"[,]", tweet.text)),
                len(re.findall(r"[;]", tweet.text)),
                len(re.findall(r"[!?.,;]", tweet.text)),
                ])

            return csr_matrix(np.vstack(feature_train)),csr_matrix(np.vstack(feature)),[
                "feature_punctuation_mks_!",
                "feature_punctuation_mks_?",
                "feature_punctuation_mks_.",
                "feature_punctuation_mks_,",
                "feature_punctuation_mks_all",
                                                                                        ]

        else:


            return csr_matrix(np.vstack(feature)),["feature_numhashtag"]


#inizializer
def make_feature_manager():

    features_manager = Features_manager()

    return features_manager

