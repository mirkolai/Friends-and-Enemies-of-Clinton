__author__ = 'mirko'

import numpy

target_alias = {
    'alias_HillaryClinton' : ["HillaryClinton","hillary","Clinton","Hill"],
    'alias_pronoums_HillaryClinton' : ["she","her"],
    'alias_party_HillaryClinton' : ["DEM","democratic","democrat","democrats","progressive"],
    'alias_opponents_in_the_party_HillaryClinton' : ["Bernie","Sanders",
                                                     "Martin","O'Malley",
                                                     "Lincoln","Chafee",
                                                     #"Jim",
                                                     "Webb",
                                                     "Lawrence","Lessig"],

    'alias_opponents_out_the_party_HillaryClinton' : [
                                        "republican","republicans","conservative",
                                        "realDonaldTrump","Donald","Trump",
                                        "Ted","Cruz",
                                        "Marco","Rubio",
                                        "John","Kasich",
                                        "Ben","Carson",
                                        "Jeb","Bush",
                                        "Rand","Paul",
                                        "Mike","Huckabee",
                                        "Carly","Fiorina",
                                        "Chris","Christie",
                                        "Rick","Santorum",
                                        #"Jim",
                                        "Gilmore",
                                        "Rick","Perry",
                                        "Scott","Walker",
                                        "Bobby","Jindal",
                                        "Lindsey","Graham",
                                        "George","Pataki"
                                        ],

    'alias_target_DonaldTrump':["realDonaldTrump","Donald","Trump"],
    'alias_pronoums_DonaldTrump' : ["he","his"],
    'alias_party_DonaldTrump' : ["republican","republicans","conservative"],

    'alias_opponents_out_the_party_DonaldTrump': ["DEM","democratic","democrat","democrats","progressive",
                                                  "HillaryClinton","hillary","Clinton","Hill",
                                                  "Bernie","Sanders",
                                                  "Martin","O'Malley",
                                                  "Lincoln","Chafee",
                                                  #"Jim",
                                                  "Webb",
                                                  "Lawrence","Lessig"],

    'alias_opponents_in_the_party_DonaldTrump' :  [
                                        "Ted","Cruz",
                                        "Marco","Rubio",
                                        "John","Kasich",
                                        "Ben","Carson",
                                        "Jeb","Bush",
                                        "Rand","Paul",
                                        "Mike","Huckabee",
                                        "Carly","Fiorina",
                                        "Chris","Christie",
                                        "Rick","Santorum",
                                        #"Jim",
                                        "Gilmore",
                                        "Rick","Perry",
                                        "Scott","Walker",
                                        "Bobby","Jindal",
                                        "Lindsey","Graham",
                                        "George","Pataki"
                                        ]
    }

feature_list = { 'feature_names' : numpy.array([
                                 "BoW",
                                 "hashtag",
                                 "numhashtag",
                                 "BoWinHashtag",
                                 "mention",
                                 "nummention",
                                 "BoWinMention",
                                 "target_in_tweet_labeled",
                                 "target_in_tweet",
                                 "target_in_tweet_alternative",
                                 "sentiment_labeled",
                                 "sentiment_afinn",
                                 "sentiment_liwc",
                                 "sentiment_hl",
                                 "sentiment_dal",
                                 "target_and_no_in_hashtag",
                                 "liwc",
                                 "BoTargetPOS",
                                 "BoTargetNegatePOS",
                                 "target_position",
                                 "conditional",
                                 "punctuation_marks",


                                 ])
                }
