__author__ = 'mirko'
import Features_manager
import Database_manager
from itertools import combinations
from sklearn.metrics.classification import precision_recall_fscore_support, accuracy_score
import config_features as cfg_feature
from sklearn.naive_bayes import GaussianNB
print("Task A (supervised framework)")

target=target_train="Hillary Clinton"

database_manager=Database_manager.make_database_manager()
feature_manager=Features_manager.make_feature_manager()

tweets_train=database_manager.return_train_tweets(filter=target)
tweets_test=database_manager.return_test_tweets(filter=target)

stance_train=feature_manager.get_stance(tweets_train)
stance_test=feature_manager.get_stance(tweets_test)

file = open("Task A.csv","w")

#feature_names=numpy.array(['numhashtag','nummention','target_in_tweet_labeled','sentiment_labeled','sentiment_liwc']) #[ 0.80495356  0.6744186   0.79558011] 0.739686082511

feature_names=cfg_feature.feature_list['feature_names']

stuff = range(0, len(feature_names) )
count=0
better_f_macro=0
better_f_macro_optimized=0
parameters=[]
parameters_optimized=[]
max=0
max_feature_set=[]

for L in range(1, len(stuff)+1):
    for subset in combinations(stuff, L):

        X_train,X_test,feature_name=feature_manager.create_feature_space(tweets_test,
                                                                         feature_names[list(subset)],
                                                                         tweets_train,
                                                                         cfg_feature.target_alias['alias_HillaryClinton'],
                                                                         cfg_feature.target_alias['alias_pronoums_HillaryClinton'],
                                                                         cfg_feature.target_alias['alias_party_HillaryClinton'],
                                                                         cfg_feature.target_alias['alias_opponents_in_the_party_HillaryClinton'],
                                                                         cfg_feature.target_alias['alias_opponents_out_the_party_HillaryClinton'])
        clf = GaussianNB()

        clf.fit(X_train.toarray(),stance_train)

        test_predict = clf.predict(X_test.toarray())

        prec, recall, f, support = precision_recall_fscore_support(
        stance_test,
        test_predict,
        beta=1)

        accuracy = accuracy_score(
        stance_test,
        test_predict
        )


        if((f[0]+f[1])/2)>max:
            max=((f[0]+f[1])/2)
            max_feature_set=feature_names[list(subset)]
        print(feature_names[list(subset)])
        print(f, ((f[0]+f[1])/2))
        file.write('"'+(' '.join(feature_names[list(subset)]))+'"'+'\t'+str((f[0]+f[1])/2)+'\t'+str(f[0])+'\t'+str(f[1])+'\t'+str(f[2])+"\n")
        print(max_feature_set,max)
print(max_feature_set,max)


"""

#Majority

count = Counter(stance_train)
print(count.most_common())
majority_class=count.most_common()[0][0]
test_predict = [majority_class]*len(stance_test)
prec, recall, f, support = precision_recall_fscore_support(
stance_test,
test_predict,
beta=1)
accuracy = accuracy_score(
stance_test,
test_predict
)

print(prec, recall, f, support,accuracy)
"""

