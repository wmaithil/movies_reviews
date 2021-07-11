import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

def extract_features(word_list):
    return dict([(word, True) for word in word_list])

# Load positive and negative reviews
positive_fileids = movie_reviews.fileids('pos')
negative_fileids = movie_reviews.fileids('neg')
features_positive = [(extract_features(movie_reviews.words(fileids=[f])),'Positive') for f in positive_fileids]
features_negative = [(extract_features(movie_reviews.words(fileids=[f])),'Negative') for f in negative_fileids]
 # Split the data into train and test (80/20)

threshold_factor = 0.8

threshold_positive = int(threshold_factor * len(features_positive))
threshold_negative = int(threshold_factor * len(features_negative))
features_train = features_positive[:threshold_positive]+features_negative[:threshold_negative]
features_test = features_positive[threshold_positive:]+features_negative[threshold_negative:]

print("Number of training datapoints: ", len(features_train))
print("Number of test datapoints: ", len(features_test))

classifier = NaiveBayesClassifier.train(features_train)
print("Accuracy of the classifier: ", nltk.classify.util.accuracy(classifier,features_test))

print("Top ten most informative words: ")
for item in classifier.most_informative_features()[:10]:
    print(item[0])
#Top ten most informative words:


#Sample input reviews
input_reviews = ["Started off as the greatest series of all time, but had the worst endingof all time.",
"Exquisite. 'Big Little Lies' takes us to an incredible journey with its emotional and intriguing storyline.",
"I love Brooklyn 99 so much. It has the best crew ever!!",
"The Big Bang Theory and to me it's one of the best written sitcoms currently on network TV.",
"'Friends' is simply the best series ever aired. The acting is amazing.",
"SUITS is smart, sassy, clever, sophisticated, timely and immensely entertaining!",
"Cumberbatch is a fantastic choice for Sherlock Holmes-he is physically right (he fits the traditional reading of the character) and he is a damn good actor",
"What sounds like a typical agent hunting serial killer, surprises with great characters, surprising turning points and amazing cast."
"This is one of the most magical things I have ever had the fortune of viewing.",
"I don't recommend watching this at all!"]

print("Predictions: ")

for review in input_reviews:
    print("\nReview:", review)
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()


print("Predictions: ")

for review in input_reviews:
    print("\nReview:", review)
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()
    print("Predicted sentiment: ", pred_sentiment)
    print("Probability: ", round(probdist.prob(pred_sentiment), 2))



