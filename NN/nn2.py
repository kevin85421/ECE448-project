from sklearn import model_selection, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

dfTrue = pd.read_csv("../data/True.csv", sep=',',  encoding='utf8')
dfFake = pd.read_csv("../data/Fake.csv", sep=',',  encoding='utf8')
dfTrue['label'] = 1
dfFake['label'] = 0
df = pd.concat([dfFake, dfTrue], ignore_index=True)

# Remove stop_words 
stop = set(stopwords.words('english'))
df['cleanText'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

# Model training
def modelTraining(model, trainX, trainY, testX, testY, fileName=None):
    """ 
        trainX = vectorized data
        trainY = labels of 'vectorized data'
        testY = test data 
    """
    # cross-validation
    index = ['True News', 'Fake News']
    classifier = model.fit(trainX, trainY)
    predictions = classifier.predict(testX)

    print( metrics.classification_report(testY, predictions, target_names=['True News', 'Fake News']))
    print('Training set score: {:.4f}'.format(classifier.score(trainX, trainY)))
    print('Test set score: {:.4f}'.format(classifier.score(testX, testY)))
    return metrics.accuracy_score(predictions, testY) * 100

# Counter Vector
countVec = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
countVec.fit(df['cleanText'].apply(lambda x: np.str_(x)))
X = countVec.transform(df['cleanText'].apply(lambda x: np.str_(x)))

# train and test split
trainXCount, testXCount, trainY, testY = model_selection.train_test_split(X, df['label'], test_size=0.3, random_state=4, shuffle=True)

# Model1
print("Model1: NN")
accuracy = modelTraining(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), trainXCount, trainY, testXCount, testY) 
print("accuracy: ", accuracy)

# Model2
print("Model2: Naive Bayes")
accuracy = modelTraining(MultinomialNB(), trainXCount, trainY, testXCount, testY) 
print("accuracy: ", accuracy)

# Model3
print("Model3: Logistic Regression")
accuracy = modelTraining(LogisticRegression(random_state=0), trainXCount, trainY, testXCount, testY) 
print("accuracy: ", accuracy)