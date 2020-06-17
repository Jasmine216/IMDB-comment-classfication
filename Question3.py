# -*-coding:utf-8-*-

import matplotlib
import pandas as pd
import nltk
import random
import operator
import sklearn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import linear_model
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import io
import string
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import  CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.preprocessing import StandardScaler

print("Start Preprocess Data")
###Preprocessing

train_neg_url = "datasets_coursework1/IMDb/train/imdb_train_neg.txt"
train_pos_url = "datasets_coursework1/IMDb/train/imdb_train_pos.txt"
dev_neg_url = "datasets_coursework1/IMDb/dev/imdb_dev_neg.txt"
dev_pos_url = "datasets_coursework1/IMDb/dev/imdb_dev_pos.txt"
test_neg_url = "datasets_coursework1/IMDb/test/imdb_test_neg.txt"
test_pos_url = "datasets_coursework1/IMDb/test/imdb_test_pos.txt"
print("Start Preprocess Data2")

def split_data(url):
    File = io.open(url, 'r',encoding='utf-8')
    strFile = File.read()
    File.close()
    #print(strFile)
    dataset_file = strFile.split("\n")

    return dataset_file


neg_reponse = split_data(train_neg_url)
pos_reponse = split_data(train_pos_url)
neg_dev = split_data(dev_neg_url)
pos_dev = split_data(dev_pos_url)
neg_test = split_data(test_neg_url)
pos_test = split_data(test_pos_url)

trainning_set = []
testing_set = []
development_set=[]

for pos_review in pos_reponse:
    trainning_set.append((pos_review,1))
for neg_review in neg_reponse:
    trainning_set.append((neg_review,0))
for pos_review in pos_dev:
    development_set.append((pos_review,1))
for neg_review in neg_dev:
    development_set.append((neg_review,0))
for i in neg_test:
    testing_set.append((i,0))
for i in pos_test:
    testing_set.append((i,1))

random.shuffle(trainning_set)
random.shuffle(development_set)
random.shuffle(testing_set)

X_train=[]
Y_train=[]
X_dev=[]
Y_dev=[]
X_test=[]
Y_test=[]
for instance in trainning_set:
    X_train.append(instance[0])
    Y_train.append(instance[1])
for instance in development_set:
    X_dev.append(instance[0])
    Y_dev.append(instance[1])
for instance in testing_set:
    X_test.append(instance[0])
    Y_test.append(instance[1])




print("Start Preprocess Data3")
# Function for training our svm classifier
def train_classifier(vectorizer,strVec,model,modelName,X_Train,Y_Train,X_Dev,Y_Dev,X_Test,Y_Test,num):

    time0 = time()
    Xtrain = vectorizer.fit_transform(X_Train)
    XtrainVec = Xtrain.toarray()
    Xtest = vectorizer.transform(X_Test)
    XtestVec = Xtest.toarray()
    Xdev = vectorizer.transform(X_Dev)
    XdevVec = Xdev.toarray()

    #print("Feature selection",strVec,num)
    fs_sentanalysis = SelectKBest(chi2, k=num).fit(XdevVec, Y_Dev) #Feature selection
    X_train_new = fs_sentanalysis.transform(XtrainVec)
    X_test_new = fs_sentanalysis.transform(XtestVec)
    print("Start Standardscaler")
    scaler = StandardScaler() #Standstard Scalar
    X_train_scaled = scaler.fit_transform(X_train_new)
    X_test_scaled = scaler.fit_transform(X_test_new)
    #print("Start training!",modelName)
    votion_model=model.fit(X_train_scaled, Y_Train) #Train the model
    #print("start predict!",votion_model)


    Y_test_predictions = votion_model.predict(X_test_scaled) #predict results
    print(strVec,modelName,num)
    accuracy_fold=accuracy_score(Y_Test, Y_test_predictions.round())
    #print ((round(accuracy_fold,3)))
    precision_fold = precision_score(Y_Test, Y_test_predictions.round(), average='macro')
    recall_fold = recall_score(Y_Test, Y_test_predictions.round(), average='macro')
    f1_fold = f1_score(Y_Test, Y_test_predictions.round(), average='macro')
    print(accuracy_fold,precision_fold,recall_fold,f1_fold)
    timeUsing=time()-time0
    print("time is :",timeUsing)  # calculate how much time they spend respectively

def train_classifier1(vectorizer,strVec,model,modelName,X_Train,Y_Train,X_Dev,Y_Dev,X_Test,Y_Test,num):

    time0 = time()
    Xtrain = vectorizer.fit_transform(X_Train)
    XtrainVec = Xtrain.toarray()
    Xtest = vectorizer.transform(X_Test)
    XtestVec = Xtest.toarray()
    Xdev = vectorizer.transform(X_Dev)
    XdevVec = Xdev.toarray()
    #print("Feature selection",strVec,num)
    fs_sentanalysis = SelectKBest(chi2, k=num).fit(XdevVec, Y_Dev)  #Feature selection
    X_train_new = fs_sentanalysis.transform(Xtrain)
    X_test_new = fs_sentanalysis.transform(Xtest)
    #print("Start training!",modelName)
    votion_model=model.fit(Xtrain, Y_Train)  #Train the model
    #print("start predict!",votion_model)


    Y_test_predictions = votion_model.predict(Xtest)  #Predict results
    print(strVec,modelName,num)
    accuracy_fold=accuracy_score(Y_Test, Y_test_predictions.round())
    #print ((round(accuracy_fold,3)))
    precision_fold = precision_score(Y_Test, Y_test_predictions.round(), average='macro')
    recall_fold = recall_score(Y_Test, Y_test_predictions.round(), average='macro')
    f1_fold = f1_score(Y_Test, Y_test_predictions.round(), average='macro')
    print(accuracy_fold,precision_fold,recall_fold,f1_fold)
    timeUsing=time()-time0
    print("time is :",timeUsing)  # calculate how much time they spend respectively


#print("Start Choice feature")
###Feature Selection

##ngram
vectorizerUnigram = CountVectorizer(max_features=5000,ngram_range=(1,1),min_df=1,stop_words='english')
##bigram
vectorizerBigram = CountVectorizer(max_features=5000,ngram_range=(2, 2), min_df=1,stop_words='english')
##ngram+bigram
vectorizerTwogram = CountVectorizer(max_features=5000,ngram_range=(1, 2), min_df=1,stop_words='english')
##ngram+bigram+Trigram
vectorizerTrigram = CountVectorizer(max_features=5000,ngram_range=(1, 3), min_df=1,stop_words='english')
##TF-IDF
vectorizerTF = TfidfVectorizer(max_features=5000,min_df=1,stop_words='english')

Vectorizer={"Unigram":vectorizerUnigram,"Bigram":vectorizerBigram,"Uni~Big":vectorizerTwogram,"Uni~Trigram":vectorizerTrigram,"TF":vectorizerTF}

##Model Fit
Kernel = ['linear', 'rbf', 'sigmoid']




def Development1():  #select model and the number of features
    for key,value in Vectorizer.items():
        #for modelPara in Kernel:
            #model= sklearn.svm.SVC(kernel=modelPara, gamma='auto')
            #train_classifier(value,key, model,"SVC" ,X_train, Y_train, X_dev, Y_dev, X_test, Y_test, 500)
        #model=sklearn.linear_model.LinearRegression()
        #train_classifier(value,key, model,"LinearRegression" ,X_train, Y_train, X_dev, Y_dev, X_test, Y_test, 500)
        #model = RandomForestClassifier(random_state=0)
        #train_classifier(value,key,model,"RandomForestClassifier", X_train, Y_train, X_dev, Y_dev, X_test, Y_test, 500)
        votion_model = VotingClassifier(estimators=[
            ('SVCLinear',sklearn.svm.SVC(kernel="linear", gamma='auto' )),
            ('SVCRbf',sklearn.svm.SVC(kernel="rbf", gamma='auto')),
            #('LinearRegression', sklearn.linear_model.LinearRegression()),
            ('RandomForest',RandomForestClassifier(random_state=0))
            ])
        train_classifier1(value, key, votion_model, "VotingClassifier", X_train, Y_train, X_dev, Y_dev, X_test, Y_test,
            500)

VectorizerDev={"Unigram":vectorizerUnigram,"Uni~Big":vectorizerTwogram,"Uni~Trigram":vectorizerTrigram,"TF":vectorizerTF}


def Development2(): #test difference of using or no using standardScaler()
    print("Start Development")
    model = sklearn.svm.SVC(kernel='linear', gamma='auto')
    train_classifier(vectorizerTF, "TF", model, "SVC", X_train, Y_train, X_dev, Y_dev, X_test, Y_test, 2000)
    train_classifier1(vectorizerTF, "TF", model, "SVC", X_train, Y_train, X_dev, Y_dev, X_test, Y_test, 2000)



def Development3():
    time0 = time()
    Xtrain = vectorizerUnigram.fit_transform(X_train)
    XtrainVec = Xtrain.toarray()
    Xtest = vectorizerUnigram.transform(X_test)
    XtestVec = Xtest.toarray()
    Xdev = vectorizerUnigram.transform(X_dev)
    XdevVec = Xdev.toarray()
    print("Start append vector")
    Xtrain = vectorizerTF.fit_transform(X_train)
    XtrainVec = np.hstack((XtrainVec,Xtrain.toarray()))
    Xtest = vectorizerTF.transform(X_test)
    XtestVec = np.hstack((XtestVec,Xtest.toarray()))
    Xdev = vectorizerTF.transform(X_dev)
    XdevVec = np.hstack((XdevVec,Xdev.toarray()))
    print("Start append vector2")
    Xtrain = vectorizerTwogram.fit_transform(X_train)
    XtrainVec = np.hstack((XtrainVec, Xtrain.toarray()))
    Xtest = vectorizerTwogram.transform(X_test)
    XtestVec = np.hstack((XtestVec, Xtest.toarray()))
    Xdev = vectorizerTwogram.transform(X_dev)
    XdevVec = np.hstack((XdevVec, Xdev.toarray()))

    fs_sentanalysis = SelectKBest(chi2, k=2000).fit(XdevVec, Y_dev)  # Feature selection
    X_train_new = fs_sentanalysis.transform(XtrainVec)
    X_test_new = fs_sentanalysis.transform(XtestVec)
    print("Start training!")
    model = sklearn.svm.SVC(kernel='linear', gamma='auto')
    votion_model = model.fit(X_train_new, Y_train)  # Train the model
    print("start predict!")

    Y_test_predictions = votion_model.predict(X_test_new)  # Predict results
    accuracy_fold = accuracy_score(Y_test, Y_test_predictions.round())
    # print ((round(accuracy_fold,3)))
    precision_fold = precision_score(Y_test, Y_test_predictions.round(), average='macro')
    recall_fold = recall_score(Y_test, Y_test_predictions.round(), average='macro')
    f1_fold = f1_score(Y_test, Y_test_predictions.round(), average='macro')
    print(accuracy_fold, precision_fold, recall_fold, f1_fold)
    timeUsing = time() - time0
    print("time is :", timeUsing)  # calculate how much time they spend respectively

model = sklearn.svm.SVC(kernel='linear', gamma='auto')
train_classifier(vectorizerTF, "TF", model, "SVC", X_train, Y_train, X_dev, Y_dev, X_test, Y_test, 2000)


#Development1()



