import pandas as pd
msg = pd.read_csv('document.csv', names=['message', 'label'])
print("Total Instances of Dataset: ", msg.shape[0])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

otp:Total Instances of Dataset:  18



X = msg.message
y = msg.labelnum
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
from sklearn.feature_extraction.text import CountVectorizer

count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)




df = pd.DataFrame(Xtrain_dm.toarray(),columns=count_v.get_feature_names_out())
print(df[0:5])

otp:
   about  am  amazing  an  awesome  beers  best  can  dance  deal  ...  the  \
0      0   0        0   1        1      0     0    0      0     0  ...    0   
1      0   0        0   0        0      0     0    0      0     0  ...    0   
2      0   0        0   0        0      0     0    0      0     0  ...    1   
3      0   1        0   0        0      0     0    0      0     0  ...    0   
4      0   0        1   1        0      0     0    0      0     0  ...    0   

   these  this  tired  to  very  view  what  with  work  
0      0     0      0   0     0     1     1     0     0  
1      0     0      0   0     0     0     1     0     0  
2      0     1      0   0     0     0     0     0     0  
3      0     1      1   0     0     0     0     0     0  
4      0     1      0   0     0     0     0     0     0  

[5 rows x 40 columns]




from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)
pred = clf.predict(Xtest_dm)



for doc, p in zip(Xtrain, pred):
    p = 'pos' if p == 1 else 'neg'
    print("%s -> %s" % (doc, p))

otp:
What an awesome view -> neg
What a great holiday -> pos
I do not like the taste of this juice -> pos
I am tired of this stuff -> pos
This is an amazing place -> pos




from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
print('Accuracy Metrics: \n')
print('Accuracy: ', accuracy_score(ytest, pred))
print('Recall: ', recall_score(ytest, pred))
print('Precision: ', precision_score(ytest, pred))
print('Confusion Matrix: \n', confusion_matrix(ytest, pred))

otp:
Accuracy Metrics: 

Accuracy:  0.4
Recall:  1.0
Precision:  0.25
Confusion Matrix: 
 [[1 3]
 [0 1]]
