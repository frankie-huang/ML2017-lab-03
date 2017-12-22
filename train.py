#encoding=utf-8
from feature import *
from ensemble import *
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    data=[]
    for i in range(500):
        x=str(i)
        if i<10:
            x='00'+x
        elif i<100:
            x='0'+x
        im1 = Image.open('datasets/original/face/face_'+x+'.jpg').resize((24,24)).convert('L')
        im2 = Image.open('datasets/original/nonface/nonface_'+x+'.jpg').resize((24,24)).convert('L')        
        data.append(NPDFeature(np.array(im1)).extract())
        data.append(NPDFeature(np.array(im2)).extract())        
    label=[]
    # #缓存特征
    # AdaBoostClassifier.save(data,'data')
    # #读取缓存的特征
    # data=np.array(AdaBoostClassifier.load('data'))
    for i in range(1000):
        if i%2==0:
            label.append(1)
        else:
            label.append(-1)
    X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.2,random_state=100)
    classifier=AdaBoostClassifier(DecisionTreeClassifier,20)
    classifier.fit(X_train,y_train)
    res_train=classifier.predict(X_train)
    res_test=classifier.predict(X_test)
    print(classification_report(y_train,res_train))
    print(classification_report(y_test,res_test))
