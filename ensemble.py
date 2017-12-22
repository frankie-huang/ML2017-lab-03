#encoding=utf-8
import pickle
import numpy as np
from sklearn.metrics import classification_report 

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier=weak_classifier
        self.n_weakers_limit=n_weakers_limit

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self.alpha=[]
        self.classifier=[]
        w=np.ones(len(X))/len(X)#样本权重
        for i in range(self.n_weakers_limit):
            print('正在训练第',i,'个模型')
            if i==0:
                res=np.zeros(len(X))-1
            else:
                res=self.classifier[-1].predict(X)
            
            self.classifier.append(self.weak_classifier(max_depth=1).fit(X,y,w))
            
            #计算错误率
            err=0.0
            for j in range(len(X)):
                if self.classifier[i].predict(X[j].reshape(1,-1))!=y[j]:
                    err+=w[j]#在weight分布下的错误率
            if err>0.5:
                break
            #计算alpha
            self.alpha.append(1.0/2*np.log(1/err-1))

            #更新w
            for k in range(len(X)):
                w[k]=w[k]*np.exp(-y[k]*self.alpha[i]*self.classifier[i].predict(X[k].reshape(1,-1)))
            
            #归一化
            s=sum(w)#规范化因子
            w/=s
        

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        labels=[]
        for j in range(len(X)):
            res=0.0
            for i in range(len(self.alpha)):
                res+=self.alpha[i]*(self.classifier[i].predict(X[j].reshape(1,-1)))
            if res >= threshold :
                labels.append(1)
            else:
                labels.append(-1)
        return labels
        

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
