# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from keras.models import Sequential
from keras.layers.core import Dense, Activation

class DigitRecognizer():
    def __init__(self):
        pass

    def reading(self):
        # サンプルデータを読み込む
        self.training = pd.read_csv('../input/train.csv')
        self.testing = pd.read_csv('../input/test.csv')

    def preprocessing(self):
        # 前処理
        self.training_datas = self.training.ix[:, 1:].values.copy()
        self.training_labels = self.training.ix[:, 0].values.copy()
        self.testing_datas = self.testing.ix[:, 0:].values.copy()
        self.testing_labels = None

    def model_tuning(self):
        # RandomForestClassifier
        #self.model = RandomForestClassifier()

        # SupportVectorMachine
        #self.model = OneVsRestClassifier(LinearSVC(random_state=0))

        # Deep learning
        dims = self.training_datas.shape[1]
        print(dims, 'dims')
        self.model = Sequential()
        self.model.add(Dense(input_dim=dims, output_dim=64, init="glorot_uniform"))
        self.model.add(Activation("relu"))
        self.model.add(Dense(input_dim=64, output_dim=10, init="glorot_uniform"))
        self.model.add(Activation("softmax"))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

    def cross_validation(self):
        scores = cross_validation.cross_val_score(
            self.model, self.training_datas, self.training_labels, cv = 50)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def learning(self):
        # 学習
        # RandomForest & SVM
        # self.model.fit(self.training_datas, self.training_labels)

        # DeepLearning
        #training_datas = [self.training_datas.ix[i].values.tolist() for i in range(len(self.training_datas))]
        #training_labels = [self.training_labels.ix[i] for i in range(len(self.training_labels))]
        self.model.fit(self.training_datas, self.training_labels,
            verbose=1, nb_epoch=2, batch_size=1, validation_split=0.15)

    def predicting(self):
        # 評価
        # RandomForest & SVM
        #self.answer = self.model.predict(self.testing_datas)

        # DeepLearning
        #testing_datas = [self.testing_datas.ix[i].values.tolist() for i in range(len(self.testing_datas))]
        self.answer = model.predict_classes(self.testing_datas)

    def writing(self):
        # 書き出し
        f = open('../answer.csv', 'w')
        f.write('ImageId,Label\n')
        for i, ans in enumerate(self.answer):
            f.write('%d,%d\n' %(i+1,ans))
        f.close()

def main():
    dr = DigitRecognizer()
    dr.reading()
    dr.preprocessing()
    dr.model_tuning()
    # dr.cross_validation() # for sklearn
    dr.learning()
    dr.predicting()
    dr.writing()

if __name__ == '__main__':
    main()
