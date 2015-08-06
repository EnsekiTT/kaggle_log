# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten

class DigitRecognizer():
    def __init__(self):
        pass

    def reading(self):
        # サンプルデータを読み込む
        self.training = pd.read_csv('../input/train.csv')
        self.testing = pd.read_csv('../input/test.csv')

    def preprocessing(self):
        # 前処理
        self.nb_classes = 10

        t_datas = self.training.ix[:, 1:].values.copy()
        self.training_datas = t_datas.reshape(t_datas.shape[0], 1, 28, 28)

        t_labels = self.training.ix[:, 0].values.copy()
        self.training_labels = np_utils.to_categorical(t_labels, self.nb_classes)

        t_datas = self.testing.ix[:, 0:].values.copy()
        self.testing_datas = t_datas.reshape(t_datas.shape[0], 1, 28, 28)

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

        self.model.add(Convolution2D(32, 1, 3, 3, border_mode='full'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, 32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(poolsize=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(32*196, 128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(128, 32))
        self.model.add(Activation('relu'))

        self.model.add(Dense(32, self.nb_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    def cross_validation(self):
        scores = cross_validation.cross_val_score(
            self.model, self.training_datas, self.training_labels, cv = 50)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def learning(self):
        # 学習
        # RandomForest & SVM
        # self.model.fit(self.training_datas, self.training_labels)

        # DeepLearning
        self.model.fit(self.training_datas, self.training_labels, show_accuracy=True,
            verbose=1, nb_epoch=12, batch_size=64, validation_split=0.15)

    def predicting(self):
        # 評価
        # RandomForest & SVM
        #self.answer = self.model.predict(self.testing_datas)

        # DeepLearning
        self.answer = self.model.predict_classes(self.testing_datas)

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
