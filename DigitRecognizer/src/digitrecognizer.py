# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

class DigitRecognizer():
    def __init__(self):
        pass

    def reading(self):
        # サンプルデータを読み込む
        self.training = pd.read_csv('../input/train.csv')
        self.testing = pd.read_csv('../input/test.csv')

    def preprocessing(self):
        # 前処理
        self.training_datas = self.training.ix[:, 1:]
        self.training_labels = self.training.ix[:, 0]
        self.testing_datas = self.testing.ix[:, 0:]
        self.testing_labels = None

    def cross_validation(self):
        self.model = RandomForestClassifier()
        scores = cross_validation.cross_val_score(
            self.model, self.training_datas, self.training_labels, cv = 50)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def learning(self):
        # 学習
        self.model = RandomForestClassifier()
        self.model.fit(self.training_datas, self.training_labels)

    def predicting(self):
        # 評価
        self.answer = self.model.predict(self.testing_datas)

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
    dr.cross_validation()
    dr.learning()
    dr.predicting()
    dr.writing()

if __name__ == '__main__':
    main()
