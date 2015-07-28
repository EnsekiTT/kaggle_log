# -*- coding:utf-8 -*-
import pandas as pd

def dataread():
    sample = pd.read_csv('datasets/sample.csv')
    labels = sample['label']
    pixels = sample.ix[:, 1:]
    print pixels

def main():
    dataread()

if __name__ == '__main__':
    main()
