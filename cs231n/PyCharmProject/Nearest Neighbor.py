# 이 코드는 원문에는 없지만 아래 코드들을 실행하기 위해서 꼭 필요한 함수라서 찾아서 가져왔습니다.
# CIFAR10 데이터를 로드하는 함수입니다. 이 부분을 실행하셔야 아래 부분의 코드가 정상적으로 실행될 것입니다.
from __future__ import print_function

from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    # load single batch of cifar
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    # load all of cifar
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        # Nearest Neighbor classifier는 단순히 모든 training data를 저장함
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]

        # 출력 유형이 입력 유형과 일치하는 지 확인
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # 모든 테스트 행을 반복
        for i in range(num_test):  # 원문에는 xrange라 되있는데 Python 3.5에서 작동을 안하여 range로 변경
            # L1 거리(모든 차를 더함)를 사용하여 i번째 test image에 가장 가까운 training image 찾기
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)  # 거리가 가장 작은 인덱스 얻기
            Ypred[i] = self.ytr[min_index]  # 가장 가까운 예제의 라벨 예측

        return Ypred

Xtr, Ytr, Xte, Yte = load_CIFAR10('../data/cifar10/') # 제공되는 함수

# 모든 이미지를 1차원으로 평평하게 만듬(32x32x3 -> 3072)
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3) # Xtr_row : 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3) # Xte_row : 10000 x 3072

nn = NearestNeighbor() # Nearest Neighbor classifier 클래스를 생성(아래에 코드에 나옴)
nn.train(Xtr_rows, Ytr) # 트레이닝 이미지와 라벨을 통해서 classifier를 학습시킴
Yte_predict = nn.predict(Xte_rows) # test set의 라벨을 예측

# 올바르게 예측된(i.e. 레이블 일치) 이미지의 평균 갯수를 계산하여 분류 정확도로 표시
print('accuracy: %f' % (np.mean(Yte_predict == Yte)))