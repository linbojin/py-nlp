# -*- coding: UTF-8 -*-

from __future__ import print_function
import numpy as np

# 《统计学习方法》 p177

class HMM(object):
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def likelihood_by_forward_algo(self, O):
        T = O.size
        alhpa = self.pi * self.B[:, O[0]]

        for i in range(1, T):
            alhpa = np.matmul(alhpa, self.A) * self.B[:, O[i]]

        return np.sum(alhpa)

    def decoding_by_viterbi_algo(self, O):
        T = O.size
        v = self.pi * self.B[:, O[0]]
        backpointer = []
        for i in range(1, T):
            v_tmp = np.reshape(v, [-1, 1]) * self.A * self.B[:, O[i]]
            v = np.max(v_tmp, axis=0)
            b = list(np.argmax(v_tmp, axis=0))
            backpointer.append(b)

        best_path_prob = np.max(v)
        best_path_pointer = np.argmax(v)

        k = best_path_pointer
        best_path = [k+1]

        for i in range(len(backpointer)-1,-1,-1):
            k = backpointer[i][k]
            best_path.insert(0, k+1)

        return best_path_prob, best_path


A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7,0.3]])
pi = np.array([0.2,0.4,0.4])
T = 3
O = np.array([0,1,0])

hmm = HMM(A, B, pi)
print(hmm.likelihood_by_forward_algo(np.array([0,1,0])))
# 0.130218
print(hmm.decoding_by_viterbi_algo(np.array([0,1,0])))
# (0.014699999999999998, [3, 3, 3])
