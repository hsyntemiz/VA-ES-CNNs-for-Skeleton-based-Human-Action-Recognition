import numpy as np


def softmax(data):
    e = np.exp(data - np.amax(data, axis=-1, keepdims=True))
    s = np.sum(e, axis=-1, keepdims=True)
    return e / s

def acc(va_steer_results, va_cnn_results, label):
    pred = va_steer_results + 1* va_cnn_results
    pred_label = np.argmax(softmax(pred), axis= -1)
    total = ((label-pred_label)==0).sum()

    return float(total) / label.shape[0] * 100

# va_cnn_results = np.loadtxt('cnn/0_pred.txt')
# va_steer_results = np.loadtxt('steer/0_pred.txt')
# label = np.loadtxt('steer/0_label.txt')
#
va_cnn_results = np.loadtxt('cnn/1_pred.txt')
va_steer_results = np.loadtxt('steer/1_pred.txt')
label = np.loadtxt('steer/1_label.txt')

print(acc(va_cnn_results,va_steer_results,label))

print(acc(va_cnn_results,va_cnn_results,label))

print(acc(va_steer_results,va_steer_results,label))
