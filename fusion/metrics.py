import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def softmax(data):
    e = np.exp(data - np.amax(data, axis=-1, keepdims=True))
    s = np.sum(e, axis=-1, keepdims=True)
    return e / s

def acc(va_steer_results, va_cnn_results, label):
    pred = va_steer_results + 1* va_cnn_results
    pred_label = np.argmax(softmax(pred), axis= -1)
    total = ((label-pred_label)==0).sum()

    return float(total) / label.shape[0] * 100

def pred(results):
    pred = results
    pred_label = np.argmax(softmax(pred), axis= -1)
    #total = ((label-pred_label)==0).sum()
    return pred_label

def fusion_pred(va_steer_results, va_cnn_results):
    pred = va_steer_results + 1* va_cnn_results
    pred_label = np.argmax(softmax(pred), axis= -1)
    return pred_label

# va_cnn_results = np.loadtxt('cnn/0_pred.txt')
# va_steer_results = np.loadtxt('steer/0_pred.txt')
# label = np.loadtxt('steer/0_label.txt')

va_cnn_results = np.loadtxt('cnn/1_pred.txt')
va_steer_results = np.loadtxt('steer/1_pred.txt')
label = np.loadtxt('steer/1_label.txt')

y_true=label
y_pred = pred(va_cnn_results)



print('precision',precision_recall_fscore_support(y_true, y_pred, average='macro'))

print('recal',precision_recall_fscore_support(y_true, y_pred, average='micro'))

print('fscore',precision_recall_fscore_support(y_true, y_pred, average='weighted'))


y_true=label
y_pred = pred(va_steer_results)

print('precision',precision_recall_fscore_support(y_true, y_pred, average='macro'))

print('recal',precision_recall_fscore_support(y_true, y_pred, average='micro'))

print('fscore',precision_recall_fscore_support(y_true, y_pred, average='weighted'))


y_true=label
y_pred = pred(va_steer_results)

print('precision',precision_recall_fscore_support(y_true, y_pred, average='macro'))

print('recal',precision_recall_fscore_support(y_true, y_pred, average='micro'))

print('fscore',precision_recall_fscore_support(y_true, y_pred, average='weighted'))


y_true=label
y_pred = fusion_pred(va_steer_results,va_cnn_results)

print('precision',precision_recall_fscore_support(y_true, y_pred, average='macro'))

print('recal',precision_recall_fscore_support(y_true, y_pred, average='micro'))

print('fscore',precision_recall_fscore_support(y_true, y_pred, average='weighted'))

labels= np.arange(60)
print(labels)
print(precision_recall_fscore_support(y_true, y_pred, average=None,labels=labels))


