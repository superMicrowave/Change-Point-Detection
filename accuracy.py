
from L1_predection import accuracy_list
from baseline import base_test_acc
from SpatialPyramidPooling import cnn_test_acc
import matplotlib.pyplot as plt

test_fold_num = 6

plt.scatter(accuracy_list, test_fold_num * ['L1_pre'], color='black')
plt.scatter(base_test_acc, test_fold_num * ['Baseline'], color='green')
plt.scatter(cnn_test_acc, test_fold_num * ['Cnn'], color='blue')
plt.xlabel("accuracy.percent %")
plt.ylabel("algorithm")
plt.tight_layout()
plt.savefig("test_accuracy.png")
