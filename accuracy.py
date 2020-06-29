
from L1_predection import accuracy_list
import matplotlib.pyplot as plt

test_fold_num = 6
plt.scatter(accuracy_list, test_fold_num * ['L1_pre'], color='black')
plt.xlabel("accuracy.percent %")
plt.ylabel("algorithm")
plt.tight_layout()
plt.savefig("test_accuracy.png")
