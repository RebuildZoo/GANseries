import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

class confusion_eval:
    def __init__(self, pClassname_Lst):
        self.class_num = len(pClassname_Lst)
        self.classname_Lst = pClassname_Lst
        self.confuse_Arr = np.zeros((self.class_num,self.class_num), dtype = np.uint8)
    
    def add_data(self, p_pred_Vec, p_gt_Vec):
        assert len(p_gt_Vec) == len(p_pred_Vec), "unpair data"

        for truth_i, pred_i in zip(p_gt_Vec,  p_pred_Vec):
            self.confuse_Arr[int(truth_i), int(pred_i)]+=1
    
    def view_mat(self, pTitle:str):
        fig = plt.figure(figsize=(self.class_num, self.class_num))
        plt.tight_layout()
        ax = fig.add_subplot(111)
        res = ax.imshow(self.confuse_Arr, cmap=plt.cm.rainbow,interpolation='nearest')
        for x in range(self.class_num):
            for y in range(self.class_num):
                ax.annotate(str(self.confuse_Arr[x,y]), xy=(y, x), \
                    horizontalalignment='center',verticalalignment='center')
        plt.xticks(range(self.class_num), self.classname_Lst, rotation=0)
        plt.yticks(range(self.class_num), self.classname_Lst, rotation=0)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        plt.title(pTitle)
        plt.show()
        # pic_name = pTitle + '_Conf-Mat.png'
        # plt.savefig(pic_name, bbox_inches='tight')


