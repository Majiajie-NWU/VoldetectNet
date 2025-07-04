import json
import numpy as np
import matplotlib.pyplot as plt

with open('/home/cyb/majiajie/dukewangluo/DINO-main-merge-duoqi/logs/cocoevalresult.json', 'r', encoding='utf-8') as f:
    eval = json.load(f)

# 取出iouThr=0.5和0.75,类别为0类，不限定检测面积区间，单张图片最大检测数量为100时的precision
pr_array1 = np.array(eval['precision'])[0, :, 0, 0, 2]
pr_array2 = np.array(eval['precision'])[5, :, 0, 0, 2]
x = np.arange(0.0, 1.01, 0.01)
plt.xlabel('recall')
plt.ylabel('precision')
plt.xlim(0, 1.0)
plt.ylim(0, 1.01)
plt.grid(True)

plt.plot(x, pr_array1, 'b-', label='IoU=0.5')
plt.plot(x, pr_array2, 'y-', label='IoU=0.75')

plt.legend(loc='lower left')
plt.show()