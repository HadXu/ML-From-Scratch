import numpy as np

### 维比特算法

# 红 白 红
A = np.array([[0.5,0.2,0.3],
			[0.3,0.5,0.2],
			[0.2,0.3,0.5]])

B = np.array([[0.5,0.5],
			[0.4,0.6],
			[0.7,0.3]])

pi = np.array([[0.2,0.4,0.4]]).T


a_1 = [pi[i]*B[i,0] for i in range(3)]

a_2 = [sum([a_1[i]*A[i,j] for i in range(3)])*B[j,1] for j in range(3)]

a_3 = [sum([a_2[i]*A[i,j] for i in range(3)])*B[j,0] for j in range(3)]



## 最终

res = sum(a_3)

print(res)