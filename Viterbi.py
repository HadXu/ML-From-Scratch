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


theta_1 = [pi[i]*B[i,0] for i in range(3)]
theta_1_index = np.argmax([pi[i]*B[i,0] for i in range(3)])


print(theta_1_index)

theta_2 = [max([theta_1[j]*A[j,i]*B[i,1] for j in range(3)]) for i in range(3)]
theta_2_index = [np.argmax([theta_1[j]*A[j,i]*B[i,1] for j in range(3)]) for i in range(3)]


theta_3 = [max([theta_2[j]*A[j,i]*B[i,0] for j in range(3)]) for i in range(3)]
theta_3_index = [np.argmax([theta_2[j]*A[j,i]*B[i,0] for j in range(3)]) for i in range(3)]


theta_4 = [max([theta_3[j]*A[j,i]*B[i,1] for j in range(3)]) for i in range(3)]
theta_4_index = [np.argmax([theta_3[j]*A[j,i]*B[i,1] for j in range(3)]) for i in range(3)]


## 那么最优路径的解为

p4 = np.argmax(theta_4_index)
p3 = theta_4_index[p4]
p2 = theta_3_index[p3]
p1 = theta_2_index[p2]
print(p1,p2,p3,p4)


