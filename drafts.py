# for i in range(100):
#     if i % 3 == 0:
#         print(i)
import matplotlib.pyplot as plt
import numpy as np

m1 = np.zeros((3,3,30))

# print(m1)

m2 = np.ones((3,3,2)) * 345

# print(m2)

m1[:, :, :2] += m2

# print(m1)

x_l, y_l, z_l = np.nonzero(m1 > 0)

print(x_l)
print(y_l)
print(z_l)
print(m1[m1>0])
data = m1
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cm = plt.colormaps['brg']
col = m1[m1>0]
ax.scatter(x_l, y_l, z_l, c=col, cmap=cm, alpha=0.5)
plt.show()
