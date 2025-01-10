((3.56012724649096*x**3 - 2.00459727141559*x**2 - 1.05552997507537*x, (x <= 1.0) & (x >= 0)), (-15.8141978223798*x**3 + 56.1183779351968*x**2 - 59.1785051816878*x + 19.3743250688708, (x >= 1.0) & (x <= 1.5)), (10.4532797309315*x**3 - 62.0852710547041*x**2 + 118.126968303164*x - 69.2784116735549, (x >= 1.5) & (x <= 2.1)), (0.475767414403608*x**3 + 0.773056539421448*x**2 - 13.8755196445*x + 23.1233298898096, (x >= 2.1) & (x <= 3.0)))
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # !IMPORTANT
#
#
# x = np.arange(1, 11)
# y = np.arange(1, 11)
# weights = np.arange(1, 11)
# weights[3] = 10
# plt.scatter(x, y, c=weights, cmap='Greys', marker='+')
# plt.colorbar()
# plt.show()
#
#
import matplotlib.pyplot as plt
import numpy as np

# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
# plt.subplot(211)
# plt.imshow(np.random.random((100, 100)))
# plt.subplot(212)
# plt.imshow(np.random.random((100, 100)))
#
# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
# plt.colorbar(cax=cax)
#
# plt.show()

m = np.zeros((1,20))
for i in range(20):
    m[0,i] = (i*5)/100.0
print(m)
plt.imshow(m, cmap='viridis', aspect=2)
plt.yticks(np.arange(0))
plt.xticks(np.arange(0,25,5), [0,25,50,75,100])
plt.show()