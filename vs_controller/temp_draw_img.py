import matplotlib.pyplot as plt
import matplotlib
import numpy as np 
'''
#fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True)
fig = plt.figure(figsize=(7, 5))
r, c = 2, 2
ax = fig.add_subplot(r, c, 1)
x = np.array([0, 1, 2, 3, 4])
y = np.array([0.876, 0.866, 0.823, 0.799, 0.802])
my_x_ticks = ['0', '4', '8', '16', '32']
ax.set_xticklabels(my_x_ticks)
ax.set_ylim((0.78, 0.90))
ax.set_xticks(x)
ax.set_yticks(np.arange(0.78, 0.90, 0.02))
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
ax.plot(x, y)
ax.grid(True)
ax.set_xlabel('$\sigma$')
ax.set_ylabel('success rate')
ax.set_title('Noise on Correspondence Offset')
plt.show()
'''

#'''
fig = plt.figure(figsize=(7, 5))
r, c = 2, 2
ax = fig.add_subplot(r, c, 1)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0.876, 0.865, 0.864, 0.851, 0.806, 0.770])
my_x_ticks = ['1.0', '0.9', '0.8', '0.7', '0.6', '0.5']
ax.set_xticklabels(my_x_ticks)
ax.set_ylim((0.75, 0.90))
ax.set_xticks(x)
ax.set_yticks(np.arange(0.75, 0.90, 0.02))
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
ax.plot(x, y)
ax.grid(True)
ax.set_ylabel('success rate')
ax.set_xlabel('Coverage')
ax.set_title('Noise on Correspondence Density')
plt.show()
#plt.savefig('noise.jpg', bbox_inches='tight')
#'''


'''
plt.subplot(211)
x = np.array([0, 1, 2, 3, 4])
y = np.array([0.876, 0.866, 0.823, 0.799, 0.802])
my_x_ticks = ['0', '4', '8', '16', '32']
plt.yticks(np.arange(0.70, 0.90, 0.02))
vals = plt.get_yticks()
plt.plot(x,y)
plt.grid(axis='y', linestyle='-')
plt.subplot(212)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0.876, 0.865, 0.864, 0.851, 0.806, 0.770])
my_x_ticks = ['1.0', '0.9', '0.8', '0.7', '0.6', '0.5']
plt.yticks(np.arange(0.70, 0.90, 0.02))
plt.plot(x,y)
plt.grid(axis='y', linestyle='-')
plt.show()
'''