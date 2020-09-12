import matplotlib.pyplot as plt
import numpy as np

list_displacement = [8989.0, 22317.666666666668, 1005.0, 5931.5, 11189.0, 1066.3333333333333, 7195.75, 6693.0, 29054.5, 7200.5, 11392.0, 49703.5, 4324.25, 9834.5, 9639.25, 9868.25, 10444.75, 9928.75, 9716.25, 9515.5, 4257.75, 3116.5, 3446.5, 6327.75, 3090.0, 7972.5, 11141.25, 7237.25, 3377.25, 3133.25, 2437.75, 2232.75, 4370.5, 2222.25, 1885.25]
list_velocity = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.04, 0.03, 0.03, 0.02, 0.0]
list_omega = [0.5235987755982988, -0.8726646259971648, 0.3490658503988659, 0.3490658503988659, -0.5235987755982988, 0.5235987755982988, -0.3490658503988659, 0.5235987755982988, -0.8726646259971648, 0.6981317007977318, 0.3490658503988659, -0.8726646259971648, 0.3490658503988659, 0.0, 0.0, -0.17453292519943295, 0.0, 0.0, 0.0, -0.17453292519943295, 0.0, 0.0, 0.0, -0.17453292519943295, 0.17453292519943295, 0.3490658503988659, -0.3490658503988659, -0.17453292519943295, 0.0, 0.0, 0.0, 0.17453292519943295, -0.17453292519943295, 0.0, 0.0]


t = np.arange(len(list_displacement))

plt.subplot(3, 1, 1)
plt.plot(t, list_displacement)
plt.xlabel('time step')
plt.ylabel('average displacement')
plt.yscale('linear')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, list_velocity)
plt.xlabel('time step')
plt.ylabel('forward velocity')
plt.yscale('linear')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, list_omega)
plt.xlabel('time step')
plt.ylabel('angular velocity')
plt.yscale('linear')
plt.grid(True)

plt.tight_layout()

#plt.savefig('vs_visualization.jpg', bbox_inches='tight')
#plt.close()