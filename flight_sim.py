import numpy as np
from scipy import integrate
import math
import matplotlib.pyplot as plt

from rocket import Rocket
from environment import Environment

# simulation
end_time = 10			# [s]

# launch site
lat0 = 47.986943			# [deg]
lon0 = -81.848339		# [deg]
alt0 = 200				# [m]
v0 = 60					# [m/s]

# set up initial rocket conditions
env = Environment(lat0, lon0, alt0)
rocket = Rocket()
rocket.x[1] = alt0
rocket.p[1] = v0*rocket.m

def step(t, state) -> np.array:
	g = [0, env.gravity(lat0, lon0, rocket.x[1]), 0]
	rocket.apply_force(g)
	return rocket.step(t, state)

# run simulation
solver = integrate.RK45(step, 0, rocket.get_state(), end_time, 0.01, )

plt_t = []
plt_y = []
while solver.t < solver.t_bound:
	plt_t.append(solver.t)
	plt_y.append(rocket.x[1])
	solver.step()

plt.plot(plt_t, plt_y)
plt.xlabel('time (s)')
plt.ylabel('altitude (m)')
plt.show()
