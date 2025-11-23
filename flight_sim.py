import numpy as np
from scipy import integrate
import math
import matplotlib.pyplot as plt

from rigid_body import RigidBody
from environment import Environment


# launch site
lat = 47.986943			# [deg]
long = -81.848339		# [deg]
alt = 200				# [m]

env = Environment(lat, long, alt)
rocket = RigidBody()

print(env.gravity(lat, long, alt))

def step(t, state) -> np.array:
	g = [0, env.gravity(lat, long, alt), 0]
	rocket.x[1]
	rocket.apply_force(g)
	return rocket.step(t, state)

solver = integrate.RK45(step, 0, rocket.get_state(), 10, 0.01, )

i = 0
while solver.t < solver.t_bound:
	solver.step()
	i += 1

print(i)
print(rocket.x)
