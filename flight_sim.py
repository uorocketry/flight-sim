import numpy as np
from scipy import integrate
import math
import matplotlib.pyplot as plt

from utils import ur
from rigid_body import RigidBody
from environment import Environment

env = Environment()
rocket = RigidBody()

# launch site
lat = 47.986943*ur.degree	# [deg]
long = -81.848339*ur.degree	# [deg]
alt = 200*ur.meter			# [m]

print(env.g(lat, long, alt))

def step(t, state) -> np.array:
	g = [0*ur.meter/ur.second, env.g(lat, long, alt), 0*ur.meter/ur.second]
	rocket.x[1]
	rocket.apply_force(g)
	return rocket.step(t*ur.second, state)

solver = integrate.RK45(step, 0, rocket.get_state(), 10, 0.01, )

i = 0
while solver.t < solver.t_bound:
	solver.step()
	i += 1

print(i)
print(rocket.x)
