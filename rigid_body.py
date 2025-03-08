import numpy as np
import math

# https://stackoverflow.com/questions/71056930/how-do-physics-engines-model-angular-velocity-and-angular-acceleration-in-3d
# https://csundergrad.science.uoit.ca/courses/2017-fall/csci3010u/lectures/rigid-body.pdf

class RigidBody:
	def __init__(self) -> None:
		self.x = np.zeros(3, float) # position
		self.r = np.zeros((3, 3), float) # orientation

		self.p = np.zeros(3, float) # linear momentum
		self.l = np.zeros(3, float) # angular momentum

		# inertia about the CG (0, 0)
		# Ixx Ixy Ixz
		# Iyx Iyy Iyz
		# Izx Izy Izz
		self.i = np.array([
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1]
		])
		self.i_inv = np.linalg.inv(self.i)
		self.m = 1.0

		self.v = np.zeros(3, float) # velcoity
		self.w = np.zeros(3, float) # angular velocity
		
		self.f = np.zeros(3, float) # force
		self.t = np.zeros(3, float) # torque
		self.dm = 0	# change in mass
					# change in moment of inertia
					# change in center of mass

		self.subcomponents = []
	
	def apply_force(self, f:np.array) -> None:
		self.f += f
	
	def apply_torque(self, t:np.array) -> None:
		self.t += t

	def get_state(self) -> np.array:
		return np.concat((self.x, self.r.flatten(), self.p, self.l))
	
	def set_state(self, arr:np.array):
		self.x = arr[0:3]
		self.r = np.reshape(arr[3:12], (3, 3))
		self.p = arr[12:15]
		self.l = arr[15:18]

	def step(self, t, state:np.array):
		self.set_state(state)

		w_tensor = np.array([
			[0, -self.w[2], self.w[1]],
			[self.w[2], 0, -self.w[0]],
			[-self.w[1], self.w[0], 0],
		])
		r_dot = np.linalg.matmul(w_tensor, self.r)

		# position dx/dt = v
		# rotation dr/dt = r_dot
		# linear momentum dp/dt = f
		# angular momentum dl/dt = t
		d_dt = np.concat((self.v, r_dot.flatten(), self.f, self.t))

		# calc velocity
		self.v = self.p/self.m
		# calc angular velocity
		i_inv_b = np.linalg.matmul(self.r, np.linalg.matmul(self.i_inv, np.transpose(self.r)))
		self.w = np.linalg.matmul(i_inv_b, self.l)
		# orthonormalize rotation matrix to reduce numerical error
		self.r, _ = np.linalg.qr(self.r)

		self.f = np.zeros(3, float)
		self.t = np.zeros(3, float)

		return d_dt

if __name__ == "__main__":
	print("a")
