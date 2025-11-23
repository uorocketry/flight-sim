import numpy as np
from solid_motor import SolidMotor
from rigid_body import RigidBody
import aerodynamics

class Rocket(RigidBody):
	def __init__(self) -> None:
		self.body = RigidBody()

	def step(self, t, state) -> np.array:
		# TODO: override rigidbody step
		# also apply aero loads, thrust, parachute loads, etc.
		pass

if __name__ == '__main__':
	print('------ testing rocket.py ------')
	# TODO: Add tests
