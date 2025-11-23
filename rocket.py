import numpy as np
from solid_motor import SolidMotor
from rigid_body import RigidBody
from parachute import Parachute
import aerodynamics

class Rocket(RigidBody):
	def __init__(self) -> None:
		self.parachutes = []
		super().__init__()

	def step(self, t, state) -> np.array:
		# TODO: apply aero loads, thrust, parachute loads
		return super().step(t, state)

if __name__ == '__main__':
	print('------ testing rocket.py ------')
	# TODO: Add tests
