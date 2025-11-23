import xml.etree.ElementTree as ET
from rigid_body import RigidBody

class SolidMotor:
	def __init__(self, file):
		tree = ET.parse(file)
		for elem in tree.iter():
			print(elem.tag, elem.attrib)

	def get_thrust(self, t) -> float:
		"""
		Motor thrust in N vs time since ignition
		"""
		pass

if __name__ == '__main__':
	print('------ testing solid_motor.py ------')

	SolidMotor('motor_files/Cesaroni_9977M2245-P.rse')
