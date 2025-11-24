import csv
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# TODO: Accept other file formats?
# TODO: Include simple drag and lift calculations from geometry (Barowman's, etc.)
# TODO: Add drag, lift, normal force, etc. given reference area, environment params

M_TO_IN = 0.0254

class Aerodynamics:
	def __init__(self, file) -> None:

		with open(file, 'r') as csvfile:
			reader = csv.reader(csvfile)
			data = list(reader)
		
		header = data[0]
		data = data[1:]
		data = [[float(x) for x in row] for row in data]

		idx_mach = header.index('Mach')
		idx_alpha = header.index('Alpha')
		idx_cd_power_off = header.index('CD Power-Off')
		idx_cd_power_on = header.index('CD Power-On')
		idx_ca_power_off = header.index('CA Power-Off')
		idx_ca_power_on = header.index('CA Power-On')
		idx_cl = header.index('CL')
		idx_cn = header.index('CN')
		idx_cp = header.index('CP')
		idx_re = header.index('Reynolds Number')

		# sort array (by ascending mach number and alpha)
		data = sorted(data, key=lambda x: (x[idx_mach], x[idx_alpha]))

		# get all alpha values
		alpha_values = []
		alpha_values.append(data[0][idx_alpha])
		for i in range(1, len(data)):
			alpha = data[i][idx_alpha]
			if alpha < alpha_values[-1]:
				break # recorded all alphas
			alpha_values.append(alpha)

		# ensure array is not jagged 
		num_alpha = len(alpha_values)
		num_mach = len(data)//num_alpha
		if len(data) != num_mach*num_alpha:
			raise ValueError('Inconsistent number of mach/alpha values')
		
		self._cd_power_off = np.empty([num_mach, num_alpha], dtype=float)	# drag coefficient, motor on			[-]
		self._cd_power_on = np.empty([num_mach, num_alpha], dtype=float)	# drag coefficient, motor on			[-]
		self._ca_power_off = np.empty([num_mach, num_alpha], dtype=float)	# axial force coefficient, motor on		[-]
		self._ca_power_on = np.empty([num_mach, num_alpha], dtype=float)	# axial force coefficient, motor off	[-]

		self._cl = np.empty([num_mach, num_alpha], dtype=float)				# lift coefficient						[-]
		self._cn = np.empty([num_mach, num_alpha], dtype=float)				# normal force coefficient				[-]
		self._cp = np.empty([num_mach, num_alpha], dtype=float)				# center of pressure 					[in 
		self._re = np.empty([num_mach, num_alpha], dtype=float)				# reynolds number						[-]

		# loop through mach numbers
		mach_values = []
		for i in range(num_mach):
			mach = data[i*num_alpha][idx_mach]
			mach_values.append(mach)

			# loop through alphas
			for j in range(num_alpha):
				# check that alpha values are consistent
				line = data[i*num_alpha+j]
				alpha = line[idx_alpha]
				if alpha_values[j] != alpha:
					raise ValueError('Inconsistent number of mach/alpha values')
				
				self._cd_power_off[i][j] = line[idx_cd_power_off]
				self._cd_power_on[i][j] = line[idx_cd_power_on]
				self._ca_power_off[i][j] = line[idx_ca_power_off]
				self._ca_power_on[i][j] = line[idx_ca_power_on]

				self._cl[i][j] = line[idx_cl]
				self._cn[i][j] = line[idx_cn]
				self._cp[i][j] = line[idx_cp]*M_TO_IN
				self._re[i][j] = line[idx_re]
		
		# configure interpolation
		self._cd_power_off_i = RegularGridInterpolator((mach_values, alpha_values), self._cd_power_off, bounds_error=False, fill_value=None)
		self._cd_power_on_i = RegularGridInterpolator((mach_values, alpha_values), self._cd_power_on, bounds_error=False, fill_value=None)
		self._ca_power_off_i = RegularGridInterpolator((mach_values, alpha_values), self._ca_power_off, bounds_error=False, fill_value=None)
		self._ca_power_on_i = RegularGridInterpolator((mach_values, alpha_values), self._ca_power_on, bounds_error=False, fill_value=None)

		self._cl_i = RegularGridInterpolator((mach_values, alpha_values), self._cl, bounds_error=False, fill_value=None)
		self._cn_i = RegularGridInterpolator((mach_values, alpha_values), self._cn, bounds_error=False, fill_value=None)
		self._cp_i = RegularGridInterpolator((mach_values, alpha_values), self._cp, bounds_error=False, fill_value=None)
		self._re_i = RegularGridInterpolator((mach_values, alpha_values), self._re, bounds_error=False, fill_value=None)

	def cd(self, mach, alpha, power_on):
		"""
		Drag coefficient vs mach number and angle of attack (degrees)
		"""
		if power_on:
			return self._cd_power_on_i((mach, alpha))
		else:
			return self._cd_power_off_i((mach, alpha))

	def ca(self, mach, alpha, power_on):
		"""
		Axial force coefficient vs mach number and angle of attack (degrees)
		"""
		if power_on:
			return self._ca_power_on_i((mach, alpha))
		else:
			return self._ca_power_off_i((mach, alpha))
		
	def cl(self, mach, alpha):
		"""
		Lift coefficient vs mach number and angle of attack (degrees)
		"""
		return self._cl_i((mach, alpha))

	def cn(self, mach, alpha):
		"""
		Normal force coefficient vs mach number and angle of attack (degrees)
		"""
		return self._cn_i((mach, alpha))

	def cp(self, mach, alpha):
		"""
		Center of pressure (m) vs mach number and angle of attack (degrees)
		"""
		return self._cp_i((mach, alpha))
	
	def re(self, mach, alpha):
		"""
		Reynolds number vs mach number and angle of attack (degrees)
		"""
		return self._re_i((mach, alpha))


if __name__ == '__main__':
	print('------ testing aerodynamics.py ------')
	import matplotlib.pyplot as plt
	aero = Aerodynamics('aero_data/ra_aero_data_sample')

	alpha = 2.0
	mach = np.linspace(0, 3, 100)

	ca = aero.ca(mach, alpha, False)
	plt.figure()
	plt.plot(mach, ca)
	plt.xlabel('mach number')
	plt.ylabel('axial force coefficient (power off)')

	cd = aero.cd(mach, alpha, False)
	plt.figure()
	plt.plot(mach, cd)
	plt.xlabel('mach number')
	plt.ylabel('drag coefficient (power off)')

	cl = aero.cl(mach, alpha)
	plt.figure()
	plt.plot(mach, cl)
	plt.xlabel('mach number')
	plt.ylabel('lift coefficient')

	cn = aero.cn(mach, alpha)
	plt.figure()
	plt.plot(mach, cn)
	plt.xlabel('mach number')
	plt.ylabel('normal force coefficient')

	cp = aero.cp(mach, alpha)
	plt.figure()
	plt.plot(mach, cp)
	plt.xlabel('mach number')
	plt.ylabel('center of pressure (m from tip)')

	plt.show()
