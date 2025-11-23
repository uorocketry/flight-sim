import numpy as np
import math

G = 6.6743E-11					# gravitational constant	[N m^2 kg^-2]
M = 5.9722E24					# earth mass				[kg]

# WGS84 ellipsoid dimensions and gravity
G_E = 9.7803253359				# gravity at equator		[m s^-2]
G_P = 9.8321849378				# gravity at poles			[m s^-2]
A = 6378137						# earth semi-major axis		[m]
B = 6356752						# earth semi-minor axis		[m]
F = 1 - B/A						# earth flattening
E_SQ = 1 - B**2/A**2			# earth eccentricity squared
E = E_SQ**0.5					# earth eccentricity
GM = G*M						# gravity formula constant	[N m^2 kg^-1]
K = (B*G_P - A*G_E)/(A*G_E)		# gravity formula constant	[-]

R_AIR = 287.052874				# specific gas constant		[J kg^-1 K^-1]

class Environment:
	def __init__(self) -> None:
		pass

	def g(self, lat:float, lon:float, alt:float) -> float:
		"""
		Acceleration due to gravity vs altitude ASL and lat/long (degrees).
		"""
		# https://mwrona.com/posts/gravity-models/
		# https://en.wikipedia.org/wiki/Theoretical_gravity#Somigliana_equation
		# Somigliana equation with WGS84 parameters 
		g = G_E * ((1+K*np.sin(math.radians(lat))**2)/math.sqrt(1-E**2))
		# free-air gravity correction for altitude 
		g_loss = GM/(A+alt)**2 - GM/A**2
		return -(g - g_loss)

	def temperature(self, alt:float) -> float:
		"""
		Air temperature vs altitude ASL
		"""
 		# TODO:
		tk = 273.15 + 20
		return tk

	def pressure(self, alt:float) -> float:
		"""
		Air pressure in Pa vs altitude ASL
		"""
		# TODO:
		return 100000 # Pa

	def rho(self, alt:float) -> float:
		"""
		Air density in kg/m^3 vs altitude ASL
		"""
		# PV=mRT => m/V=P/(RT)
		return self.pressure(alt)/(R_AIR*self.temperature(alt))

	def wind(self, lat:float, long:float, alt:float) -> tuple[float, float, float]:
		"""
		Wind velocity vector vs altitude ASL and coordinates
		"""
		# TODO:
		return 0	# [m/s]

if __name__ == "__main__":
	env = Environment()

	lat = 47.986943		# [deg]
	lon = -81.848339	# [deg]
	alt = 200			# [m ASL]

	ecef = env.lla_to_ecef(lat, lon, alt)
	print('x:{}, y:{}, z:{}'.format(round(ecef[0]), round(ecef[1]), round(ecef[2])))

	print('lat:{}, long:{}, alt:{}'.format(lat, lon, alt))
	lla = env.ecef_to_lla(ecef[0], ecef[1], ecef[2])
	print('lat:{}, long:{}, alt:{}'.format(lla[0], lla[1], lla[2]))
	#lla = env.ecef_to_lla_hugues(ecef[0], ecef[1], ecef[2])
	#print('lat:{}, long:{}, alt:{}'.format(lla[0], lla[1], lla[2]))

	print('eccentricity:', E)

	# Test gravity
	print(env.g(lat, lon, alt))

	# Test temperature
	print(env.temperature(alt))

	# Test pressure
	print(env.pressure(alt))

	# Test air density
	print(env.rho(alt))

	# Test wind speed
	print(env.wind(lat, lon, alt))
	
