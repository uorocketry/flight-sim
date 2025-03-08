import numpy as np
import math
from utils import ur

G = 6.6743E-11*ur.newton*ur.meter**2*ur.kilogram**-2		# gravitational constant	[N m^2 kg^-2]
M = 5.9722E24*ur.kilogram									# earth mass				[kg]

# WGS84 ellipsoid dimensions and gravity
G_E = 9.7803253359*ur.meter*ur.second**-2					# gravity at equator		[m s^-2]
G_P = 9.8321849378*ur.meter*ur.second**-2 					# gravity at poles			[m s^-2]
A = 6378137*ur.meter										# earth semi-major axis		[m]
B = 6356752*ur.meter										# earth semi-minor axis		[m]
F = 1 - B/A													# earth flattening
E = (1 - B**2/A**2)**0.5									# earth eccentricity
GM = G*M													# gravity formula constant	[N m^2 kg^-1]
K = (B*G_P - A*G_E)/(A*G_E)									# gravity formula constant	[-]

R_AIR = 287.052874*ur.joule*ur.kilogram**-1*ur.kelvin**-1	# specific gas constant		[J kg^-1 K^-1]

class Environment:
	def __init__(self) -> None:
		pass

	def geodetic_to_ecef(self, lat:float, long:float, alt:float) -> tuple[float, float, float]:
		print('lat, long:',lat, long)
		# convert west (negative) longditude to east
		if long < 0:
			long = 360*ur.degree + long
		print('lat, long:',lat, long)
		# convert from degrees to radians
		phi = np.deg2rad(lat)
		lam = np.deg2rad(long)
		print('phi, lam:',phi, lam)
		# prime vertical radius of curvature
		n = A**2/(A**2*np.cos(phi)**2 + B**2*np.sin(phi))**0.5
		print('n',n)
		n = A/(1-E**2*np.sin(phi)**2)
		print('n',n)
		# earth centered earth fixed cooordinates (origin at center of earth)
		# z through ellipsoid north pole, x through intersection of equator and prime meridian
		x = (n+alt)*np.cos(phi)*np.cos(lam)
		y = (n+alt)*np.cos(phi)*np.sin(lam)
		z = ((1-E**2)*n+alt)*np.sin(phi)
		return (x, y, z)

	def g(self, lat:float, long:float, alt:float) -> float:
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
		tc = ur.Quantity(20, ur.degC)
		tk = tc.to(ur.degK)
		return tk

	def pressure(self, alt:float) -> float:
		"""
		Air pressure in Pa vs altitude ASL
		"""
		# TODO:
		return 100000*ur.pascal

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
		return 0*ur.meter/ur.second

if __name__ == "__main__":
	env = Environment()

	lat = 47.986943*ur.degree		# [deg]
	long = -81.848339*ur.degree		# [deg]
	alt = 200*ur.meter				# [m ASL]

	ecef = env.geodetic_to_ecef(lat, long, alt)
	print('x:{}, y:{}, z:{}'.format(round(ecef[0]), round(ecef[1]), round(ecef[2])))

	print('eccentricity:', E)

	# Test gravity
	print(env.g(lat, long, alt))

	# Test temperature
	print(env.temperature(alt))

	# Test pressure
	print(env.pressure(alt))

	# Test air density
	print(env.rho(alt))

	# Test wind speed
	print(env.wind(lat, long, alt))
	
