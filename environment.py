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

def lla_to_ecef(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
	# convert degrees to radians
	lat_rad = np.deg2rad(lat)
	lon_rad = np.deg2rad(lon)

	# prime vertical radius of curvature
	n = A / np.sqrt(1 - E**2 * np.sin(lat_rad)**2)

	# ECEF coordinates cooordinates (origin at center of earth)
	# z through ellipsoid north pole, x through intersection of equator and prime meridian
	x = (n + alt) * np.cos(lat_rad) * np.cos(lon_rad)
	y = (n + alt) * np.cos(lat_rad) * np.sin(lon_rad)
	z = ((1 - E**2) * n + alt) * np.sin(lat_rad)

	return (x, y, z)

# Converting ECEF (x, y, z) coordinates to (lat, lon, alt) values
# Author: Karthik Venkataramani
# Date 02/23/2019

def ecef_to_lla(x, y, z):
	# distance from z-axis
	r = np.sqrt(x**2 + y**2)

	# Calculate auxiliary values
	ep_sq = (A**2 - B**2) / B**2  # Second eccentricity squared
	ee = (A**2 - B**2)  # Difference of squared axes

	f = (54 * B**2) * (z**2)
	g = r**2 + (1 - E_SQ) * (z**2) - E_SQ * ee * 2

	c = (E_SQ**2) * f * r**2 / (g**3)
	s = np.cbrt(1 + c + np.sqrt(c**2 + 2*c))
	
	p = f / (3.0 * g**2 * (s + (1.0 / s) + 1)**2)

	q = np.sqrt(1 + 2 * p * E_SQ**2)
	r_0 = -(p * E_SQ * r) / (1 + q) + np.sqrt(0.5 * A**2 * (1 + (1.0 / q)) - p * (z**2) * (1 - E_SQ) / (q * (1 + q)) - 0.5 * p * (r**2))

	# Calculate u and v for altitude and latitude correction
	u = np.sqrt((r - E_SQ * r_0)**2 + z**2)
	v = np.sqrt((r - E_SQ * r_0)**2 + (1 - E_SQ) * z**2)

	# corrected altitude component
	z_0 = (B**2) * z / (A * v)
	
	h = u * (1 - B**2 / (A * v))
	lat_rad = np.arctan((z + ep_sq * z_0) / r)
	lon_rad = np.arctan2(y, x)
	return (np.rad2deg(lat_rad), np.rad2deg(lon_rad), h)

def lla_to_enu(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
	pass

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
	print('------ testing environment.py ------')

	lat = 47.986943		# [deg]
	lon = -81.848339	# [deg]
	alt = 200			# [m ASL]

	print('original lla: lat: {}, long: {}, alt: {}'.format(lat, lon, alt))
	ecef = lla_to_ecef(lat, lon, alt)
	print('lla to ecef:  x: {:.2f} m, y: {:.2f} m, z: {:.2f} m'.format(ecef[0], ecef[1], ecef[2]))
	lla = ecef_to_lla(ecef[0], ecef[1], ecef[2])
	print('ecef to lla:  lat: {}, long: {}, alt: {}'.format(lla[0], lla[1], lla[2]))
	
	env = Environment()

	# Test gravity
	print('gravity: {:.4f} m/s^2'.format(env.g(lat, lon, alt)))

	# Test temperature
	print('temperature: {:.2f} K'.format(env.temperature(alt)))

	# Test pressure
	print('pressure: {:.2f} Pa'.format(env.pressure(alt)))

	# Test air density
	print('air density: {:.4f} kg/m^3'.format(env.rho(alt)))

	# Test wind speed
	print('wind speed: {:.2f} m/s'.format(env.wind(lat, lon, alt)))
	
