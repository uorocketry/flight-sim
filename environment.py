import numpy as np
import math
import pymap3d as pm

G = 6.6743E-11                # gravitational constant    [N m^2 kg^-2]
M = 5.9722E24                 # earth mass                [kg]
R_AIR = 287.052874            # specific gas constant     [J kg^-1 K^-1]

# WGS84 ellipsoid dimensions and gravity
G_E = 9.7803253359            # gravity at equator       [m s^-2]
G_P = 9.8321849378            # gravity at poles         [m s^-2]
A = 6378137                   # earth semi-major axis    [m]
B = 6356752                   # earth semi-minor axis    [m]
F = 1 - B/A                   # earth flattening         [-]
E_SQ = 1 - B**2/A**2          # earth eccentricity square [-]
E = E_SQ**0.5                 # earth eccentricity       [-]
GM = G*M                      # gravity formula constant [N m^2 kg^-1]
K = (B*G_P - A*G_E)/(A*G_E)   # gravity formula constant [-]
P0 = 101325                   # sea-level pressure      [Pa]
T0 = 288.15                   # sea-level temperature   [K]
# Ra is redundant with R_AIR; keep for compatibility if needed
Ra = 287.05                   # gas constant for R
L = 0.0065                    # Lapse rate               [K/m]
V_ref = 5.0                   # m/s at 10 m AGL
z_ref = 10.0                  # reference height [m]
alpha = 0.14                  # open terrain


class Environment:
	def __init__(self, lat0: float, lon0: float, alt0: float, wind: list[float] = []) -> None:
		# origin for enu coordinates
		self.lat0 = lat0
		self.lon0 = lon0
		self.alt0 = alt0

	def lla_to_enu(self, lat: float, lon: float, alt: float) -> tuple[float, float, float]:
		return pm.geodetic2enu(
			lat, lon, alt,
			self.lat0, self.lon0, self.alt0,
			ell=pm.utils.Ellipsoid('wgs84')
		)

	def enu_to_lla(self, east: float, north: float, up: float) -> tuple[float, float, float]:
		return pm.enu2geodetic(
			east, north, up,
			self.lat0, self.lon0, self.alt0,
			ell=pm.utils.Ellipsoid('wgs84')
		)

	def gravity(self, lat: float, lon: float, alt: float) -> float:
		"""
		Acceleration due to gravity vs altitude ASL and lat/long (degrees).
		"""
		# https://mwrona.com/posts/gravity-models/
		# https://en.wikipedia.org/wiki/Theoretical_gravity#Somigliana_equation
		# Somigliana equation with WGS84 parameters
		phi = math.radians(lat)
		g = G_E * ((1 + K * np.sin(phi) ** 2) / math.sqrt(1 - (E ** 2) * np.sin(phi) ** 2))
		# free-air gravity correction for altitude
		g_loss = GM / (A + alt) ** 2 - GM / A ** 2
		return -(g - g_loss)

	def temperature(self, alt: float) -> float:
		"""
		Air temperature vs altitude ASL
		"""
		tk = T0 - L * alt
		return tk

	def pressure(self, alt: float) -> float:
		"""
		Air pressure in Pa vs altitude ASL
		Uses barometric formula for the troposphere.
		"""
		base = 1 - (L * alt) / T0
		if base <= 0:
			# avoid negative base for extreme altitudes; return near-zero pressure
			return 0.0
		return P0 * base ** (G_E / (R_AIR * L))

	def rho(self, alt: float) -> float:
		"""
		Air density in kg/m^3 vs altitude ASL
		"""
		return self.pressure(alt) / (R_AIR * self.temperature(alt))

	def wind(self, lat: float, lon: float, alt: float) -> tuple[float, float, float]:
		"""
		Wind velocity vector vs altitude ASL and coordinates
		Input Altitude in ASL then use of AGL for Wind model
		"""
		z = max(alt - self.alt0, 0.1)  # altitude above ground level

		V = V_ref * (z / z_ref) ** alpha

		# East, North, Up
		return (V, 0.0, 0.0)


if __name__ == '__main__':
	print('------ testing environment.py ------')

	lat = 47.986943     # [deg]
	lon = -81.848339    # [deg]
	alt = 200           # [m ASL]

	env = Environment(lat, lon, alt)

	# test geodetic conversions
	print('original lla: lat: {:.6f}, long: {:.6f}, alt: {:.2f}'.format(lat, lon, alt))
	enu = env.lla_to_enu(lat + 0.001, lon - 0.001, alt + 100)
	print('enu: e: {:.2f} m, n: {:.2f} m, u: {:.2f} m'.format(enu[0], enu[1], enu[2]))
	lla = env.enu_to_lla(enu[0], enu[1], enu[2])
	print('lla: lat: {:.6f}, long: {:.6f}, alt: {:.2f}'.format(lla[0], lla[1], lla[2]))
	enu = env.lla_to_enu(lla[0], lla[1], lla[2])
	print('enu: e: {:.2f} m, n: {:.2f} m, u: {:.2f} m'.format(enu[0], enu[1], enu[2]))

	# Test gravity
	print('gravity: {:.4f} m/s^2'.format(env.gravity(lat, lon, alt)))

	# Test temperature
	print('temperature: {:.2f} K'.format(env.temperature(alt)))

	# Test pressure
	print('pressure: {:.2f} Pa'.format(env.pressure(alt)))

	# Test air density
	print('air density: {:.4f} kg/m^3'.format(env.rho(alt)))

	# Test wind speed
	print('wind speed: {:.2f} m/s'.format(env.wind(lat, lon, alt)[0]))
