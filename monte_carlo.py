import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from time import process_time, time
from numpy.random import choice, normal
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os

from rocketpy import Environment, Flight, Function, MonteCarlo, Rocket, SolidMotor
from rocketpy.stochastic import (
    StochasticEnvironment,
    StochasticFlight,
    StochasticNoseCone,
    StochasticParachute,
    StochasticRailButtons,
    StochasticRocket,
    StochasticSolidMotor,
    StochasticTail,
    StochasticTrapezoidalFins,
)

from rocketpy import MonteCarlo


print("="*80)
print("PERIGE-GEE ROCKET DISPERSION ANALYSIS")
print("6 CASES WITH SPECIFIED WIND CONDITIONS")
print("="*80)

PERIGE_GEE_PARAMS = {
    'name': 'Perige-Gee',
    'mass': 7.257,  # kg (dry)
    'radius': 40.45 / 1000,  # m
    'motor_impulse': 1415.15,  # N-s
    'burn_time': 5.274,  # s
    'cd_s_drogue': 0.349 * 1.3,  # m²
    'cd_s_main': 1.5,  # m² (adjust based on your main chute)
    'rail_length': 5.7,  # m
    'inclination': 84.7,  # degrees
    'heading': 53,  # degrees from north
}

LAUNCH_SITE = {
    'name': 'Timmins',
    'latitude': 40.910,
    'longitude': -119.056,
    'elevation': 1196,  # m
    'utm_zone': '11T',
}

# Analysis cases
ANALYSIS_CASES = {
    1: {'name': 'Ballistic', 'drogue': False, 'main': False, 'wind': 'high_winds_august'},
    2: {'name': 'Drogue Only', 'drogue': True, 'main': False, 'wind': 'high_winds_august'},
    3: {'name': 'Main at Apogee', 'drogue': False, 'main': True, 'wind': 'high_winds_august'},
    4: {'name': 'Ballistic', 'drogue': False, 'main': False, 'wind': 'typical_august'},
    5: {'name': 'Drogue Only', 'drogue': True, 'main': False, 'wind': 'typical_august'},
    6: {'name': 'Main at Apogee', 'drogue': False, 'main': True, 'wind': 'typical_august'},
}

# Wind profiles -simplified ->use real atmospheric datas
WIND_PROFILES = {
    'high_winds_august': {
        'description': '20 mph ground winds with higher winds aloft (historical worst-case August)',
        'ground_speed': 8.94,  # 20 mph in m/s
        'aloft_multiplier': 2.5,  # Winds aloft are 2.5x stronger
        'direction_ground': 270,  # West wind
        'direction_variation': 30,  # ±30 degree variation
    },
    'typical_august': {
        'description': 'Typical mid-August multi-level winds',
        'ground_speed': 4.47,  # 10 mph in m/s
        'aloft_multiplier': 1.5,  # Moderate increase with altitude
        'direction_ground': 240,  # WSW wind
        'direction_variation': 45,  # ±45 degree variation
    }
}

def create_wind_profile(wind_type, altitude_m):
    """Create wind speed and direction profile based on altitude."""
    profile = WIND_PROFILES[wind_type]
    
    # Linear increase with altitude up to 5000m
    if altitude_m <= 5000:
        speed_factor = 1 + (profile['aloft_multiplier'] - 1) * (altitude_m / 5000)
    else:
        speed_factor = profile['aloft_multiplier']
    
    wind_speed = profile['ground_speed'] * speed_factor
    
    # Direction variation with altitude
    if altitude_m <= 1000:
        direction = profile['direction_ground']
    else:
        # Veer with altitude 
        direction = profile['direction_ground'] + (altitude_m - 1000) / 4000 * 30
    
    # Add random variation
    direction_variation = np.random.normal(0, profile['direction_variation'])
    direction = (direction + direction_variation) % 360
    
    # Convert to u, v components
    wind_u = -wind_speed * np.sin(np.radians(direction))  # East-West
    wind_v = -wind_speed * np.cos(np.radians(direction))  # North-South
    
    return wind_u, wind_v, wind_speed, direction

    
def create_custom_environment(wind_type,date=(2026, 1, 15,12)):
    #Base environment object
    env = Environment(
        date = date,
        latitude=LAUNCH_SITE['latitude'],
        longitude=LAUNCH_SITE['longitude'],
        elevation=LAUNCH_SITE['elevation'],
    )
    altitudes = np.linspace(0, 10000, 21)
    wind_u = []
    wind_v = []
    wind_speeds = []
    wind_directions = []
    for alt in altitudes : 
        u, v, speed,direction, = create_wind_profile(wind_type,alt)
        wind_u.append(u)
        wind_v.append(v)
        wind_speeds.append(speed)
        wind_directions.append(direction)

    atmosphere_dict = {
        'height': altitudes.tolist(),
        'wind_u': wind_u,
        'wind_v': wind_v,
        'wind_speed': wind_speeds,
        'wind_direction': wind_directions,
    }

    env.set_atmospheric_model(type='standard_atmosphere')
    
    # overriding wind functions
    env.wind_velocity_x = lambda h: np.interp(h, altitudes, wind_u)
    env.wind_velocity_y = lambda h: np.interp(h, altitudes, wind_v)
    
    # Storing wind data
    env.wind_profile = {
        'altitude': altitudes,
        'wind_u': wind_u,
        'wind_v': wind_v,
        'wind_speed': wind_speeds,
        'wind_direction': wind_directions,
    }
    return env

def create_perige_gee_rocket(motor_params, rocket_params, drogue = True, main = True) :
    #Base motor
    motor = SolidMotor(
        thrust_source="path_to_thrust_curve.csv",  # Replace with actual path
        dry_mass=1.0,
        dry_inertia=(0.1, 0.1, 0.01),
        grains_center_of_mass_position=0.1,
        grain_number=1,
        grain_density=1700,
        grain_outer_radius=0.03,
        grain_initial_inner_radius=0.01,
        grain_initial_height=0.1,
        grain_separation=0.005,
        nozzle_radius=0.03,
        throat_radius=0.01,
    )

    #Base Rocket
    rocket = Rocket(
        radius=rocket_params['radius'],
        mass=rocket_params['mass'],
        inertia=rocket_params['inertia'],
        power_off_drag="aerodynamics/cd_power_off.csv",
        power_on_drag="aerodynamics/cd_power_on.csv",
        center_of_mass_without_motor=rocket_params['com_without_motor'],
        coordinate_system_orientation="tail_to_nose",
    )

    rocket.set_rail_buttons(
        upper_button_position=0.224,
        lower_button_position=-0.93,
        angular_position=30
    )
    
    #  motor
    rocket.add_motor(motor, position=0)
    
    #  nose cone
    rocket.add_nose(
        length=0.274,
        kind="vonKarman",
        position=1.134 + 0.274,
    )
    
    #  fins
    rocket.add_trapezoidal_fins(
        n=3,
        span=0.077,
        root_chord=0.058,
        tip_chord=0.018,
        position=-0.906,
        cant_angle=0,
        airfoil=None,
    )
    
    # parachutes 
    if drogue:
        def drogue_trigger(p, h, y):
            vertical_velocity = y[5]
            return True if vertical_velocity < 0 else False
        
        rocket.add_parachute(
            "Drogue",
            cd_s=rocket_params['cd_s_drogue'],
            trigger=drogue_trigger,
            sampling_rate=105,
            lag=1.73,  # 1.0 + 0.73 seconds
            noise=(0, 8.3, 0.5),
        )
    if main:
        def main_trigger(p, h, y):
            return h < 300  # Deploy at 300m AGL
        
        rocket.add_parachute(
            "Main",
            cd_s=rocket_params['cd_s_main'],
            trigger=main_trigger,
            sampling_rate=105,
            lag=1.73,
            noise=(0, 8.3, 0.5),
        )
    
    return rocket

def run_monte_carlo_simulation(case_config, num_simulations=100):
    print(f"\nRunning Monte Carlo for: {case_config['name']} ({case_config['wind']})")
    print(f"Configuration: Drogue={case_config['drogue']}, Main={case_config['main']}")
    
    # Create environment with specified wind profile
    env = create_custom_environment(case_config['wind'])
    
    # Define motor parameters (adjust based on your motor)
    motor_params = {
        'dry_mass': 1.0,
        'dry_inertia': (1.675, 1.675, 0.003),
        'grain_com_position': -0.571,
        'grain_number': 6,
        'grain_density': 1707,
        'grain_outer_radius': 21.4 / 1000,
        'grain_inner_radius': 9.65 / 1000,
        'grain_height': 120 / 1000,
        'grain_separation': 6 / 1000,
        'nozzle_radius': 21.642 / 1000,
        'throat_radius': 8 / 1000,
        'burn_time': 5.274,
    }
    
    # Define rocket parameters
    rocket_params = {
        'radius': 40.45 / 1000,
        'mass': 7.257,
        'inertia': (3.675, 3.675, 0.007),
        'com_without_motor': 0,
        'cd_s_drogue': PERIGE_GEE_PARAMS['cd_s_drogue'],
        'cd_s_main': PERIGE_GEE_PARAMS['cd_s_main'],
    }
    
    # Lists to store results
    impact_points = []
    apogee_points = []
    flight_data = []
    
    successful_sims = 0
    
    for i in range(num_simulations):
        try:
            # Create rocket with current configuration
            rocket = create_perige_gee_rocket(
                motor_params, 
                rocket_params, 
                drogue=case_config['drogue'],
                main=case_config['main']
            )
            
            # Create flight
            flight = Flight(
                rocket=rocket,
                environment=env,
                rail_length=PERIGE_GEE_PARAMS['rail_length'],
                inclination=PERIGE_GEE_PARAMS['inclination'],
                heading=PERIGE_GEE_PARAMS['heading'],
                max_time=600,
                terminate_on_apogee=False,
            )
            
            # Store results
            impact_points.append((flight.x_impact, flight.y_impact))
            apogee_points.append((flight.apogee_x, flight.apogee_y))
            
            flight_info = {
                'apogee': flight.apogee,
                'apogee_time': flight.apogee_time,
                'max_velocity': flight.max_speed,
                'impact_velocity': flight.impact_velocity,
                'flight_time': flight.t_final,
            }
            flight_data.append(flight_info)
            
            successful_sims += 1
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_simulations} simulations")
                
        except Exception as e:
            if i < 5:  # Print of first few errors only
                print(f"  Simulation {i + 1} failed: {str(e)}")
            continue
    
    print(f"  Successfully completed {successful_sims}/{num_simulations} simulations")
    
    return {
        'impact_points': np.array(impact_points),
        'apogee_points': np.array(apogee_points),
        'flight_data': flight_data,
        'success_rate': successful_sims / num_simulations,
    }

def calculate_dispersion_ellipses(points, sigmas=[1, 2]):
    """Calculating dispersion ellipses for given points."""
    if len(points) < 2:
        return []
    
    # calculating mean and covariance
    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)
    
    # Calculating eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sorting eigenvalues and eigenvectors
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # angle of the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    
    ellipses = []
    for sigma in sigmas:
        # Width and height of ellipse
        width = 2 * sigma * np.sqrt(eigenvalues[0])
        height = 2 * sigma * np.sqrt(eigenvalues[1])
        
        ellipses.append({
            'mean': mean,
            'width': width,
            'height': height,
            'angle': angle,
            'sigma': sigma,
        })
    
    return ellipses


# MAIN DISPERSION ANALYSIS


def run_all_dispersion_analyses(num_simulations=50):
    """Run all 6 dispersion analysis cases."""
    print("\n" + "="*80)
    print("RUNNING ALL 6 DISPERSION ANALYSIS CASES")
    print("="*80)
    
    all_results = {}
    
    for case_num, case_config in ANALYSIS_CASES.items():
        print(f"\nCASE {case_num}: {case_config['name']} - {case_config['wind']}")
        print("-" * 60)
        
        results = run_monte_carlo_simulation(case_config, num_simulations)
        all_results[case_num] = {
            'config': case_config,
            'results': results,
            'dispersion_ellipses': calculate_dispersion_ellipses(results['impact_points'])
        }
        
        # Print summary statistics
        if len(results['flight_data']) > 0:
            apogees = [f['apogee'] for f in results['flight_data']]
            flight_times = [f['flight_time'] for f in results['flight_data']]
            
            print(f"  Apogee: {np.mean(apogees):.0f} ± {np.std(apogees):.0f} m")
            print(f"  Flight Time: {np.mean(flight_times):.1f} ± {np.std(flight_times):.1f} s")
            print(f"  Impact Dispersion Area (1σ): {np.pi * all_results[case_num]['dispersion_ellipses'][0]['width'] * all_results[case_num]['dispersion_ellipses'][0]['height'] / 4:.0f} m²")
    
    return all_results


# VISUALIZATION FUNCTIONS

def create_dispersion_map(all_results):
    """Create overlaid dispersion maps for both wind conditions."""
    
    # Colors for different trajectories
    colors = {
        'Ballistic': '#FF6B6B',  # Red
        'Drogue Only': '#4ECDC4',  # Teal
        'Main at Apogee': '#45B7D1',  # Blue
    }
    
    # figure with two subplots (one for each wind condition)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
 
    wind_titles = {
        'high_winds_august': 'High Winds (20 mph ground + higher aloft)',
        'typical_august': 'Typical Mid-August Winds'
    }
    
    # Processing each wind condition
    for wind_idx, wind_type in enumerate(['high_winds_august', 'typical_august']):
        ax = axes[wind_idx]
        
        # Set up the map background
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('East Position (m)')
        ax.set_ylabel('North Position (m)')
        ax.set_title(f'{wind_titles[wind_type]}', fontsize=14, fontweight='bold')
        
        # Plot launch site
        ax.scatter(0, 0, color='black', s=200, marker='*', label='Launch Site', zorder=10)
        
        # Highways (simplified representation)
        # Highway 144 -> approx ->change later
        hwy_144_x = np.array([-2000, 2000])
        hwy_144_y = np.array([500, -500])  
        ax.plot(hwy_144_x, hwy_144_y, 'k-', linewidth=3, label='Hwy 144', alpha=0.7)
        ax.plot(hwy_144_x, hwy_144_y, 'y-', linewidth=1.5, alpha=0.7)
        
        # Highway 101 -> approx -> compute coordinates
        hwy_101_x = np.array([-1500, 1500])
        hwy_101_y = np.array([1000, 1000])  # Roughly E-W orientation
        ax.plot(hwy_101_x, hwy_101_y, 'k-', linewidth=3, label='Hwy 101', alpha=0.7)
        ax.plot(hwy_101_x, hwy_101_y, 'y-', linewidth=1.5, alpha=0.7)
        
        # Plot dispersion for each type of trajectory
        for case_num, case_data in all_results.items():
            if case_data['config']['wind'] != wind_type:
                continue
            
            trajectory_type = case_data['config']['name']
            color = colors[trajectory_type]
            
            # individual impact points plot
            points = case_data['results']['impact_points']
            if len(points) > 0:
                ax.scatter(points[:, 0], points[:, 1], 
                          color=color, s=10, alpha=0.3, 
                          label=f'{trajectory_type} Impacts')
            
            # dispersion ellipses plot
            ellipses = case_data['dispersion_ellipses']
            for ellipse in ellipses:
                if ellipse['sigma'] == 1:
                    alpha = 0.3
                    linestyle = '-'
                else:
                    alpha = 0.15
                    linestyle = '--'
                
                ell_patch = Ellipse(
                    xy=ellipse['mean'],
                    width=ellipse['width'],
                    height=ellipse['height'],
                    angle=ellipse['angle'],
                    facecolor=color,
                    edgecolor=color,
                    alpha=alpha,
                    linestyle=linestyle,
                    linewidth=1.5,
                )
                ax.add_patch(ell_patch)
        
        # plot limits
        ax.set_xlim(-3000, 3000)
        ax.set_ylim(-2000, 2000)
        
        # legend
        ax.legend(loc='upper right', fontsize=10)
        
        # grid and background
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#f5f5f5')
    
    # title
    plt.suptitle('Perige-Gee Rocket Dispersion Analysis\nComparison of Wind Conditions', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('perige_gee_dispersion_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def create_individual_case_plots(all_results):
    """Create individual plots for each case."""
    
    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    
    # 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (case_num, case_data) in enumerate(all_results.items()):
        ax = axes[idx]
        
        config = case_data['config']
        points = case_data['results']['impact_points']
        ellipses = case_data['dispersion_ellipses']
        
        # impact points plot
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], 
                      color=colors[idx], s=20, alpha=0.6)
        
        # dispersion ellipses plot
        for ellipse in ellipses:
            if ellipse['sigma'] == 1:
                alpha = 0.3
                label = f"1σ ({ellipse['width']/2:.0f}m x {ellipse['height']/2:.0f}m)"
            else:
                alpha = 0.15
                label = f"2σ ({ellipse['width']/2:.0f}m x {ellipse['height']/2:.0f}m)"
            
            ell_patch = Ellipse(
                xy=ellipse['mean'],
                width=ellipse['width'],
                height=ellipse['height'],
                angle=ellipse['angle'],
                facecolor=colors[idx],
                edgecolor=colors[idx],
                alpha=alpha,
                linestyle='--' if ellipse['sigma'] == 2 else '-',
            )
            ax.add_patch(ell_patch)
        
        # launch site plot
        ax.scatter(0, 0, color='red', s=100, marker='*', zorder=10)
        
        # title and labels
        wind_label = "High Winds" if config['wind'] == 'high_winds_august' else "Typical Winds"
        title = f"Case {case_num}: {config['name']}\n{wind_label}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # limits
        if len(points) > 0:
            x_center = np.mean(points[:, 0])
            y_center = np.mean(points[:, 1])
            max_range = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 500)
            ax.set_xlim(x_center - max_range, x_center + max_range)
            ax.set_ylim(y_center - max_range, y_center + max_range)
        else:
            ax.set_xlim(-1000, 1000)
            ax.set_ylim(-1000, 1000)
    
    plt.suptitle('Perige-Gee: Individual Case Dispersion Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('perige_gee_individual_cases.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_statistical_summary(all_results):
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    summary_data = []
    
    for case_num, case_data in all_results.items():
        config = case_data['config']
        flight_data = case_data['results']['flight_data']
        
        if len(flight_data) == 0:
            continue
        
        #statistics extraction
        apogees = [f['apogee'] for f in flight_data]
        max_velocities = [f['max_velocity'] for f in flight_data]
        impact_velocities = [f['impact_velocity'] for f in flight_data]
        flight_times = [f['flight_time'] for f in flight_data]
        
        # dispersion statistics calculations
        points = case_data['results']['impact_points']
        if len(points) > 1:
            mean_impact = np.mean(points, axis=0)
            std_impact = np.std(points, axis=0)
            max_distance = np.max(np.sqrt(points[:, 0]**2 + points[:, 1]**2))
            cep = 0.589 * (std_impact[0] + std_impact[1])  # Circular Error Probable
        else:
            mean_impact = (0, 0)
            std_impact = (0, 0)
            max_distance = 0
            cep = 0
        
        summary_data.append({
            'Case': case_num,
            'Configuration': config['name'],
            'Wind Condition': 'High' if config['wind'] == 'high_winds_august' else 'Typical',
            'Success Rate': f"{case_data['results']['success_rate']*100:.1f}%",
            'Mean Apogee (m)': f"{np.mean(apogees):.0f}",
            'Std Apogee (m)': f"{np.std(apogees):.0f}",
            'Mean Max Velocity (m/s)': f"{np.mean(max_velocities):.1f}",
            'Mean Impact Velocity (m/s)': f"{np.mean(impact_velocities):.1f}",
            'Mean Flight Time (s)': f"{np.mean(flight_times):.1f}",
            'Mean Impact East (m)': f"{mean_impact[0]:.0f}",
            'Mean Impact North (m)': f"{mean_impact[1]:.0f}",
            'Std Impact East (m)': f"{std_impact[0]:.0f}",
            'Std Impact North (m)': f"{std_impact[1]:.0f}",
            'Max Range (m)': f"{max_distance:.0f}",
            'CEP (m)': f"{cep:.0f}",
        })
    
    # DataFrame and display
    df_summary = pd.DataFrame(summary_data)
    print("\nSummary Statistics:")
    print("-" * 120)
    print(df_summary.to_string(index=False))
    print("-" * 120)
    
    # Save to CSV
    df_summary.to_csv('perige_gee_summary_statistics.csv', index=False)
    print(f"\nSummary saved to 'perige_gee_summary_statistics.csv'")
    
    # comparative bar plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Apogee comparison
    case_labels = [f"Case {row['Case']}" for _, row in df_summary.iterrows()]
    apogee_means = [float(row['Mean Apogee (m)']) for _, row in df_summary.iterrows()]
    apogee_stds = [float(row['Std Apogee (m)']) for _, row in df_summary.iterrows()]
    
    bars1 = axes[0, 0].bar(case_labels, apogee_means, yerr=apogee_stds, 
                          capsize=5, color=plt.cm.Set2(np.arange(len(case_labels))))
    axes[0, 0].set_ylabel('Apogee (m)')
    axes[0, 0].set_title('Apogee Altitude Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Impact velocity comparison
    impact_vels = [float(row['Mean Impact Velocity (m/s)']) for _, row in df_summary.iterrows()]
    bars2 = axes[0, 1].bar(case_labels, impact_vels, color=plt.cm.Set2(np.arange(len(case_labels))))
    axes[0, 1].set_ylabel('Impact Velocity (m/s)')
    axes[0, 1].set_title('Impact Velocity Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Flight time comparison
    flight_times = [float(row['Mean Flight Time (s)']) for _, row in df_summary.iterrows()]
    bars3 = axes[1, 0].bar(case_labels, flight_times, color=plt.cm.Set2(np.arange(len(case_labels))))
    axes[1, 0].set_ylabel('Flight Time (s)')
    axes[1, 0].set_title('Flight Duration Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Dispersion (CEP) comparison
    cep_values = [float(row['CEP (m)']) for _, row in df_summary.iterrows()]
    bars4 = axes[1, 1].bar(case_labels, cep_values, color=plt.cm.Set2(np.arange(len(case_labels))))
    axes[1, 1].set_ylabel('CEP (m)')
    axes[1, 1].set_title('Dispersion (Circular Error Probable)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Perige-Gee Performance Comparison Across All Cases', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('perige_gee_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return df_summary


# main execution
def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("PERIGE-GEE ROCKET DISPERSION ANALYSIS")
    print("="*80)
    print("\nAnalysis Cases:")
    print("-" * 40)
    for case_num, config in ANALYSIS_CASES.items():
        wind_desc = "High Winds (20 mph ground)" if config['wind'] == 'high_winds_august' else "Typical August Winds"
        print(f"Case {case_num}: {config['name']} - {wind_desc}")
    
    print("\n" + "="*80)
    
    # Creating output directory
    os.makedirs('dispersion_outputs', exist_ok=True)
    
    print("\nStarting Monte Carlo simulations...")
    print("Note: This may take several minutes depending on number of simulations.")
    
    
    NUM_SIMULATIONS = 50
    
    all_results = run_all_dispersion_analyses(num_simulations=NUM_SIMULATIONS)
    
    # visualizations
    print("\nGenerating visualizations...")
    
    # 1. Overlaid dispersion maps
    print("1. Creating overlaid dispersion maps...")
    fig1 = create_dispersion_map(all_results)
    
    # 2. Individual case plots
    print("2. Creating individual case plots...")
    create_individual_case_plots(all_results)
    
    # 3. Statistical summary
    print("3. Generating statistical summary...")
    df_summary = create_statistical_summary(all_results)
    
    # 4. Create wind profile comparison
    print("4. Creating wind profile comparison...")
    create_wind_profile_comparison()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nOutput files created:")
    print("  - perige_gee_dispersion_comparison.png")
    print("  - perige_gee_individual_cases.png")
    print("  - perige_gee_performance_comparison.png")
    print("  - perige_gee_summary_statistics.csv")
    print("  - wind_profile_comparison.png")
    print("\nAll files saved in current directory.")
    
    return all_results, df_summary

def create_wind_profile_comparison():
    """Create wind profile comparison plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    altitudes = np.linspace(0, 10000, 100)
    
    # Plot for high winds
    wind_speeds_high = []
    wind_directions_high = []
    
    for alt in altitudes:
        _, _, speed, direction = create_wind_profile('high_winds_august', alt)
        wind_speeds_high.append(speed)
        wind_directions_high.append(direction)
    
    ax1.plot(wind_speeds_high, altitudes, 'r-', linewidth=2, label='Wind Speed')
    ax1.set_xlabel('Wind Speed (m/s)')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('High Winds Profile (20 mph ground)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(wind_directions_high, altitudes, 'b--', linewidth=2, label='Wind Direction')
    ax1_twin.set_ylabel('Direction (degrees)')
    ax1_twin.set_ylim(0, 360)
    ax1_twin.legend(loc='upper right')
    
    # Plot for typical winds
    wind_speeds_typical = []
    wind_directions_typical = []
    
    for alt in altitudes:
        _, _, speed, direction = create_wind_profile('typical_august', alt)
        wind_speeds_typical.append(speed)
        wind_directions_typical.append(direction)
    
    ax2.plot(wind_speeds_typical, altitudes, 'r-', linewidth=2, label='Wind Speed')
    ax2.set_xlabel('Wind Speed (m/s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Typical August Winds Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(wind_directions_typical, altitudes, 'b--', linewidth=2, label='Wind Direction')
    ax2_twin.set_ylabel('Direction (degrees)')
    ax2_twin.set_ylim(0, 360)
    ax2_twin.legend(loc='upper right')
    
    plt.suptitle('Wind Profile Comparison for Perige-Gee Launch Conditions', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('wind_profile_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

#Main Analysis
if __name__ == "__main__":
    TEST_MODE = True  # Set to False for full analysis
    
    if TEST_MODE : 
        print("\n" + "="*80)
        print("TEST MODE: Using reduced simulation count")
        print("Set TEST_MODE = False for full analysis")
        print("="*80)
        NUM_SIMULATIONS = 10
    else:
        NUM_SIMULATIONS = 50
    
    # Run  analysis
    try:
        results, summary = main()
        
        # Save raw results for further analysis
        import pickle
        with open('perige_gee_results.pkl', 'wb') as f:
            pickle.dump({'results': results, 'summary': summary}, f)
        
        print("\nRaw results saved to 'perige_gee_results.pkl'")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have the required data files:")
        print("   - motors/perige_gee_thrust.csv")
        print("   - aerodynamics/cd_power_off.csv")
        print("   - aerodynamics/cd_power_on.csv")
        print("2. Install required packages: pip install rocketpy numpy matplotlib pandas cartopy")
        print("3. Check that the rocket parameters match your actual Perige-Gee design")