import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.io.img_tiles as c_tiles
import cartopy.feature as cfeature
import pandas as pd
import os
import sys
import random
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime

# Import RocketPy components
from rocketpy import Environment, Flight, Function, Rocket, SolidMotor

winddatapath = r"C:\flight-sim\Dispersion Analysis\winddata.csv"

print("="*80)
print("PERIGE-GEE ROCKET DISPERSION ANALYSIS")
print("6 CASES WITH SPECIFIED WIND CONDITIONS")
print("="*80)

PERIGE_GEE_PARAMS = {
    'name': 'Perige-Gee',
    'mass': 12.863,  # kg (dry)
    'radius': 39.6215/1000,  # m
    'inertia': (0.85, 0.85, 0.02),  
    'com_without_motor': 0.62,
    'motor_impulse': 1415.15,  # N-s
    'burn_time': 5.274,  # s
    'cd_s_drogue': 1.4,  # m²
    'cd_s_main': 2.2,  # m² 
    'inclination': 84.7,  # degrees
    'heading': 53,  # degrees from north
    'rail_length': 5.2,
}

LAUNCH_SITE = {
    'name': 'Timmins',
    'latitude': 48.4669,
    'longitude': -81.333,
    'elevation': 305,  # m
    'utm_zone': '17T',
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

# Global wind profile pools
high_pool = None
typical_pool = None

# Color scheme for cases
CASE_COLORS = {
    'Ballistic': {'color': 'red', 'linestyle': '-', 'label': 'Ballistic'},
    'Drogue Only': {'color': 'lime', 'linestyle': '--', 'label': 'Drogue Only'},
    'Main at Apogee': {'color': 'cyan', 'linestyle': ':', 'label': 'Main at Apogee'}
}

def check_required_files():
    required_files = [
        'motors/perige_gee_thrust.csv',
        'aerodynamics/cd_power_off.csv', 
        'aerodynamics/cd_power_on.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("\n❌ ERROR: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure these files are in the correct directories.")
        return False
    
    print("✓ All required files found")
    return True

def load_and_filter_wind_data(filepath):    
   
    print(f"Loading wind data from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Wind data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} rows, {df['day'].nunique()} days")

    # Rename columns for consistency
    for col in df.columns:
        low_col = col.lower()
        if 'wind' in low_col and 'speed' in low_col:
            df = df.rename(columns={col: 'wind_speed'})
        elif 'alt' in low_col or 'height' in low_col:
            df = df.rename(columns={col: 'altitude_ft'})
        elif 'dir' in low_col or 'deg' in low_col:
            df = df.rename(columns={col: 'wind_direction'})
        elif 'day' in low_col or 'date' in low_col:
            df = df.rename(columns={col: 'day'})

    # Convert units
    df['wind_speed_ms'] = df['wind_speed'] * 0.514444  # knots to m/s
    df['altitude_m'] = df['altitude_ft'] * 0.3048
    
    # Group by day and create profiles
    profiles = [group.sort_values('altitude_ft') for _, group in df.groupby('day')]
    
    high_wind_days = []
    typical_days = []

    for p in profiles:
        # Use lowest altitude for ground wind (usually ~3000ft in sounding data)
        if 3000 in p['altitude_ft'].values:
            ground_wind = p[p['altitude_ft'] == 3000]['wind_speed_ms'].values[0]
        else:
            # Use minimum altitude if 3000ft not available
            min_alt = p['altitude_ft'].min()
            ground_wind = p[p['altitude_ft'] == min_alt]['wind_speed_ms'].values[0]

        # Categorize based on ground wind speed
        if 8.0 <= ground_wind <= 10.0:  # 8-10 m/s = ~16-20 knots
            high_wind_days.append(p)
        else: 
            typical_days.append(p)
    
    print(f"  High wind days: {len(high_wind_days)}")
    print(f"  Typical days: {len(typical_days)}")
    
    return high_wind_days, typical_days

def create_wind_profile(selected_day_df, altitude_m):
    """Create wind profile at specified altitude from selected day data."""
    # Convert altitude to feet for interpolation (since data is in feet)
    altitude_ft = altitude_m * 3.28084
    
    alts = selected_day_df['altitude_ft'].values
    speeds = selected_day_df['wind_speed_ms'].values  # Already in m/s
    directions = selected_day_df['wind_direction'].values
    
    # Interpolate wind components
    wind_speed = np.interp(altitude_ft, alts, speeds)
    wind_dir = np.interp(altitude_ft, alts, directions)
    
    # Convert to U/V components (meteorological convention: direction from)
    rads = np.radians(270 - wind_dir)  # Convert to math coordinates
    u = wind_speed * np.cos(rads)
    v = wind_speed * np.sin(rads)
    
    return u, v, wind_speed, wind_dir

def create_custom_environment(selected_day_df):
  
    env = Environment(
        latitude=LAUNCH_SITE['latitude'],
        longitude=LAUNCH_SITE['longitude'],
        elevation=LAUNCH_SITE['elevation'],
        datum="WGS84",
    )
    
    # Define altitude range for wind profile (0 to max altitude in data)
    max_alt_m = selected_day_df['altitude_m'].max()
    altitudes = np.linspace(0, max_alt_m, 100)
    
    # Create wind profiles
    wind_u = []
    wind_v = []
    
    for alt in altitudes:
        u, v = create_wind_profile(selected_day_df, alt)
        wind_u.append(u)
        wind_v.append(v)
    
    # Create RocketPy wind functions
    wind_u_data = np.column_stack((altitudes, wind_u))
    wind_v_data = np.column_stack((altitudes, wind_v))
    
    env.wind_velocity_x = Function(wind_u_data, inputs='Height (m)', outputs='Wind U (m/s)')
    env.wind_velocity_y = Function(wind_v_data, inputs='Height (m)', outputs='Wind V (m/s)')
    
    return env

def create_perige_gee_rocket(rocket_params, drogue=True, main=True, main_at_apogee=False):
    """Create Perige-Gee rocket with specified parachute configuration."""
    
    # Create motor
    motor = SolidMotor(
        thrust_source="motors/perige_gee_thrust.csv",
        dry_mass=3.108,
        dry_inertia=(0.397, 0.004, 0.397),
        center_of_dry_mass_position=0.512,
        grains_center_of_mass_position=0.563,
        burn_time=5.274,
        grain_number=6,
        grain_density=1560,
        grain_outer_radius=0.035,
        grain_initial_inner_radius=0.01,
        grain_initial_height=0.155,
        grain_separation=0.005,
        nozzle_radius=0.026,
        throat_radius=0.0076,
        interpolation_method="linear",
    )
    
    # Create rocket
    rocket = Rocket(
        radius=rocket_params['radius'],
        mass=rocket_params['mass'] + 3.108,  # Include motor mass
        inertia=rocket_params['inertia'],
        power_off_drag="aerodynamics/cd_power_off.csv",
        power_on_drag="aerodynamics/cd_power_on.csv",
        center_of_mass_without_motor=rocket_params['com_without_motor'],
        coordinate_system_orientation="tail_to_nose",
    )
    
    # Add motor
    rocket.add_motor(motor, position=0)
    
    # Add nose
    rocket.add_nose(
        length=0.5334,  
        kind="vonKarman",
        position=1.64,
    )
    
    # Add fins
    rocket.add_trapezoidal_fins(
        n=4,
        span=0.0762,  # semi-span
        root_chord=0.254,
        tip_chord=0.0508,
        position=-0.101,
        cant_angle=0,
    )
    
    # Add rail buttons
    rocket.set_rail_buttons(
        upper_button_position=0.1,
        lower_button_position=0.8,
    )
    
    if drogue:
        # Create a closure to track the maximum altitude
        max_altitude = [0]
        
        def drogue_trigger(p, h, y):
            """Trigger drogue at apogee (when altitude starts decreasing)."""
            # Update maximum altitude
            if h > max_altitude[0]:
                max_altitude[0] = h
                return False
            
            # Trigger when we're 50m below apogee
            return h < max_altitude[0] - 50
        
        rocket.add_parachute(
            "Drogue",
            cd_s=rocket_params['cd_s_drogue'],
            trigger=drogue_trigger,
            sampling_rate=105,
            lag=1.0,
            noise=(0, 8.3, 0.5),
        )
    
    if main:
        if main_at_apogee:
            # For main at apogee, use same logic as drogue
            max_altitude = [0]
            
            def main_trigger(p, h, y):
                """Trigger main at apogee."""
                if h > max_altitude[0]:
                    max_altitude[0] = h
                    return False
                
                # Trigger when we're 100m below apogee (slightly later than drogue)
                return h < max_altitude[0] - 100
        else:
            def main_trigger(p, h, y):
                """Trigger main at specified altitude."""
                return h <= 295  # Deploy at 295m (slightly before 300m)
            
        rocket.add_parachute(
            "Main",
            cd_s=rocket_params['cd_s_main'],
            trigger=main_trigger,
            sampling_rate=105,
            lag=1.5,
            noise=(0, 8.3, 0.5),
        )
    
    return rocket

def test_single_flight():
    """Test a single flight simulation to verify setup."""
    global high_pool, typical_pool
    
    print("\n" + "="*60)
    print("TESTING SINGLE SIMULATION")
    print("="*60)
    
    if high_pool is None or typical_pool is None:
        print("Loading wind data...")
        high_pool, typical_pool = load_and_filter_wind_data(winddatapath)
    
    # Pick a random day from high wind pool
    if len(high_pool) > 0:
        test_day = random.choice(high_pool)
        print(f"Testing with day {int(test_day['day'].iloc[0])}")
    else:
        print("No high wind days found, using typical pool")
        test_day = random.choice(typical_pool)
    
    try:
        env = create_custom_environment(test_day)
        print("✓ Environment created successfully")
        
        # First test ballistic (no parachutes)
        print("\nTesting ballistic flight (no parachutes)...")
        rocket_ballistic = create_perige_gee_rocket(
            PERIGE_GEE_PARAMS,
            drogue=False,
            main=False,
            main_at_apogee=False
        )
        print("✓ Ballistic rocket created successfully")
        
        flight_ballistic = Flight(
            rocket=rocket_ballistic,
            environment=env,
            rail_length=PERIGE_GEE_PARAMS['rail_length'],
            inclination=PERIGE_GEE_PARAMS['inclination'],
            heading=PERIGE_GEE_PARAMS['heading'],
            verbose=False,
            terminate_on_apogee=False
        )
        print("✓ Ballistic flight simulation successful")
        print(f"  Apogee: {flight_ballistic.apogee:.2f} m")
        print(f"  Impact: X={flight_ballistic.x_impact:.2f}m, Y={flight_ballistic.y_impact:.2f}m")
        print(f"  Flight time: {flight_ballistic.t_final:.1f} s")
        
        # Now test with drogue parachute
        print("\nTesting flight with drogue parachute...")
        rocket_drogue = create_perige_gee_rocket(
            PERIGE_GEE_PARAMS,
            drogue=True,
            main=False,
            main_at_apogee=False
        )
        print("✓ Rocket with drogue parachute created")
        
        flight_drogue = Flight(
            rocket=rocket_drogue,
            environment=env,
            rail_length=PERIGE_GEE_PARAMS['rail_length'],
            inclination=PERIGE_GEE_PARAMS['inclination'],
            heading=PERIGE_GEE_PARAMS['heading'],
            verbose=False,
            terminate_on_apogee=False
        )
        print("✓ Flight with drogue simulation successful")
        print(f"  Apogee: {flight_drogue.apogee:.2f} m")
        print(f"  Impact: X={flight_drogue.x_impact:.2f}m, Y={flight_drogue.y_impact:.2f}m")
        print(f"  Flight time: {flight_drogue.t_final:.1f} s")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_monte_carlo_simulation(case_id, num_simulations):
    """Run Monte Carlo simulation for a specific case."""
    global high_pool, typical_pool
    
    case_config = ANALYSIS_CASES[case_id]
    wind_type = case_config['wind']
    
    # Select wind pool based on case
    profile_pool = high_pool if wind_type == 'high_winds_august' else typical_pool
    
    print(f"\n>>> Starting Case {case_id}: {case_config['name']} ({wind_type})")
    
    impact_points = []
    flight_data = []
    successful_sims = 0
    
    for i in range(num_simulations):
        try:
            print(f"  Simulation {i+1}/{num_simulations}...", end='\r')
            
            selected_day_df = random.choice(profile_pool)
            env = create_custom_environment(selected_day_df)
            
            is_apogee_main = (case_config['name'] == 'Main at Apogee')
            
            rocket = create_perige_gee_rocket(
                rocket_params=PERIGE_GEE_PARAMS,
                drogue=case_config['drogue'],
                main=case_config['main'],
                main_at_apogee=is_apogee_main
            )
            
            # Create and run flight
            flight = Flight(
                rocket=rocket,
                environment=env,
                rail_length=PERIGE_GEE_PARAMS['rail_length'],
                inclination=PERIGE_GEE_PARAMS['inclination'],
                heading=PERIGE_GEE_PARAMS['heading'],
                verbose=False,
                terminate_on_apogee=False,
            )
            
            # Store results
            impact_points.append((flight.x_impact, flight.y_impact))
            
            flight_data.append({
                'apogee': flight.apogee,
                'apogee_time': flight.apogee_time,
                'max_velocity': flight.max_speed,
                'impact_velocity': flight.impact_velocity,
                'flight_time': flight.t_final,
                'impact_x': flight.x_impact,
                'impact_y': flight.y_impact,
            })
            successful_sims += 1
            
        except Exception as e:
            print(f"    ! Simulation {i+1} failed: {e}")
            continue
    
    print(f"\n  Successfully completed {successful_sims}/{num_simulations} simulations")
    
    # Calculate sigma statistics if we have enough data
    sigma_stats = {}
    if len(impact_points) >= 5:
        points_array = np.array(impact_points)
        mean = np.mean(points_array, axis=0)
        cov = np.cov(points_array.T)  # covariance matrix
        
        # Calculate eigenvalues for dispersion
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvalues to find Major and Minor axes
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        sigma_stats = {
            'mean_impact': mean.tolist(),
            '1sigma_major': 1 * np.sqrt(eigenvalues[0]),
            '1sigma_minor': 1 * np.sqrt(eigenvalues[1]),
            '2sigma_major': 2 * np.sqrt(eigenvalues[0]),
            '2sigma_minor': 2 * np.sqrt(eigenvalues[1]),
            'covariance_angle': np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        }
    
    return {
        'impact_points': np.array(impact_points) if impact_points else np.array([]),
        'flight_data': flight_data,
        'success_rate': successful_sims / num_simulations if num_simulations > 0 else 0,
        'sigma_stats': sigma_stats,
    }

def create_comprehensive_visualization(all_results):
    """Create comprehensive visualization of all results."""
    # Filter cases that have data
    cases_with_data = {k: v for k, v in all_results.items() 
                       if len(v['results']['impact_points']) > 0}
    
    if not cases_with_data:
        print("No data available for visualization")
        return
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Main dispersion plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title('Impact Points Dispersion', fontsize=12, fontweight='bold')
    
    for case_id, res in cases_with_data.items():
        config = ANALYSIS_CASES[case_id]
        points = res['results']['impact_points']
        color = CASE_COLORS[config['name']]['color']
        
        ax1.scatter(points[:, 0], points[:, 1], alpha=0.5, s=20, 
                   color=color, label=config['name'])
        
        # Add mean point
        if res['results']['sigma_stats']:
            mean = res['results']['sigma_stats']['mean_impact']
            ax1.plot(mean[0], mean[1], 'X', color=color, markersize=10, markeredgewidth=2)
    
    ax1.set_xlabel('East-West Displacement (m)')
    ax1.set_ylabel('North-South Displacement (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.axis('equal')
    
    # 2. Dispersion ellipses comparison
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title('2σ Dispersion Ellipses', fontsize=12, fontweight='bold')
    
    for case_id, res in cases_with_data.items():
        config = ANALYSIS_CASES[case_id]
        style = CASE_COLORS[config['name']]
        
        if res['results']['sigma_stats']:
            stats = res['results']['sigma_stats']
            ellipse = Ellipse(
                xy=stats['mean_impact'],
                width=stats['2sigma_major'] * 2,
                height=stats['2sigma_minor'] * 2,
                angle=stats['covariance_angle'],
                edgecolor=style['color'],
                facecolor=style['color'],
                alpha=0.3,
                linestyle=style['linestyle'],
                linewidth=2,
                label=f"{config['name']} ({config['wind']})"
            )
            ax2.add_patch(ellipse)
    
    ax2.set_xlabel('East-West (m)')
    ax2.set_ylabel('North-South (m)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=8)
    ax2.axis('equal')
    
    # 3. Performance metrics comparison
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    
    metrics_data = []
    case_labels = []
    
    for case_id, res in cases_with_data.items():
        config = ANALYSIS_CASES[case_id]
        case_labels.append(f"C{case_id}\n{config['name'][:8]}")
        
        if res['results']['flight_data']:
            flight_data = res['results']['flight_data']
            metrics_data.append({
                'Apogee': np.mean([f['apogee'] for f in flight_data]),
                'Flight Time': np.mean([f['flight_time'] for f in flight_data]),
                'Max Velocity': np.mean([f['max_velocity'] for f in flight_data]),
            })
    
    # Convert to arrays for plotting
    x = np.arange(len(case_labels))
    width = 0.25
    
    if metrics_data:
        apogees = [m['Apogee']/1000 for m in metrics_data]  # Convert to km
        flight_times = [m['Flight Time'] for m in metrics_data]
        max_velocities = [m['Max Velocity'] for m in metrics_data]
        
        bars1 = ax3.bar(x - width, apogees, width, label='Apogee (km)', 
                       color='skyblue', alpha=0.7)
        bars2 = ax3.bar(x, flight_times, width, label='Flight Time (s)',
                       color='lightgreen', alpha=0.7)
        bars3 = ax3.bar(x + width, max_velocities, width, label='Max Velocity (m/s)',
                       color='salmon', alpha=0.7)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(case_labels)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Dispersion statistics
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title('Dispersion Statistics (2σ)', fontsize=12, fontweight='bold')
    
    dispersion_data = []
    for case_id, res in cases_with_data.items():
        config = ANALYSIS_CASES[case_id]
        if res['results']['sigma_stats']:
            stats = res['results']['sigma_stats']
            dispersion_data.append({
                'Major Axis': stats['2sigma_major'],
                'Minor Axis': stats['2sigma_minor'],
                'Area': np.pi * stats['2sigma_major'] * stats['2sigma_minor'] / 1e6,  # km²
            })
    
    if dispersion_data:
        x = np.arange(len(case_labels))
        width = 0.25
        
        major_axes = [d['Major Axis']/1000 for d in dispersion_data]  # Convert to km
        minor_axes = [d['Minor Axis']/1000 for d in dispersion_data]
        areas = [d['Area'] for d in dispersion_data]
        
        bars1 = ax4.bar(x - width, major_axes, width, label='Major Axis (km)',
                       color='orange', alpha=0.7)
        bars2 = ax4.bar(x, minor_axes, width, label='Minor Axis (km)',
                       color='purple', alpha=0.7)
        bars3 = ax4.bar(x + width, areas, width, label='Area (km²)',
                       color='brown', alpha=0.7)
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(case_labels)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Success rates
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title('Simulation Success Rates', fontsize=12, fontweight='bold')
    
    success_rates = []
    colors = []
    for case_id, res in cases_with_data.items():
        config = ANALYSIS_CASES[case_id]
        success_rates.append(res['results']['success_rate'] * 100)
        colors.append(CASE_COLORS[config['name']]['color'])
    
    bars = ax5.bar(range(len(case_labels)), success_rates, color=colors, alpha=0.7)
    ax5.set_xticks(range(len(case_labels)))
    ax5.set_xticklabels(case_labels)
    ax5.set_ylabel('Success Rate (%)')
    ax5.set_ylim(0, 105)
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10)
    
    # 6. Wind condition comparison
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title('Wind Condition Effect', fontsize=12, fontweight='bold')
    
    # Group by configuration type
    config_types = {}
    for case_id, res in cases_with_data.items():
        config = ANALYSIS_CASES[case_id]
        config_name = config['name']
        wind_type = config['wind']
        
        if config_name not in config_types:
            config_types[config_name] = {'high': None, 'typical': None}
        
        if res['results']['sigma_stats']:
            stats = res['results']['sigma_stats']
            dispersion_radius = np.sqrt(stats['2sigma_major'] * stats['2sigma_minor'])
            
            if wind_type == 'high_winds_august':
                config_types[config_name]['high'] = dispersion_radius
            else:
                config_types[config_name]['typical'] = dispersion_radius
    
    # Plot comparison
    x = np.arange(len(config_types))
    width = 0.35
    
    high_values = []
    typical_values = []
    config_names = []
    
    for config_name, values in config_types.items():
        config_names.append(config_name)
        high_values.append(values['high']/1000 if values['high'] else 0)  # Convert to km
        typical_values.append(values['typical']/1000 if values['typical'] else 0)
    
    bars1 = ax6.bar(x - width/2, high_values, width, label='High Winds', 
                   color='red', alpha=0.7)
    bars2 = ax6.bar(x + width/2, typical_values, width, label='Typical Winds',
                   color='blue', alpha=0.7)
    
    ax6.set_xticks(x)
    ax6.set_xticklabels(config_names)
    ax6.set_ylabel('Dispersion Radius (km)')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Perige-Gee Rocket Dispersion Analysis - Complete Results', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('perige_gee_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_detailed_dispersion_map(all_results):
    """Create detailed dispersion map with all cases."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Filter cases with data
    cases_with_data = {k: v for k, v in all_results.items() 
                       if len(v['results']['impact_points']) > 0}
    
    # Plot for high winds
    ax1.set_title('High Wind Conditions', fontsize=14, fontweight='bold')
    
    high_wind_cases = {k: v for k, v in cases_with_data.items() 
                       if ANALYSIS_CASES[k]['wind'] == 'high_winds_august'}
    
    for case_id, res in high_wind_cases.items():
        config = ANALYSIS_CASES[case_id]
        style = CASE_COLORS[config['name']]
        
        # Plot impact points
        points = res['results']['impact_points']
        ax1.scatter(points[:, 0]/1000, points[:, 1]/1000,  # Convert to km
                   alpha=0.3, s=10, color=style['color'])
        
        # Plot dispersion ellipse
        if res['results']['sigma_stats']:
            stats = res['results']['sigma_stats']
            ellipse = Ellipse(
                xy=(stats['mean_impact'][0]/1000, stats['mean_impact'][1]/1000),
                width=stats['2sigma_major'] * 2 / 1000,
                height=stats['2sigma_minor'] * 2 / 1000,
                angle=stats['covariance_angle'],
                edgecolor=style['color'],
                facecolor=style['color'],
                alpha=0.2,
                linestyle=style['linestyle'],
                linewidth=2,
                label=f"{config['name']}"
            )
            ax1.add_patch(ellipse)
    
    ax1.set_xlabel('East-West Displacement (km)')
    ax1.set_ylabel('North-South Displacement (km)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.axis('equal')
    ax1.set_aspect('equal', adjustable='datalim')
    
    # Plot for typical winds
    ax2.set_title('Typical Wind Conditions', fontsize=14, fontweight='bold')
    
    typical_wind_cases = {k: v for k, v in cases_with_data.items() 
                          if ANALYSIS_CASES[k]['wind'] == 'typical_august'}
    
    for case_id, res in typical_wind_cases.items():
        config = ANALYSIS_CASES[case_id]
        style = CASE_COLORS[config['name']]
        
        # Plot impact points
        points = res['results']['impact_points']
        ax2.scatter(points[:, 0]/1000, points[:, 1]/1000,  # Convert to km
                   alpha=0.3, s=10, color=style['color'])
        
        # Plot dispersion ellipse
        if res['results']['sigma_stats']:
            stats = res['results']['sigma_stats']
            ellipse = Ellipse(
                xy=(stats['mean_impact'][0]/1000, stats['mean_impact'][1]/1000),
                width=stats['2sigma_major'] * 2 / 1000,
                height=stats['2sigma_minor'] * 2 / 1000,
                angle=stats['covariance_angle'],
                edgecolor=style['color'],
                facecolor=style['color'],
                alpha=0.2,
                linestyle=style['linestyle'],
                linewidth=2,
                label=f"{config['name']}"
            )
            ax2.add_patch(ellipse)
    
    ax2.set_xlabel('East-West Displacement (km)')
    ax2.set_ylabel('North-South Displacement (km)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.axis('equal')
    ax2.set_aspect('equal', adjustable='datalim')
    
    plt.suptitle('Perige-Gee Rocket Dispersion Analysis by Wind Condition', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('perige_gee_dispersion_by_wind.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_statistical_summary(all_results):
    """Create statistical summary DataFrame."""
    summary_data = []
    
    for case_id, res in all_results.items():
        config = ANALYSIS_CASES[case_id]
        
        if res['results']['flight_data']:
            flight_data = res['results']['flight_data']
            
            # Calculate statistics
            apogees = [f['apogee'] for f in flight_data]
            flight_times = [f['flight_time'] for f in flight_data]
            max_velocities = [f['max_velocity'] for f in flight_data]
            impact_velocities = [f['impact_velocity'] for f in flight_data]
            
            # Calculate dispersion metrics if available
            if res['results']['sigma_stats']:
                stats = res['results']['sigma_stats']
                dispersion_area = np.pi * stats['2sigma_major'] * stats['2sigma_minor']
                dispersion_radius = np.sqrt(stats['2sigma_major'] * stats['2sigma_minor'])
                major_minor_ratio = stats['2sigma_major'] / stats['2sigma_minor'] if stats['2sigma_minor'] > 0 else 0
            else:
                dispersion_area = 0
                dispersion_radius = 0
                major_minor_ratio = 0
            
            summary_data.append({
                'Case': case_id,
                'Configuration': config['name'],
                'Wind Condition': config['wind'],
                'Simulations': len(flight_data),
                'Success Rate (%)': res['results']['success_rate'] * 100,
                'Mean Apogee (m)': np.mean(apogees),
                'Std Apogee (m)': np.std(apogees),
                'Mean Flight Time (s)': np.mean(flight_times),
                'Std Flight Time (s)': np.std(flight_times),
                'Mean Max Velocity (m/s)': np.mean(max_velocities),
                'Mean Impact Velocity (m/s)': np.mean(impact_velocities),
                '2σ Major Axis (m)': stats['2sigma_major'] if res['results']['sigma_stats'] else 0,
                '2σ Minor Axis (m)': stats['2sigma_minor'] if res['results']['sigma_stats'] else 0,
                '2σ Area (m²)': dispersion_area,
                '2σ Radius (m)': dispersion_radius,
                'Major/Minor Ratio': major_minor_ratio,
            })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save to CSV
    df_summary.to_csv('perige_gee_summary_statistics.csv', index=False)
    print("\nStatistical summary saved to 'perige_gee_summary_statistics.csv'")
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(df_summary.to_string())
    
    return df_summary

def main():
    """Main execution function."""
    global high_pool, typical_pool
    
    print("\n" + "="*80)
    print("PERIGE-GEE ROCKET DISPERSION ANALYSIS")
    print("="*80)
    
    # Check required files
    if not check_required_files():
        return None, None
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Load wind data
    print("\nLoading wind data...")
    high_pool, typical_pool = load_and_filter_wind_data(winddatapath)
    
    if len(high_pool) == 0 and len(typical_pool) == 0:
        print("❌ ERROR: No wind data loaded")
        return None, None
    
    # Test flight
    print("\n" + "="*80)
    print("FLIGHT TESTING")
    print("="*80)
    test_single_flight()
    
    # Run Monte Carlo simulations
    print("\n" + "="*80)
    print("MONTE CARLO SIMULATIONS")
    print("="*80)
    
    TEST_MODE = True  # Set to False for full analysis
    NUM_SIMULATIONS = 10 if TEST_MODE else 50
    
    if TEST_MODE:
        print(f"TEST MODE: Running {NUM_SIMULATIONS} simulations per case")
        print("Note: Set TEST_MODE = False for full 50-simulation analysis")
    else:
        print(f"FULL ANALYSIS MODE: Running {NUM_SIMULATIONS} simulations per case")
    
    all_results = {}
    
    # Run all 6 cases
    for case_num in ANALYSIS_CASES.keys():
        print(f"\n{'='*60}")
        print(f"PROCESSING CASE {case_num}")
        print('='*60)
        
        results = run_monte_carlo_simulation(case_num, NUM_SIMULATIONS)
        all_results[case_num] = {
            'config': ANALYSIS_CASES[case_num],
            'results': results
        }
    
    # Generate visualizations
    if all_results:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        print("1. Creating comprehensive analysis visualization...")
        create_comprehensive_visualization(all_results)
        
        print("2. Creating detailed dispersion maps...")
        create_detailed_dispersion_map(all_results)
        
        print("3. Generating statistical summary...")
        df_summary = create_statistical_summary(all_results)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nOutput files created:")
        print("  - perige_gee_comprehensive_analysis.png")
        print("  - perige_gee_dispersion_by_wind.png")
        print("  - perige_gee_summary_statistics.csv")
        
        return all_results, df_summary
    else:
        print("\n❌ No results generated")
        return None, None

if __name__ == "__main__":
    try:
        results, summary = main()
        
        # Save raw results for further analysis
        if results is not None:
            import pickle
            with open('perige_gee_results.pkl', 'wb') as f:
                pickle.dump({'results': results, 'summary': summary}, f)
            print("\n✓ Raw results saved to 'perige_gee_results.pkl'")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting tips:")
        print("1. If parachute triggers still fail, try using simpler altitude-based triggers.")
        print("2. Check RocketPy documentation for the correct trigger function signature.")
        print("3. Make sure all required data files exist.")
        print("4. Try running with verbose=True in Flight() to see more details.")