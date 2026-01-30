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



def load_and_filter_wind_data(filepath):    
    df = pd.read_csv(filepath)

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

    df['wind_speed_ms'] = df['wind_speed'] * 0.514444
    df['altitude_m'] = df['altitude_ft'] * 0.3048
    

    profiles = [group.sort_values('altitude_ft') for _, group in df.groupby('day')]

    high_wind_days = []
    typical_days = []

    for p in profiles : 
        ground_wind = p[p['altitude_ft'] == 3000]['wind_speed_ms'].values[0]

        if 8.0 <= ground_wind <=10.0:
            high_wind_days.append(p)
        else : 
            typical_days.append(p)
    return high_wind_days, typical_days


all_profiles = load_and_filter_wind_data(winddatapath)

def create_wind_profile(selected_day_df, altitude_m):
    alts = selected_day_df['altitude_ft'].values
    speeds = selected_day_df['wind_speed'].values
    directions = selected_day_df['wind_direction'].values

    rads = np.radians(directions)
    u_data = speeds * np.sin(rads)
    v_data = speeds * np.cos(rads)

    wind_u = np.interp(altitude_m, alts, u_data)
    wind_v = np.interp(altitude_m, alts, v_data)

    return wind_u, wind_v


def create_custom_environment(selected_day_df):
    
    """Create environment with custom wind profile."""
    env = Environment (
        date=(2026, 8, 15, 12),
        **LAUNCH_SITE
    )
    altitudes_m = selected_day_df['altitude_ft'].values/3.28084
    wind_u_list = []
    wind_v_list = []
    
    for alt in altitudes_m:
        u, v = create_wind_profile(selected_day_df, alt)
        wind_u_list.append(u)
        wind_v_list.append(v)
    
    # rocketpy wind functions format
    wind_u_data = np.column_stack((altitudes_m, wind_u_list))
    wind_v_data = np.column_stack((altitudes_m, wind_v_list))
    
    env.wind_velocity_x = Function(wind_u_data, inputs = 'Height(m)', outputs = 'Wind U (m/s)')
    env.wind_velocity_y = Function(wind_v_data,inputs='Height (m)', outputs='Wind V (m/s)')
    
    return env

def create_perige_gee_rocket(rocket_params, drogue=True, main=True, main_at_apogee=False):
    
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
        radius=PERIGE_GEE_PARAMS['radius'],
        mass=PERIGE_GEE_PARAMS['mass'] + 3.108,  # Include motor mass
        inertia=PERIGE_GEE_PARAMS['inertia'],
        power_off_drag="aerodynamics/cd_power_off.csv",
        power_on_drag="aerodynamics/cd_power_on.csv",
        center_of_mass_without_motor=PERIGE_GEE_PARAMS['com_without_motor'],
        coordinate_system_orientation="tail_to_nose",
    )
    
    # Add motor
    rocket.add_motor(motor, position=0)
    
    
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
        upper_button_position=0.081,
        lower_button_position=-0.93,
    )
    
    # Add parachutes if specified - FIXED TRIGGER FUNCTIONS
    if drogue:
        def drogue_trigger(p, y, h):
          
            return y[5] < 0 
        
        rocket.add_parachute(
            "Drogue",
            cd_s=rocket_params['cd_s_drogue'],
            trigger=drogue_trigger,
            sampling_rate=105,
            lag=1.0,  # Increased lag to prevent conflicts
            noise=(0, 8.3, 0.5),
        )
    
    if main:
        if main_at_apogee : 
            def main_trigger(p,y,h):
                return y[5] < -2
        else : 
            def main_trigger(p, y, h):
                """Trigger main at specified altitude"""
                return h <= 295  # Deploy slightly before 300m
            
        rocket.add_parachute(
            "Main",
            cd_s=rocket_params['cd_s_main'],
            trigger=main_trigger,
            sampling_rate=105,
            lag=1.5,  # Increased lag to prevent conflicts
            noise=(0, 8.3, 0.5),
        )
    
    return rocket




def test_single_flight():
    print("\n" + "="*60)
    print("TESTING SINGLE SIMULATION")
    print("="*60)
    
    # Pick a random day from high wind pool
    test_day = random.choice(high_pool)
    print(f"Testing with day {int(test_day['day'].iloc[0])}")
    
    try:
        env = create_custom_environment(test_day)
        print("✓ Environment created successfully")
        
        rocket = create_perige_gee_rocket(
            PERIGE_GEE_PARAMS,
            drogue=True,
            main=False,
            main_at_apogee=False
        )
        print("✓ Rocket created successfully")
        
        flight = Flight(
            rocket=rocket,
            environment=env,
            rail_length=PERIGE_GEE_PARAMS['rail_length'],
            inclination=PERIGE_GEE_PARAMS['inclination'],
            heading=PERIGE_GEE_PARAMS['heading'],
            verbose=False,
            terminate_on_apogee=False
        )
        print("✓ Flight simulation successful")
        print(f"  Apogee: {flight.apogee:.2f} m")
        print(f"  Impact: X={flight.x_impact:.2f}m, Y={flight.y_impact:.2f}m")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_monte_carlo_simulation(case_id, num_simulations):
    
    case_config = ANALYSIS_CASES[case_id]
    wind_type = case_config['wind']
    
    # 2. Map the case wind string to our filtered CSV pools
    # This ensures Case 1-3 use High Winds and 4-6 use Typical Winds
    profile_pool = high_pool if wind_type == 'high_winds_august' else typical_pool
    
    print(f"\n>>> Starting Case {case_id}: {case_config['name']} ({wind_type})")

    successful_sims = 0
    impact_points = []
    flight_data = []
    # Lists to store results
    impact_points = []
    flight_data = []
    
    for i in range(num_simulations):
        try:
            print(f"  Simulation {i+1}/{num_simulations}...")
            
            selected_day_df = random.choice(profile_pool)
            env = create_custom_environment(selected_day_df)
     
            is_apogee_main = (case_config['name']== 'Main at Apogee')

            rocket = create_perige_gee_rocket(
                rocket_params=PERIGE_GEE_PARAMS,
                drogue=case_config['drogue'],
                main=case_config['main'],
                main_at_apogee = is_apogee_main
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
            
            if (i + 1) % 10 == 0:
                print(f"    Completed {i + 1}/{num_simulations} simulations...")
                
        except Exception as e:
            print(f"    ! Simulation {i+1} failed: {e}")
            continue
    
    print(f"\n  Successfully completed {successful_sims}/{num_simulations} simulations")
    
    # Calculate sigma statistics if we have enough data
    sigma_stats = {}
    if len(impact_points) >= 5:
        points_array = np.array(impact_points)
        mean = np.mean(points_array, axis=0)


        cov = np.cov(points_array.T)#covariance matrix
        
        # Calculate 1σ and 2σ
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        #sort eigenvalues to find Major and Minor axes
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

def calculate_dispersion_ellipses(points, sigmas=[1, 2]):
    """Calculate dispersion ellipses for given points."""
    if len(points) < 2:
        return []
    
    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    
    ellipses = []
    for sigma in sigmas:
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

def run_all_dispersion_analyses(num_simulations=10):
    """Run all 6 dispersion analysis cases."""
    print("\n" + "="*80)
    print("RUNNING ALL 6 DISPERSION ANALYSIS CASES")
    print("="*80)
    
    all_results = {}
    
    for case_num, case_config in ANALYSIS_CASES.items():
        print(f"\nCASE {case_num}: {case_config['name']} - {case_config['wind']}")
        print("-" * 60)
        
        results = run_monte_carlo_simulation(case_num, num_simulations)
        
        if len(results['impact_points']) > 0:
            dispersion_ellipses = calculate_dispersion_ellipses(results['impact_points'])
        else:
            dispersion_ellipses = []
        
        all_results[case_num] = {
            'config': case_config,
            'results': results,
            'dispersion_ellipses': dispersion_ellipses
        }
        
        # Print summary
        if len(results['flight_data']) > 0:
            apogees = [f['apogee'] for f in results['flight_data']]
            print(f"  Apogee: {np.mean(apogees):.0f} ± {np.std(apogees):.0f} m")
            if results['sigma_stats']:
                print(f"  1σ Dispersion: {results['sigma_stats']['1sigma_major']:.0f} x {results['sigma_stats']['1sigma_minor']:.0f} m")
            print(f"  Success rate: {results['success_rate']*100:.1f}%")
    
    return all_results


def plot_statistical_summary(all_case_results, wind_condition_key):

    
    tiler = c_tiles.GoogleTiles(style='satellite')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=tiler.crs)
    
   
    extent = [-81.38, -81.28, 48.43, 48.50] 
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(tiler, 13)

    #Add Infrastructure (Hwy 144 and 101)
    hwy_101 = np.array([[-81.40, 48.47], [-81.25, 48.48]])
    hwy_144 = np.array([[-81.34, 48.40], [-81.33, 48.55]])
    
    for hwy, name in [(hwy_101, 'Hwy 101'), (hwy_144, 'Hwy 144')]:
        ax.plot(hwy[:,0], hwy[:,1], color='yellow', linewidth=2, 
                transform=ccrs.PlateCarree(), path_effects=[pe.withStroke(linewidth=4, foreground="black")])
        ax.text(hwy[1,0], hwy[1,1], name, color='white', transform=ccrs.PlateCarree(), fontweight='bold')

    #Launch Site
    ax.plot(LAUNCH_SITE['longitude'], LAUNCH_SITE['latitude'], 'k*', 
            markersize=15, transform=ccrs.PlateCarree(), label='Launch Site')

    #Ellipses for the 3 Trajectories
    case_styles = {
        'Ballistic': {'color': 'red', 'label': 'Ballistic'},
        'Drogue Only': {'color': 'lime', 'label': 'Drogue Only'},
        'Main at Apogee': {'color': 'cyan', 'label': 'Main at Apogee'}
    }

    for case_id, res in all_case_results.items():
        config = ANALYSIS_CASES[case_id]
        if config['wind'] == wind_condition_key:
            stats = res['sigma_stats']
            style = case_styles[config['name']]
            
            # Convert meters of dispersion to geographic coordinates for the map
            # (Using a simplified conversion factor for Timmins latitude)
            m_to_deg = 1 / 111139.0 
            
            for s, ls, alpha in [(1, '--', 0.6), (2, '-', 0.3)]:
                width = stats[f'{s}sigma_major'] * 2 * m_to_deg
                height = stats[f'{s}sigma_minor'] * 2 * m_to_deg
                
                ell = Ellipse(xy=(LAUNCH_SITE['longitude'], LAUNCH_SITE['latitude']),
                              width=width, height=height, angle=stats['angle'],
                              edgecolor=style['color'], facecolor=style['color'], alpha=alpha,
                              linestyle=ls, linewidth=2, transform=ccrs.PlateCarree())
                ax.add_patch(ell)

    plt.title(f"Dispersion Analysis: {wind_condition_key.replace('_', ' ').title()}", fontsize=15)
    plt.legend(loc='lower right')
    plt.show()
def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("PERIGE-GEE ROCKET DISPERSION ANALYSIS")
    print("="*80)
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # First, run a test flight
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

    high_pool, typical_pool = load_and_filter_wind_data(winddatapath)
    test_single_flight()

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