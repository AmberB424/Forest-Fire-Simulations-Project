#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import subprocess
import time
import os
from PIL import Image
class ForestFireVisualizer:
    def __init__(self, width=1024, height=1024):
        self.width = width
        self.height = height
        
        colors = ['#333333', '#228B22', '#FF4500', '#8B4513']
        self.cmap = ListedColormap(colors)
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 12))
        self.fig.suptitle('Forest Fire Simulation - Real-time Monitor', fontsize=16)
        
        self.im_state = self.axes[0, 0].imshow(np.zeros((height, width)), 
                                                cmap=self.cmap, vmin=0, vmax=3)
        self.axes[0, 0].set_title('Fire State')
        self.axes[0, 0].axis('off')
        
        self.temp_data = np.zeros((height, width))
        self.im_temp = self.axes[0, 1].imshow(self.temp_data, cmap='hot', vmin=20, vmax=500)
        self.axes[0, 1].set_title('Temperature Field')
        self.axes[0, 1].axis('off')
        self.fig.colorbar(self.im_temp, ax=self.axes[0, 1], label='Temperature (°C)')
        
        self.time_data = []
        self.fire_percent = []
        self.burned_percent = []
        self.line_fire, = self.axes[1, 0].plot([], [], 'r-', label='Active Fire %')
        self.line_burned, = self.axes[1, 0].plot([], [], 'k-', label='Total Burned %')
        self.axes[1, 0].set_xlabel('Time Steps')
        self.axes[1, 0].set_ylabel('Percentage')
        self.axes[1, 0].set_title('Fire Progression')
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True)
        self.axes[1, 0].set_xlim(0, 500)
        self.axes[1, 0].set_ylim(0, 100)
        
        self.wind_arrow = None
        self.axes[1, 1].set_xlim(-1.5, 1.5)
        self.axes[1, 1].set_ylim(-1.5, 1.5)
        self.axes[1, 1].set_aspect('equal')
        self.axes[1, 1].set_title('Wind Conditions')
        self.axes[1, 1].grid(True)
        
        plt.tight_layout()
    
    def update_state(self, state_data):
        self.im_state.set_data(state_data)
    
    def update_temperature(self, temp_data):
        self.im_temp.set_data(temp_data)
        self.im_temp.set_clim(vmin=temp_data.min(), vmax=temp_data.max())
    
    def update_statistics(self, timestep, fire_pct, burned_pct):
        self.time_data.append(timestep)
        self.fire_percent.append(fire_pct)
        self.burned_percent.append(burned_pct)
        
        self.line_fire.set_data(self.time_data, self.fire_percent)
        self.line_burned.set_data(self.time_data, self.burned_percent)
        
        if timestep > 500:
            self.axes[1, 0].set_xlim(timestep - 500, timestep)
    
    def update_wind(self, speed, direction):
        if self.wind_arrow:
            self.wind_arrow.remove()
        
        dx = speed / 20.0 * np.cos(direction)
        dy = speed / 20.0 * np.sin(direction)
        
        self.wind_arrow = self.axes[1, 1].arrow(0, 0, dx, dy, 
                                                head_width=0.1, head_length=0.1, 
                                                fc='blue', ec='blue')
        self.axes[1, 1].set_title(f'Wind: {speed:.1f} m/s @ {np.degrees(direction):.0f}°')
    
    def save_frame(self, filename):
        self.fig.savefig(filename, dpi=100, bbox_inches='tight')
    
    def show(self):
        plt.show()

def load_ppm_file(filename):
    img = Image.open(filename)
    data = np.array(img)
    
    state_map = {
        (50, 50, 50): 0,
        (0, 150, 0): 1,
        (255, 100, 0): 2,
        (150, 50, 0): 3
    }
    
    state = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            pixel = tuple(data[i, j])
            state[i, j] = state_map.get(pixel, 0)
    
    return state

def generate_temperature_field(state):
    temp = np.ones_like(state, dtype=np.float32) * 20.0
    temp[state == 2] = 500.0
    temp[state == 3] = 100.0
    
    from scipy.ndimage import gaussian_filter
    temp = gaussian_filter(temp, sigma=2.0)
    
    return temp

def run_live_visualization():
    print("Starting Forest Fire Simulation with Live Visualization...")
    
    viz = ForestFireVisualizer(1024, 1024)
    
    process = subprocess.Popen(['./forest_fire_sim'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
    
    timestep = 0
    wind_speed = 10.0
    wind_direction = np.pi / 4
    
    viz.update_wind(wind_speed, wind_direction)
    
    def update_plot(frame):
        nonlocal timestep
        
        ppm_file = f"fire_state_{timestep * 100}.ppm"
        if os.path.exists(ppm_file):
            state = load_ppm_file(ppm_file)
            temp = generate_temperature_field(state)
            
            viz.update_state(state)
            viz.update_temperature(temp)
            
            fire_pct = np.sum(state == 2) / state.size * 100
            burned_pct = np.sum(state == 0) / np.sum((state == 0) | (state == 1) | (state == 2) | (state == 3)) * 100
            
            viz.update_statistics(timestep, fire_pct, burned_pct)
            
            if timestep % 10 == 0:
                print(f"Timestep {timestep}: Fire={fire_pct:.2f}%, Burned={burned_pct:.2f}%")
            
            timestep += 1
        
        return [viz.im_state, viz.im_temp, viz.line_fire, viz.line_burned]
    
    ani = animation.FuncAnimation(viz.fig, update_plot, interval=100, blit=True)
    # ani.save('forest_fire_simulation.gif', writer='pillow', fps=10)
    
    try:
        viz.show()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        process.terminate()
        print("Cleaning up temporary files...")
        for f in os.listdir('.'):
            if f.endswith('.ppm'):
                os.remove(f)

def analyze_results():
    print("Analyzing simulation results...")
    
    results = {
        'scenarios': [],
        'final_burned': [],
        'peak_fire': [],
        'time_to_peak': [],
        'control_effectiveness': []
    }
    
    scenarios = [
        {'name': 'No Control', 'firebreaks': 0, 'water_bombing': False},
        {'name': 'Firebreaks Only', 'firebreaks': 4, 'water_bombing': False},
        {'name': 'Water Bombing Only', 'firebreaks': 0, 'water_bombing': True},
        {'name': 'Combined Strategy', 'firebreaks': 4, 'water_bombing': True}
    ]
    
    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario['name']}...")
        
        peak_fire = 0
        time_to_peak = 0
        
        for t in range(500):
            if t % 100 == 0:
                ppm_file = f"fire_state_{t}.ppm"
                if os.path.exists(ppm_file):
                    state = load_ppm_file(ppm_file)
                    fire_pct = np.sum(state == 2) / state.size * 100
                    
                    if fire_pct > peak_fire:
                        peak_fire = fire_pct
                        time_to_peak = t
        
        final_state = load_ppm_file("fire_state_400.ppm") if os.path.exists("fire_state_400.ppm") else np.zeros((1024, 1024))
        final_burned = np.sum(final_state == 0) / final_state.size * 100
        
        results['scenarios'].append(scenario['name'])
        results['final_burned'].append(final_burned)
        results['peak_fire'].append(peak_fire)
        results['time_to_peak'].append(time_to_peak)
        
        baseline = 50.0
        effectiveness = (baseline - final_burned) / baseline * 100
        results['control_effectiveness'].append(effectiveness)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Forest Fire Control Strategy Analysis', fontsize=16)
    
    x = np.arange(len(results['scenarios']))
    
    axes[0, 0].bar(x, results['final_burned'], color=['red', 'orange', 'yellow', 'green'])
    axes[0, 0].set_ylabel('Final Burned Area (%)')
    axes[0, 0].set_title('Total Damage by Strategy')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(results['scenarios'], rotation=45, ha='right')
    
    axes[0, 1].bar(x, results['peak_fire'], color='orange')
    axes[0, 1].set_ylabel('Peak Fire Coverage (%)')
    axes[0, 1].set_title('Maximum Fire Intensity')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(results['scenarios'], rotation=45, ha='right')
    
    axes[1, 0].bar(x, results['time_to_peak'], color='blue')
    axes[1, 0].set_ylabel('Time Steps')
    axes[1, 0].set_title('Time to Peak Fire')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(results['scenarios'], rotation=45, ha='right')
    
    axes[1, 1].bar(x, results['control_effectiveness'], color='green')
    axes[1, 1].set_ylabel('Effectiveness (%)')
    axes[1, 1].set_title('Control Strategy Effectiveness')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(results['scenarios'], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('fire_control_analysis.png', dpi=150)
    plt.show()
    
    print("\n=== Analysis Summary ===")
    for i, scenario in enumerate(results['scenarios']):
        print(f"\n{scenario}:")
        print(f"  - Final burned area: {results['final_burned'][i]:.2f}%")
        print(f"  - Peak fire coverage: {results['peak_fire'][i]:.2f}%")
        print(f"  - Time to peak: {results['time_to_peak'][i]} steps")
        print(f"  - Effectiveness: {results['control_effectiveness'][i]:.2f}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "live":
            run_live_visualization()
        elif sys.argv[1] == "analyze":
            analyze_results()
    else:
        print("Usage: python visualize.py [live|analyze]")
        print("  live    - Run simulation with live visualization")
        print("  analyze - Analyze and compare control strategies")
