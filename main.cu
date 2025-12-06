#include "forest_fire.cuh"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace std;

void saveStateToFile(const vector<uint8_t>& state, int width, int height, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }
    
    file << "P3\n" << width << " " << height << "\n255\n";
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t s = state[y * width + x];
            switch (s) {
                case EMPTY:
                    file << "50 50 50 ";
                    break;
                case TREE:
                    file << "0 150 0 ";
                    break;
                case BURNING:
                    file << "255 100 0 ";
                    break;
                case SMOLDERING:
                    file << "150 50 0 ";
                    break;
            }
        }
        file << "\n";
    }
    
    file.close();
}

void runBenchmark(int grid_size, int num_steps) {
    cout << "\n=== Performance Benchmark ===" << endl;
    cout << "Grid Size: " << grid_size << "x" << grid_size << endl;
    cout << "Total Cells: " << (grid_size * grid_size) << endl;
    cout << "Timesteps: " << num_steps << endl;
    
    ForestFireSimulation sim(grid_size, grid_size);
    sim.initialize(0.7f, 30.0f);
    
    sim.setIgnitionPoint(grid_size / 2, grid_size / 2);
    sim.setWindConditions(10.0f, 3.14159f / 4);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int i = 0; i < num_steps; i++) {
        sim.step();
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    float seconds = milliseconds / 1000.0f;
    float cells_per_second = (static_cast<float>(grid_size * grid_size * num_steps)) / seconds;
    float timesteps_per_second = num_steps / seconds;
    
    cout << "\nResults:" << endl;
    cout << "Total Time: " << seconds << " seconds" << endl;
    cout << "Timesteps/second: " << timesteps_per_second << endl;
    cout << "Cell Updates/second: " << cells_per_second / 1e6 << " million" << endl;
    cout << "Memory Bandwidth Used: "
         << (cells_per_second * sizeof(CellData) * 2) / (1024.0f * 1024.0f * 1024.0f) 
         << " GB/s" << endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrateControlStrategies() {
    cout << "\n=== Control Strategy Demonstration ===" << endl;
    
    const int width = 1024;
    const int height = 1024;
    const int num_steps = 500;
    
    ForestFireSimulation sim(width, height);
    sim.initialize(0.65f, 25.0f);
    
    cout << "Adding firebreaks..." << endl;
    sim.addFirebreak(200, 0, 200, height, 20);
    sim.addFirebreak(400, 0, 400, height, 20);
    sim.addFirebreak(600, 0, 600, height, 20);
    sim.addFirebreak(800, 0, 800, height, 20);
    
    sim.addFirebreak(0, 200, width, 200, 20);
    sim.addFirebreak(0, 400, width, 400, 20);
    sim.addFirebreak(0, 600, width, 600, 20);
    sim.addFirebreak(0, 800, width, 800, 20);
    
    cout << "Setting wind conditions (15 m/s, NE direction)..." << endl;
    sim.setWindConditions(15.0f, -3.14159f / 4);
    
    cout << "Igniting fire at center..." << endl;
    sim.setIgnitionPoint(width / 2, height / 2);
    
    vector<uint8_t> state;
    
    for (int step = 0; step < num_steps; step++) {
        sim.step();
        
        if (step % 50 == 0) {
            float fire_pct = sim.getFirePercentage();
            float burned_pct = sim.getBurnedPercentage();
            
            cout << "Step " << setw(3) << step 
                 << " - Active Fire: " << fixed << setprecision(2) << fire_pct << "%"
                 << ", Total Burned: " << burned_pct << "%" << endl;
            
            if (fire_pct > 5.0f && step == 100) {
                cout << "  -> Applying water bombing at hotspots..." << endl;
                sim.addWaterBombing(width / 2 + 50, height / 2 + 50, 30, 0.9f);
                sim.addWaterBombing(width / 2 - 50, height / 2 - 50, 30, 0.9f);
                sim.addWaterBombing(width / 2 + 100, height / 2, 30, 0.9f);
            }
            
            if (step % 100 == 0) {
                sim.getState(state);
                string filename = "fire_state_" + to_string(step) + ".ppm";
                saveStateToFile(state, width, height, filename);
                cout << "  -> Saved visualization to " << filename << endl;
            }
        }
    }
    
    cout << "\nSimulation Complete!" << endl;
    float final_burned = sim.getBurnedPercentage();
    cout << "Final burned area: " << final_burned << "%" << endl;
    
    if (final_burned < 20.0f) {
        cout << "Control strategies were HIGHLY EFFECTIVE" << endl;
    } else if (final_burned < 40.0f) {
        cout << "Control strategies were MODERATELY EFFECTIVE" << endl;
    } else {
        cout << "Control strategies had LIMITED EFFECTIVENESS" << endl;
    }
}

int main(int argc, char** argv) {
    int device_id = 0;
    cudaSetDevice(device_id);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    cout << "==================================" << endl;
    cout << "Forest Fire Simulation - CUDA" << endl;
    cout << "==================================" << endl;
    cout << "Device: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "SMs: " << prop.multiProcessorCount << endl;
    cout << "Global Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB" << endl;
    cout << "Memory Bandwidth: " 
         << (prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2.0 / 1e6) 
         << " GB/s" << endl;
    
    cout << "\n1. Running small-scale test (512x512)..." << endl;
    runBenchmark(512, 100);
    
    cout << "\n2. Running medium-scale test (2048x2048)..." << endl;
    runBenchmark(2048, 100);
    
    cout << "\n3. Running large-scale test (8192x8192)..." << endl;
    runBenchmark(8192, 50);
    
    cout << "\n4. Demonstrating control strategies..." << endl;
    demonstrateControlStrategies();
    
    cudaDeviceReset();
    return 0;
}
