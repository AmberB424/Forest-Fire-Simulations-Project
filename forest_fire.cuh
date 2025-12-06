#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <memory>
using namespace std;

constexpr int BLOCK_SIZE = 32;
constexpr int TILE_SIZE = 256;
constexpr int MAX_GRID_SIZE = 16384;

enum CellState : uint8_t {
    EMPTY = 0,
    TREE = 1,
    BURNING = 2,
    SMOLDERING = 3
};

struct SimulationParams {
    float base_spread_probability;
    float wind_speed;
    float wind_direction;
    float ignition_probability;
    float burnout_time;
    float smolder_time;
    float ember_probability;
    float suppression_effectiveness;
    int grid_width;
    int grid_height;
    float time_step;
};

struct CellData {
    uint8_t state;
    float moisture;
    float fuel_load;
    float temperature;
    float elevation;
    float slope;
};

struct FireGrid {
    uint8_t* states;
    float* moisture;
    float* fuel_load;
    float* temperature;
    float* elevation;
    float* slope;
    float* smoke_density;
    float* burn_time;
    size_t width;
    size_t height;
    size_t pitch;
};

class ForestFireSimulation {
public:
    ForestFireSimulation(int width, int height);
    ~ForestFireSimulation();
    
    void initialize(float tree_density, float initial_moisture);
    void setIgnitionPoint(int x, int y);
    void setWindConditions(float speed, float direction);
    void addFirebreak(int x1, int y1, int x2, int y2, int width);
    void addWaterBombing(int x, int y, int radius, float effectiveness);
    void step();
    void getState(vector<uint8_t>& output);
    float getFirePercentage() const;
    float getBurnedPercentage() const;
    
private:
    FireGrid d_current;
    FireGrid d_next;
    SimulationParams h_params;
    SimulationParams* d_params;
    curandState* d_rand_states;
    
    dim3 grid_dims;
    dim3 block_dims;
    
    int width_;
    int height_;
    int step_count_;
    
    void allocateMemory();
    void freeMemory();
    void swapBuffers();
};
