#include "forest_fire.cuh"
#include "forest_fire_kernels.cu"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

ForestFireSimulation::ForestFireSimulation(int width, int height) 
    : width_(width), height_(height), step_count_(0) {
    
    h_params.grid_width = width;
    h_params.grid_height = height;
    h_params.base_spread_probability = 0.58f;
    h_params.wind_speed = 5.0f;
    h_params.wind_direction = 0.0f;
    h_params.ignition_probability = 0.00001f;
    h_params.burnout_time = 10.0f;
    h_params.smolder_time = 5.0f;
    h_params.ember_probability = 0.01f;
    h_params.suppression_effectiveness = 0.8f;
    h_params.time_step = 0.1f;
    
    block_dims = dim3(BLOCK_SIZE, BLOCK_SIZE);
    grid_dims = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                     (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    allocateMemory();
    
    cudaMemcpyToSymbol(c_params, &h_params, sizeof(SimulationParams));
}

ForestFireSimulation::~ForestFireSimulation() {
    freeMemory();
}

void ForestFireSimulation::allocateMemory() {
    size_t pitch;
    size_t width_bytes = width_ * sizeof(uint8_t);
    size_t float_width_bytes = width_ * sizeof(float);
    
    cudaMallocPitch((void**)&d_current.states, &pitch, width_bytes, height_);
    d_current.pitch = pitch / sizeof(uint8_t);
    d_next.pitch = d_current.pitch;
    
    cudaMallocPitch((void**)&d_next.states, &pitch, width_bytes, height_);
    
    cudaMalloc((void**)&d_current.moisture, float_width_bytes * height_);
    cudaMalloc((void**)&d_current.fuel_load, float_width_bytes * height_);
    cudaMalloc((void**)&d_current.temperature, float_width_bytes * height_);
    cudaMalloc((void**)&d_current.elevation, float_width_bytes * height_);
    cudaMalloc((void**)&d_current.slope, float_width_bytes * height_);
    cudaMalloc((void**)&d_current.smoke_density, float_width_bytes * height_);
    cudaMalloc((void**)&d_current.burn_time, float_width_bytes * height_);
    
    cudaMalloc((void**)&d_next.moisture, float_width_bytes * height_);
    cudaMalloc((void**)&d_next.fuel_load, float_width_bytes * height_);
    cudaMalloc((void**)&d_next.temperature, float_width_bytes * height_);
    cudaMalloc((void**)&d_next.elevation, float_width_bytes * height_);
    cudaMalloc((void**)&d_next.slope, float_width_bytes * height_);
    cudaMalloc((void**)&d_next.smoke_density, float_width_bytes * height_);
    cudaMalloc((void**)&d_next.burn_time, float_width_bytes * height_);
    
    d_current.width = width_;
    d_current.height = height_;
    d_next.width = width_;
    d_next.height = height_;
    
    cudaMalloc((void**)&d_rand_states, width_ * height_ * sizeof(curandState));
    cudaMalloc((void**)&d_params, sizeof(SimulationParams));
    cudaMemcpy(d_params, &h_params, sizeof(SimulationParams), cudaMemcpyHostToDevice);
    
    initializeRandomStates<<<grid_dims, block_dims>>>(d_rand_states, time(0), width_, height_);
    cudaDeviceSynchronize();
}

void ForestFireSimulation::freeMemory() {
    cudaFree(d_current.states);
    cudaFree(d_current.moisture);
    cudaFree(d_current.fuel_load);
    cudaFree(d_current.temperature);
    cudaFree(d_current.elevation);
    cudaFree(d_current.slope);
    cudaFree(d_current.smoke_density);
    cudaFree(d_current.burn_time);
    
    cudaFree(d_next.states);
    cudaFree(d_next.moisture);
    cudaFree(d_next.fuel_load);
    cudaFree(d_next.temperature);
    cudaFree(d_next.elevation);
    cudaFree(d_next.slope);
    cudaFree(d_next.smoke_density);
    cudaFree(d_next.burn_time);
    
    cudaFree(d_rand_states);
    cudaFree(d_params);
}

void ForestFireSimulation::initialize(float tree_density, float initial_moisture) {
    initializeGrid<<<grid_dims, block_dims>>>(d_current, tree_density, initial_moisture, d_rand_states);
    cudaDeviceSynchronize();
    
    cudaMemcpy(d_next.states, d_current.states, width_ * height_ * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_next.moisture, d_current.moisture, width_ * height_ * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_next.fuel_load, d_current.fuel_load, width_ * height_ * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_next.temperature, d_current.temperature, width_ * height_ * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_next.elevation, d_current.elevation, width_ * height_ * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_next.slope, d_current.slope, width_ * height_ * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_next.smoke_density, d_current.smoke_density, width_ * height_ * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_next.burn_time, d_current.burn_time, width_ * height_ * sizeof(float), cudaMemcpyDeviceToDevice);
}

void ForestFireSimulation::setIgnitionPoint(int x, int y) {
    if (x >= 0 && x < width_ && y >= 0 && y < height_) {
        uint8_t state = BURNING;
        size_t offset = y * d_current.pitch + x;
        cudaMemcpy(&d_current.states[offset], &state, sizeof(uint8_t), cudaMemcpyHostToDevice);
        
        float temp = 500.0f;
        cudaMemcpy(&d_current.temperature[y * width_ + x], &temp, sizeof(float), cudaMemcpyHostToDevice);
    }
}

void ForestFireSimulation::setWindConditions(float speed, float direction) {
    h_params.wind_speed = speed;
    h_params.wind_direction = direction;
    cudaMemcpyToSymbol(c_params, &h_params, sizeof(SimulationParams));
    cudaMemcpy(d_params, &h_params, sizeof(SimulationParams), cudaMemcpyHostToDevice);
}

void ForestFireSimulation::addFirebreak(int x1, int y1, int x2, int y2, int width) {
    addFirebreakKernel<<<grid_dims, block_dims>>>(d_current, x1, y1, x2, y2, width);
    cudaDeviceSynchronize();
}

void ForestFireSimulation::addWaterBombing(int x, int y, int radius, float effectiveness) {
    applyWaterBombingKernel<<<grid_dims, block_dims>>>(d_current, x, y, radius, effectiveness);
    cudaDeviceSynchronize();
}

void ForestFireSimulation::step() {
    size_t shared_size = (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * sizeof(uint8_t);
    fireSpreadKernel<<<grid_dims, block_dims, shared_size>>>(
        d_current, d_next, d_rand_states, h_params);
    
    float* temp_buffer;
    cudaMalloc((void**)&temp_buffer, width_ * height_ * sizeof(float));
    
    size_t temp_shared_size = (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * sizeof(float);
    heatDiffusionKernel<<<grid_dims, block_dims, temp_shared_size>>>(
        d_next, temp_buffer, 0.1f);
    
    cudaMemcpy(d_next.temperature, temp_buffer, width_ * height_ * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(temp_buffer);
    
    cudaDeviceSynchronize();
    swapBuffers();
    step_count_++;
}

void ForestFireSimulation::swapBuffers() {
    std::swap(d_current.states, d_next.states);
    std::swap(d_current.moisture, d_next.moisture);
    std::swap(d_current.fuel_load, d_next.fuel_load);
    std::swap(d_current.temperature, d_next.temperature);
    std::swap(d_current.smoke_density, d_next.smoke_density);
    std::swap(d_current.burn_time, d_next.burn_time);
}

void ForestFireSimulation::getState(std::vector<uint8_t>& output) {
    output.resize(width_ * height_);
    
    std::vector<uint8_t> pitched_data(d_current.pitch * height_);
    cudaMemcpy2D(pitched_data.data(), width_ * sizeof(uint8_t),
                 d_current.states, d_current.pitch * sizeof(uint8_t),
                 width_ * sizeof(uint8_t), height_,
                 cudaMemcpyDeviceToHost);
    
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            output[y * width_ + x] = pitched_data[y * d_current.pitch + x];
        }
    }
}

float ForestFireSimulation::getFirePercentage() const {
    int *d_fire_count, *d_burned_count, *d_tree_count;
    cudaMalloc((void**)&d_fire_count, sizeof(int));
    cudaMalloc((void**)&d_burned_count, sizeof(int));
    cudaMalloc((void**)&d_tree_count, sizeof(int));
    
    cudaMemset(d_fire_count, 0, sizeof(int));
    cudaMemset(d_burned_count, 0, sizeof(int));
    cudaMemset(d_tree_count, 0, sizeof(int));
    
    size_t shared_size = BLOCK_SIZE * BLOCK_SIZE * 3 * sizeof(int);
    calculateStatisticsKernel<<<grid_dims, block_dims, shared_size>>>(
        d_current, d_fire_count, d_burned_count, d_tree_count);
    
    int fire_count, burned_count, tree_count;
    cudaMemcpy(&fire_count, d_fire_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&burned_count, d_burned_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tree_count, d_tree_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_fire_count);
    cudaFree(d_burned_count);
    cudaFree(d_tree_count);
    
    int total = fire_count + burned_count + tree_count;
    return (total > 0) ? (100.0f * fire_count / total) : 0.0f;
}

float ForestFireSimulation::getBurnedPercentage() const {
    int *d_fire_count, *d_burned_count, *d_tree_count;
    cudaMalloc((void**)&d_fire_count, sizeof(int));
    cudaMalloc((void**)&d_burned_count, sizeof(int));
    cudaMalloc((void**)&d_tree_count, sizeof(int));
    
    cudaMemset(d_fire_count, 0, sizeof(int));
    cudaMemset(d_burned_count, 0, sizeof(int));
    cudaMemset(d_tree_count, 0, sizeof(int));
    
    size_t shared_size = BLOCK_SIZE * BLOCK_SIZE * 3 * sizeof(int);
    calculateStatisticsKernel<<<grid_dims, block_dims, shared_size>>>(
        d_current, d_fire_count, d_burned_count, d_tree_count);
    
    int fire_count, burned_count, tree_count;
    cudaMemcpy(&fire_count, d_fire_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&burned_count, d_burned_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tree_count, d_tree_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_fire_count);
    cudaFree(d_burned_count);
    cudaFree(d_tree_count);
    
    int total = width_ * height_;
    int initial_trees = tree_count + fire_count + burned_count;
    return (initial_trees > 0) ? (100.0f * burned_count / initial_trees) : 0.0f;
}
