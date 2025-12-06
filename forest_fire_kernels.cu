#include "forest_fire.cuh"
#include <cmath>
using namespace std;

__constant__ SimulationParams c_params;

__device__ float calculateWindFactor(float wind_speed, float wind_direction, int dx, int dy) 
{
    float spread_direction = atan2f(static_cast<float>(dy), static_cast<float>(dx));
    float angle_diff = cosf(wind_direction - spread_direction);
    return 1.0f + 0.5f * wind_speed * fmaxf(0.0f, angle_diff);
}

__device__ float calculateSlopeFactor(float elevation_current, float elevation_neighbor) {
    float slope = (elevation_neighbor - elevation_current) / 10.0f;
    return expf(0.1f * slope * 100.0f);
}

__device__ float calculateMoistureFactor(float moisture) {
    return expf(-0.05f * moisture);
}

__device__ float calculateSpreadProbability(
    const FireGrid& grid, int x, int y, int nx, int ny, 
    const SimulationParams& params) {
    
    size_t idx = y * grid.pitch + x;
    size_t nidx = ny * grid.pitch + nx;
    
    float wind_factor = calculateWindFactor(params.wind_speed, params.wind_direction, nx - x, ny - y);
    float slope_factor = calculateSlopeFactor(grid.elevation[idx], grid.elevation[nidx]);
    float moisture_factor = calculateMoistureFactor(grid.moisture[nidx]);
    float fuel_factor = grid.fuel_load[nidx] / 100.0f;
    
    return params.base_spread_probability * wind_factor * slope_factor * moisture_factor * fuel_factor;
}

__global__ void initializeRandomStates(curandState* states, unsigned long seed, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) 
{
        int idx = y * width + x;
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void initializeGrid(FireGrid grid, float tree_density, float initial_moisture, curandState* rand_states) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < grid.width && y < grid.height) 
{
        size_t idx = y * grid.pitch + x;
        
        curandState local_state = rand_states[idx];
        float rand = curand_uniform(&local_state);
        
        grid.states[idx] = (rand < tree_density) ? TREE : EMPTY;
        grid.moisture[idx] = initial_moisture * (0.8f + 0.4f * curand_uniform(&local_state));
        grid.fuel_load[idx] = 50.0f + 50.0f * curand_uniform(&local_state);
        grid.temperature[idx] = 20.0f;
        grid.smoke_density[idx] = 0.0f;
        grid.burn_time[idx] = 0.0f;
        
        float base_elevation = 100.0f * sinf(x * 0.01f) * cosf(y * 0.01f);
        grid.elevation[idx] = base_elevation + 10.0f * curand_uniform(&local_state);
        
        float dx = sinf(x * 0.01f + 0.1f) - sinf(x * 0.01f - 0.1f);
        float dy = cosf(y * 0.01f + 0.1f) - cosf(y * 0.01f - 0.1f);
        grid.slope[idx] = sqrtf(dx * dx + dy * dy) * 100.0f;
        
        rand_states[idx] = local_state;
    }
}

__global__ void fireSpreadKernel(FireGrid current, FireGrid next, curandState* rand_states, SimulationParams params) {
    extern __shared__ uint8_t shared_states[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    int shared_width = blockDim.x + 2;
    int shared_idx = (ty + 1) * shared_width + (tx + 1);
    
    if (x < current.width && y < current.height) 
{
        shared_states[shared_idx] = current.states[y * current.pitch + x];
    }

    if (tx == 0 && x > 0) {
        shared_states[shared_idx - 1] = current.states[y * current.pitch + x - 1];
    }
    if (tx == blockDim.x - 1 && x < current.width - 1) {
        shared_states[shared_idx + 1] = current.states[y * current.pitch + x + 1];
    }
    if (ty == 0 && y > 0) {
        shared_states[shared_idx - shared_width] = current.states[(y - 1) * current.pitch + x];
    }
    if (ty == blockDim.y - 1 && y < current.height - 1) {
        shared_states[shared_idx + shared_width] = current.states[(y + 1) * current.pitch + x];
    }
    
    __syncthreads();
    
    if (x < current.width && y < current.height) {
        size_t idx = y * current.pitch + x;
        curandState local_state = rand_states[idx];
        
        uint8_t current_state = shared_states[shared_idx];
        uint8_t new_state = current_state;
        float new_temperature = current.temperature[idx];
        float new_moisture = current.moisture[idx];
        float new_smoke = current.smoke_density[idx];
        float new_burn_time = current.burn_time[idx];
        
        if (current_state == TREE) 
{
            bool will_ignite = false;
            
            for (int dy = -1; dy <= 1; dy++) 
{
                for (int dx = -1; dx <= 1; dx++) 
{
                    if (dx == 0 && dy == 0) continue;
                    
                    int nx = x + dx;
                    int ny = y + dy;
                    int neighbor_idx = (ty + 1 + dy) * shared_width + (tx + 1 + dx);
                    
                    if (nx >= 0 && nx < current.width && ny >= 0 && ny < current.height) {
            if (shared_states[neighbor_idx] == BURNING) 
{
float spread_prob = calculateSpreadProbability(current, x, y, nx, ny, params);
            if (curand_uniform(&local_state) < spread_prob) {
                                will_ignite = true;
                                break;
                            }
                        }
                    }
                }
                if (will_ignite) break;
            }
            
            if (will_ignite || (curand_uniform(&local_state) < params.ignition_probability)) {
                new_state = BURNING;
                new_temperature = 500.0f;
                new_burn_time = 0.0f;
            }
            
            new_moisture = fmaxf(0.0f, new_moisture - 0.1f * params.time_step);
        }
        else if (current_state == BURNING) {
            new_burn_time += params.time_step;
            new_temperature = 500.0f + 300.0f * expf(-new_burn_time / params.burnout_time);
            new_smoke = fminf(100.0f, new_smoke + 10.0f * params.time_step);
            
            if (new_burn_time > params.burnout_time) {
                new_state = SMOLDERING;
                new_temperature = 100.0f;
            }
            
            if (curand_uniform(&local_state) < params.ember_probability * params.time_step) {
                int ember_distance = 5 + static_cast<int>(10 * curand_uniform(&local_state));
                float ember_angle = 2.0f * 3.14159f * curand_uniform(&local_state);
                int ember_x = x + static_cast<int>(ember_distance * cosf(ember_angle + params.wind_direction));
                int ember_y = y + static_cast<int>(ember_distance * sinf(ember_angle + params.wind_direction));
                
                if (ember_x >= 0 && ember_x < current.width && ember_y >= 0 && ember_y < current.height) {
                    size_t ember_idx = ember_y * current.pitch + ember_x;
                    if (current.states[ember_idx] == TREE && curand_uniform(&local_state) < 0.3f) {
                        next.states[ember_idx] = BURNING;
                    }
                }
            }
        }
        else if (current_state == SMOLDERING) {
            new_burn_time += params.time_step;
            new_temperature = fmaxf(20.0f, 100.0f * expf(-(new_burn_time - params.burnout_time) / params.smolder_time));
            new_smoke = fmaxf(0.0f, new_smoke - 1.0f * params.time_step);
            
            if (new_burn_time > params.burnout_time + params.smolder_time) {
                new_state = EMPTY;
                new_temperature = 20.0f;
            }
        }
        
        next.states[idx] = new_state;
        next.temperature[idx] = new_temperature;
        next.moisture[idx] = new_moisture;
        next.smoke_density[idx] = new_smoke;
        next.burn_time[idx] = new_burn_time;
        next.fuel_load[idx] = current.fuel_load[idx];
        next.elevation[idx] = current.elevation[idx];
        next.slope[idx] = current.slope[idx];
        
        rand_states[idx] = local_state;
    }
}

__global__ void heatDiffusionKernel(FireGrid grid, float* temp_buffer, float diffusion_rate) {
    extern __shared__ float shared_temp[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    int shared_width = blockDim.x + 2;
    int shared_idx = (ty + 1) * shared_width + (tx + 1);
    
    if (x < grid.width && y < grid.height) {
        shared_temp[shared_idx] = grid.temperature[y * grid.pitch + x];
    }
    
    if (tx == 0 && x > 0) {
        shared_temp[shared_idx - 1] = grid.temperature[y * grid.pitch + x - 1];
    }
    if (tx == blockDim.x - 1 && x < grid.width - 1) {
        shared_temp[shared_idx + 1] = grid.temperature[y * grid.pitch + x + 1];
    }
    if (ty == 0 && y > 0) {
        shared_temp[shared_idx - shared_width] = grid.temperature[(y - 1) * grid.pitch + x];
    }
    if (ty == blockDim.y - 1 && y < grid.height - 1) {
        shared_temp[shared_idx + shared_width] = grid.temperature[(y + 1) * grid.pitch + x];
    }
    
    __syncthreads();
    
    if (x > 0 && x < grid.width - 1 && y > 0 && y < grid.height - 1) {
        float laplacian = shared_temp[shared_idx - 1] + shared_temp[shared_idx + 1] +
                         shared_temp[shared_idx - shared_width] + shared_temp[shared_idx + shared_width] -
                         4.0f * shared_temp[shared_idx];
        
        size_t idx = y * grid.pitch + x;
        temp_buffer[idx] = shared_temp[shared_idx] + diffusion_rate * laplacian;
    }
}

__global__ void addFirebreakKernel(FireGrid grid, int x1, int y1, int x2, int y2, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < grid.width && y < grid.height) {
        float dx = x2 - x1;
        float dy = y2 - y1;
        float length = sqrtf(dx * dx + dy * dy);
        
        if (length > 0) {
            dx /= length;
            dy /= length;
            
            float dist_to_line = fabsf((x - x1) * dy - (y - y1) * dx);
            float proj = (x - x1) * dx + (y - y1) * dy;
            
            if (dist_to_line <= width / 2.0f && proj >= 0 && proj <= length) {
                size_t idx = y * grid.pitch + x;
                grid.states[idx] = EMPTY;
                grid.fuel_load[idx] = 0.0f;
            }
        }
    }
}

__global__ void applyWaterBombingKernel(FireGrid grid, int cx, int cy, int radius, float effectiveness) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < grid.width && y < grid.height) {
        int dx = x - cx;
        int dy = y - cy;
        float dist = sqrtf(static_cast<float>(dx * dx + dy * dy));
        
        if (dist <= radius) {
            size_t idx = y * grid.pitch + x;
            
            float effect = effectiveness * (1.0f - dist / radius);
            grid.moisture[idx] = fminf(100.0f, grid.moisture[idx] + 50.0f * effect);
            
            if (grid.states[idx] == BURNING && effect > 0.7f) {
                grid.states[idx] = SMOLDERING;
                grid.temperature[idx] = 100.0f;
            }
            
            grid.temperature[idx] *= (1.0f - 0.5f * effect);
        }
    }
}

__global__ void calculateStatisticsKernel(FireGrid grid, int* fire_count, int* burned_count, int* tree_count) {
    extern __shared__ int shared_counts[];
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    shared_counts[tid * 3] = 0;
    shared_counts[tid * 3 + 1] = 0;
    shared_counts[tid * 3 + 2] = 0;
    
    if (x < grid.width && y < grid.height) {
        size_t idx = y * grid.pitch + x;
        uint8_t state = grid.states[idx];
        
        if (state == BURNING || state == SMOLDERING) {
            shared_counts[tid * 3]++;
        }
        if (state == EMPTY) {
            shared_counts[tid * 3 + 1]++;
        }
        if (state == TREE) {
            shared_counts[tid * 3 + 2]++;
        }
    }
    
    __syncthreads();
    
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_counts[tid * 3] += shared_counts[(tid + stride) * 3];
            shared_counts[tid * 3 + 1] += shared_counts[(tid + stride) * 3 + 1];
            shared_counts[tid * 3 + 2] += shared_counts[(tid + stride) * 3 + 2];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(fire_count, shared_counts[0]);
        atomicAdd(burned_count, shared_counts[1]);
        atomicAdd(tree_count, shared_counts[2]);
    }
}
