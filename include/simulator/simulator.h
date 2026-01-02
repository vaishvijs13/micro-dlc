#pragma once

#include "codegen/codegen.h"
#include <vector>

namespace dlcompiler {
namespace simulator {

// hardware config
struct ChipConfig {
    int compute_units = 16; // num of parallel compute units
    double memory_bandwidth_gb_s = 100; // mem bandwidth
    int cache_size_kb = 256; // L1 cache size
    int simd_width = 8; // SIMD vector width
    double clock_freq_ghz = 1.5; // clock frequency
    
    std::string toString() const;
};

// execution stats
struct ExecutionStats {
    int64_t cycles = 0;
    int64_t memory_accesses = 0;
    int64_t cache_hits = 0;
    int64_t cache_misses = 0;
    double execution_time_ms = 0;
    double compute_utilization = 0;
    double memory_bound_time = 0;
    
    void print() const;
};

// simple cache model
class CacheModel {
public:
    CacheModel(int size_kb) : size_bytes_(size_kb * 1024) {}
    
    bool access(int64_t address, int64_t size);
    void reset();
    
    int64_t hits() const { return hits_; }
    int64_t misses() const { return misses_; }
    
private:
    int64_t size_bytes_;
    int64_t current_usage_ = 0;
    int64_t hits_ = 0;
    int64_t misses_ = 0;
};

// simulator
class Simulator {
public:
    Simulator(const ChipConfig& config) : config_(config), cache_(config.cache_size_kb) {}
    
    ExecutionStats execute(const std::vector<codegen::Instruction>& instructions);
    
    const ChipConfig& config() const { return config_; }
    
private:
    int64_t simulateLoad(const codegen::Instruction& inst);
    int64_t simulateStore(const codegen::Instruction& inst);
    int64_t simulateCompute(const codegen::Instruction& inst);
    
    ChipConfig config_;
    CacheModel cache_;
};

}
}

