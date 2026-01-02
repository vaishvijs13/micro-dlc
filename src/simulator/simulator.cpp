#include "simulator/simulator.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace dlcompiler {
namespace simulator {

std::string ChipConfig::toString() const {
    std::stringstream ss;
    ss << "ChipConfig{\n";
    ss << "  compute_units: " << compute_units << "\n";
    ss << "  memory_bandwidth: " << memory_bandwidth_gb_s << " GB/s\n";
    ss << "  cache_size: " << cache_size_kb << " KB\n";
    ss << "  simd_width: " << simd_width << "\n";
    ss << "  clock_freq: " << clock_freq_ghz << " GHz\n";
    ss << "}";
    return ss.str();
}

void ExecutionStats::print() const {
    std::cout << "\n=== Execution Statistics ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total cycles:          " << cycles << "\n";
    std::cout << "Execution time:        " << execution_time_ms << " ms\n";
    std::cout << "Memory accesses:       " << memory_accesses << "\n";
    std::cout << "Cache hits:            " << cache_hits << " (" 
              << (100.0 * cache_hits / std::max(1LL, cache_hits + cache_misses)) << "%)\n";
    std::cout << "Cache misses:          " << cache_misses << " (" 
              << (100.0 * cache_misses / std::max(1LL, cache_hits + cache_misses)) << "%)\n";
    std::cout << "Compute utilization:   " << compute_utilization << "%\n";
    std::cout << "Memory bound time:     " << memory_bound_time << "%\n";
    std::cout << "-----------------------\n";
}

bool CacheModel::access(int64_t, int64_t size) {
    // simple cache model: if data fits in cache, it's a hit

    if (size <= size_bytes_) {
        if (current_usage_ + size <= size_bytes_) {
            current_usage_ += size;
            hits_++;
            return true;
        }
    }
    
    // cache miss
    misses_++;
    current_usage_ = std::min(current_usage_ + size, size_bytes_);
    return false;
}

void CacheModel::reset() {
    current_usage_ = 0;
    hits_ = 0;
    misses_ = 0;
}

ExecutionStats Simulator::execute(const std::vector<codegen::Instruction>& instructions) {
    std::cout << "\n ----> Simulating Execution <----\n";
    std::cout << config_.toString() << "\n\n";
    
    ExecutionStats stats;
    cache_.reset();
    
    int64_t compute_cycles = 0;
    int64_t memory_cycles = 0;
    
    for (const auto& inst : instructions) {
        int64_t inst_cycles = 0;
        
        switch (inst.type) {
            case codegen::InstructionType::LOAD:
                inst_cycles = simulateLoad(inst);
                memory_cycles += inst_cycles;
                stats.memory_accesses++;
                break;
                
            case codegen::InstructionType::STORE:
                inst_cycles = simulateStore(inst);
                memory_cycles += inst_cycles;
                stats.memory_accesses++;
                break;
                
            case codegen::InstructionType::COMPUTE:
                inst_cycles = simulateCompute(inst);
                compute_cycles += inst_cycles;
                break;
                
            case codegen::InstructionType::SYNC:
                inst_cycles = 10;
                break;
        }
        
        stats.cycles += inst_cycles;
    }
    
    // cache stats
    stats.cache_hits = cache_.hits();
    stats.cache_misses = cache_.misses();
    
    // calc execution time
    stats.execution_time_ms = stats.cycles / (config_.clock_freq_ghz * 1e6);
    
    // calc utilization
    int64_t total_cycles = compute_cycles + memory_cycles;
    if (total_cycles > 0) {
        stats.compute_utilization = 100.0 * compute_cycles / total_cycles;
        stats.memory_bound_time = 100.0 * memory_cycles / total_cycles;
    }
    
    std::cout << "Simulation complete\n";
    stats.print();
    
    return stats;
}

int64_t Simulator::simulateLoad(const codegen::Instruction& inst) {
    // check cache
    bool cache_hit = cache_.access(0, inst.input_size);
    
    if (cache_hit) {
        return 10;
    } else {
        // cache miss: load from mem
        // cycles = data_size / (bandwidth * bytes_per_cycle)
        double bytes_per_cycle = (config_.memory_bandwidth_gb_s * 1e9) / 
                                 (config_.clock_freq_ghz * 1e9);
        int64_t cycles = static_cast<int64_t>(inst.input_size / bytes_per_cycle);
        return std::max(cycles, 100LL);
    }
}

int64_t Simulator::simulateStore(const codegen::Instruction& inst) {
    double bytes_per_cycle = (config_.memory_bandwidth_gb_s * 1e9) / 
                             (config_.clock_freq_ghz * 1e9);
    int64_t cycles = static_cast<int64_t>(inst.output_size / bytes_per_cycle);
    return std::max(cycles, 100LL);
}

int64_t Simulator::simulateCompute(const codegen::Instruction& inst) {
    double flops_per_cycle = config_.compute_units * config_.simd_width * 2.0;
    int64_t cycles = static_cast<int64_t>(inst.flops / flops_per_cycle);
    return std::max(cycles, 1LL);
}

}
}

