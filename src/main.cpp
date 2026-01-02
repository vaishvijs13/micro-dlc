#include "ir/graph.h"
#include "optimizer/optimizer.h"
#include "codegen/codegen.h"
#include "simulator/simulator.h"
#include <iostream>
#include <memory>

using namespace dlcompiler;

void runEx() {
    
    auto graph = ir::Graph::create();
    auto input = graph->addInput({1, 3, 224, 224}); // [N, C, H, W]
    auto conv = graph->addConv2D(input, 64, 3, 1, 1); // 64 filters, 3x3 kernel
    auto relu = graph->addReLU(conv);
    auto output = graph->addOutput(relu);
    
    std::cout << "\nOriginal Graph:\n";
    graph->print();
    
    optimizer::Optimizer opt;
    opt.addPass(std::make_unique<optimizer::FusionPass>());
    opt.addPass(std::make_unique<optimizer::MemoryLayoutPass>());
    opt.run(graph.get());
    
    std::cout << "\nOptimized Graph:\n";
    graph->print();
    
    // Generate code
    codegen::CodeGenerator codegen;
    auto instructions = codegen.generate(graph.get());
    
    // simulate on different hardware configs
    simulator::ChipConfig high_end{
        .compute_units = 32,
        .memory_bandwidth_gb_s = 200,
        .cache_size_kb = 512,
        .simd_width = 16,
        .clock_freq_ghz = 2.0
    };
    simulator::Simulator sim1(high_end);
    auto stats1 = sim1.execute(instructions);
    
    simulator::ChipConfig low_end{
        .compute_units = 4,
        .memory_bandwidth_gb_s = 50,
        .cache_size_kb = 128,
        .simd_width = 4,
        .clock_freq_ghz = 1.0
    };
    simulator::Simulator sim2(low_end);
    auto stats2 = sim2.execute(instructions);
    
    std::cout << "\nSpeedup from high-end chip: " 
              << (stats2.execution_time_ms / stats1.execution_time_ms) << "x\n";
}

int main(int argc, char** argv) {
    
    try {
        runEx();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

