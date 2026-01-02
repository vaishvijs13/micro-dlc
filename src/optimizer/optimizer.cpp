#include "optimizer/optimizer.h"
#include <iostream>
#include <unordered_set>

namespace dlcompiler {
namespace optimizer {

void Optimizer::run(ir::Graph* graph) {
    std::cout << "\n ----> Running Optimization Passes <----\n";
    for (auto& pass : passes_) {
        std::cout << "Running " << pass->name() << "...\n";
        bool changed = pass->run(graph);
        std::cout << "  " << (changed ? "Modified graph" : "No changes") << "\n";
    }
    std::cout << " ----> Optimization Complete <----\n\n";
}

bool FusionPass::run(ir::Graph* graph) {
    bool changed = false;
    changed |= fuseConvReLU(graph);
    changed |= fuseMatMulAdd(graph);
    return changed;
}

bool FusionPass::fuseConvReLU(ir::Graph* graph) {
    bool changed = false;
    auto nodes = graph->getNodesInTopoOrder();
    
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto* node = nodes[i];
        
        // look for Conv2D followed by ReLU
        if (node->type() == ir::OpType::CONV2D && node->outputs().size() == 1) {
            auto* conv_output = node->outputs()[0];
            
            // find nodes that use this output
            for (size_t j = i + 1; j < nodes.size(); ++j) {
                auto* next_node = nodes[j];
                
                if (next_node->type() == ir::OpType::RELU && 
                    next_node->inputs().size() == 1 &&
                    next_node->inputs()[0] == conv_output) {
                    
                    // fuse! change Conv2D to FusedConvReLU
                    node->setType(ir::OpType::FUSED_CONV_RELU);
                    std::cout << "  Fused Conv2D + ReLU into FusedConvReLU\n";
                    changed = true;
                    
                    break;
                }
            }
        }
    }
    
    return changed;
}

bool FusionPass::fuseMatMulAdd(ir::Graph* graph) {
    bool changed = false;
    auto nodes = graph->getNodesInTopoOrder();
    
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto* node = nodes[i];
        
        // look for MatMul followed by add
        if (node->type() == ir::OpType::MATMUL && node->outputs().size() == 1) {
            auto* matmul_output = node->outputs()[0];
            
            for (size_t j = i + 1; j < nodes.size(); ++j) {
                auto* next_node = nodes[j];
                
                if (next_node->type() == ir::OpType::ADD && 
                    next_node->inputs().size() == 2 &&
                    (next_node->inputs()[0] == matmul_output || 
                     next_node->inputs()[1] == matmul_output)) {
                    
                    // fuse! change MatMul to FusedMatMulAdd
                    node->setType(ir::OpType::FUSED_MATMUL_ADD);
                    std::cout << "  Fused MatMul + Add into FusedMatMulAdd\n";
                    changed = true;
                    break;
                }
            }
        }
    }
    
    return changed;
}

}
}

