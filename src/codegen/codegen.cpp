#include "codegen/codegen.h"
#include <sstream>
#include <iostream>

namespace dlcompiler {
namespace codegen {

std::string Instruction::toString() const {
    std::stringstream ss;
    ss << "Instruction{";
    switch (type) {
        case InstructionType::LOAD: ss << "LOAD"; break;
        case InstructionType::STORE: ss << "STORE"; break;
        case InstructionType::COMPUTE: ss << "COMPUTE"; break;
        case InstructionType::SYNC: ss << "SYNC"; break;
    }
    ss << ", op=" << op_name;
    ss << ", in=" << input_size << "B";
    ss << ", out=" << output_size << "B";
    ss << ", flops=" << flops << "}";
    return ss.str();
}

std::vector<Instruction> CodeGenerator::generate(ir::Graph* graph) {
    std::vector<Instruction> instructions;
    
    std::cout << "\n ----> Code Generation <----\n";
    
    auto nodes = graph->getNodesInTopoOrder();
    for (auto* node : nodes) {
        generateForNode(node, instructions);
    }
    
    std::cout << "Generated " << instructions.size() << " instructions\n";
    std::cout << " ----> Code Generation Complete <----\n\n";
    
    return instructions;
}

void CodeGenerator::generateForNode(ir::Node* node, std::vector<Instruction>& instructions) {
    // skip in and out nodes
    if (node->type() == ir::OpType::INPUT || node->type() == ir::OpType::OUTPUT) {
        return;
    }
    
    // calc i/o sizes
    int64_t input_size = 0;
    for (auto* input : node->inputs()) {
        input_size += input->shape().numel() * sizeof(float);
    }
    
    int64_t output_size = 0;
    for (auto* output : node->outputs()) {
        output_size += output->shape().numel() * sizeof(float);
    }
    
    // gen LOAD instruction
    instructions.push_back({
        InstructionType::LOAD,
        ir::opTypeToString(node->type()),
        input_size,
        0,
        0
    });
    
    // gen COMPUTE instruction
    int64_t flops = computeFLOPs(node);
    instructions.push_back({
        InstructionType::COMPUTE,
        ir::opTypeToString(node->type()),
        input_size,
        output_size,
        flops
    });
    
    // gen STORE instruction
    instructions.push_back({
        InstructionType::STORE,
        ir::opTypeToString(node->type()),
        0,
        output_size,
        0
    });
}

int64_t CodeGenerator::computeFLOPs(ir::Node* node) {
    switch (node->type()) {
        case ir::OpType::CONV2D:
        case ir::OpType::FUSED_CONV_RELU: {
            // flops = 2 * C_in * K * K * C_out * H_out * W_out * N
            auto* output = node->outputs()[0];
            auto* input = node->inputs()[0];
            int64_t k = node->getAttr("kernel_size", 3);
            int64_t c_in = input->shape().dims[1];
            int64_t c_out = output->shape().dims[1];
            int64_t h_out = output->shape().dims[2];
            int64_t w_out = output->shape().dims[3];
            int64_t n = output->shape().dims[0];
            return 2 * c_in * k * k * c_out * h_out * w_out * n;
        }
        
        case ir::OpType::MATMUL:
        case ir::OpType::FUSED_MATMUL_ADD: {
            // flops = 2 * M * N * K
            auto* a = node->inputs()[0];
            auto* b = node->inputs()[1];
            int64_t m = a->shape().dims[0];
            int64_t k = a->shape().dims[1];
            int64_t n = b->shape().dims[1];
            return 2 * m * n * k;
        }
        
        case ir::OpType::RELU: {
            auto* output = node->outputs()[0];
            return output->shape().numel();
        }
        
        case ir::OpType::ADD: {
            auto* output = node->outputs()[0];
            return output->shape().numel();
        }
        
        case ir::OpType::MAXPOOL: {
            auto* output = node->outputs()[0];
            int64_t k = node->getAttr("kernel_size", 2);
            return output->shape().numel() * k * k;
        }
        
        default:
            return 0;
    }
}

}
}

