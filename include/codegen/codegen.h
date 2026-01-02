#pragma once

#include "ir/graph.h"
#include <vector>
#include <string>

namespace dlcompiler {
namespace codegen {

enum class InstructionType {
    LOAD, //load data from mem
    STORE, //store data to mem
    COMPUTE, //perform computation
    SYNC //synchronization barrier
};

struct Instruction {
    InstructionType type;
    std::string op_name;
    int64_t input_size;
    int64_t output_size;
    int64_t flops;
    
    std::string toString() const;
};

// generate instruction sequences from IR
class CodeGenerator {
public:
    std::vector<Instruction> generate(ir::Graph* graph);
    
private:
    void generateForNode(ir::Node* node, std::vector<Instruction>& instructions);
    int64_t computeFLOPs(ir::Node* node);
};

}
}

