#pragma once

#include "ir/graph.h"
#include <memory>
#include <vector>

namespace dlcompiler {
namespace optimizer {

class Pass {
public:
    virtual ~Pass() = default;
    virtual bool run(ir::Graph* graph) = 0;
    virtual std::string name() const = 0;
};

// fuse consecutive ops
class FusionPass : public Pass {
public:
    bool run(ir::Graph* graph) override;
    std::string name() const override { return "FusionPass"; }
    
private:
    bool fuseConvReLU(ir::Graph* graph);
    bool fuseMatMulAdd(ir::Graph* graph);
};

class MemoryLayoutPass : public Pass {
public:
    bool run(ir::Graph* graph) override;
    std::string name() const override { return "MemoryLayoutPass"; }
};

// remove unused ops
class DeadCodeEliminationPass : public Pass {
public:
    bool run(ir::Graph* graph) override;
    std::string name() const override { return "DeadCodeEliminationPass"; }
};

// manage and run optimization passes
class Optimizer {
public:
    void addPass(std::unique_ptr<Pass> pass) {
        passes_.push_back(std::move(pass));
    }
    
    void run(ir::Graph* graph);
    
private:
    std::vector<std::unique_ptr<Pass>> passes_;
};

}
}

