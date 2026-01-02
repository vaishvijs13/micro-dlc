#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace dlcompiler {
namespace ir {

// tensor shape rep
struct Shape {
    std::vector<int64_t> dims;
    
    Shape() = default;
    Shape(std::initializer_list<int64_t> d) : dims(d) {}
    
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
    
    std::string toString() const;
};

// op types
enum class OpType {
    INPUT,
    OUTPUT,
    CONV2D,
    MATMUL,
    RELU,
    ADD,
    MAXPOOL,
    BATCHNORM,
    FUSED_CONV_RELU, // optimized fused operation
    FUSED_MATMUL_ADD // optimized fused operation
};

std::string opTypeToString(OpType type);

// forward declarations
class Node;
class Graph;

// tensor value in graph
class Value {
public:
    Value(int id, const Shape& shape) : id_(id), shape_(shape) {}
    
    int id() const { return id_; }
    const Shape& shape() const { return shape_; }
    void setShape(const Shape& shape) { shape_ = shape; }
    
private:
    int id_;
    Shape shape_;
};

// operation node in computation graph
class Node {
public:
    Node(int id, OpType type) : id_(id), type_(type) {}
    
    int id() const { return id_; }
    OpType type() const { return type_; }
    void setType(OpType type) { type_ = type; }
    
    const std::vector<Value*>& inputs() const { return inputs_; }
    const std::vector<Value*>& outputs() const { return outputs_; }
    
    void addInput(Value* v) { inputs_.push_back(v); }
    void addOutput(Value* v) { outputs_.push_back(v); }
    
    void setAttr(const std::string& key, int64_t value) { 
        int_attrs_[key] = value; 
    }
    int64_t getAttr(const std::string& key, int64_t default_val = 0) const {
        auto it = int_attrs_.find(key);
        return it != int_attrs_.end() ? it->second : default_val;
    }
    
    const std::unordered_map<std::string, int64_t>& getAttrs() const {
        return int_attrs_;
    }
    
    std::string toString() const;
    
private:
    int id_;
    OpType type_;
    std::vector<Value*> inputs_;
    std::vector<Value*> outputs_;
    std::unordered_map<std::string, int64_t> int_attrs_;
};

// computation graph
class Graph {
public:
    static std::unique_ptr<Graph> create() {
        return std::make_unique<Graph>();
    }
    
    // add operations
    Value* addInput(const Shape& shape);
    Value* addOutput(Value* input);
    Value* addConv2D(Value* input, int64_t out_channels, int64_t kernel_size, 
                     int64_t stride, int64_t padding);
    Value* addMatMul(Value* a, Value* b);
    Value* addReLU(Value* input);
    Value* addAdd(Value* a, Value* b);
    Value* addMaxPool(Value* input, int64_t kernel_size, int64_t stride);
    
    std::vector<Node*> getNodes() const;
    std::vector<Node*> getNodesInTopoOrder() const;
    
    // graph q's
    int numNodes() const { return nodes_.size(); }
    int numValues() const { return values_.size(); }
    
    Node* getNode(int id) const { return nodes_[id].get(); }
    Value* getValue(int id) const { return values_[id].get(); }
    
    void print() const;
    
private:
    Node* createNode(OpType type);
    Value* createValue(const Shape& shape);
    
    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<std::unique_ptr<Value>> values_;
    int next_node_id_ = 0;
    int next_value_id_ = 0;
};

}
}

