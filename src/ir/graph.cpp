#include "ir/graph.h"
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <iostream>

namespace dlcompiler {
namespace ir {

std::string Shape::toString() const {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        ss << dims[i];
        if (i < dims.size() - 1) ss << ", ";
    }
    ss << "]";
    return ss.str();
}

std::string opTypeToString(OpType type) {
    switch (type) {
        case OpType::INPUT: return "Input";
        case OpType::OUTPUT: return "Output";
        case OpType::CONV2D: return "Conv2D";
        case OpType::MATMUL: return "MatMul";
        case OpType::RELU: return "ReLU";
        case OpType::ADD: return "Add";
        case OpType::MAXPOOL: return "MaxPool";
        case OpType::BATCHNORM: return "BatchNorm";
        case OpType::FUSED_CONV_RELU: return "FusedConvReLU";
        case OpType::FUSED_MATMUL_ADD: return "FusedMatMulAdd";
        default: return "Unknown";
    }
}

std::string Node::toString() const {
    std::stringstream ss;
    ss << "Node" << id_ << " [" << opTypeToString(type_) << "]";
    ss << " inputs=[";
    for (size_t i = 0; i < inputs_.size(); ++i) {
        ss << "v" << inputs_[i]->id();
        if (i < inputs_.size() - 1) ss << ", ";
    }
    ss << "] outputs=[";
    for (size_t i = 0; i < outputs_.size(); ++i) {
        ss << "v" << outputs_[i]->id();
        if (i < outputs_.size() - 1) ss << ", ";
    }
    ss << "]";
    return ss.str();
}

Node* Graph::createNode(OpType type) {
    auto node = std::make_unique<Node>(next_node_id_++, type);
    auto* ptr = node.get();
    nodes_.push_back(std::move(node));
    return ptr;
}

Value* Graph::createValue(const Shape& shape) {
    auto value = std::make_unique<Value>(next_value_id_++, shape);
    auto* ptr = value.get();
    values_.push_back(std::move(value));
    return ptr;
}

Value* Graph::addInput(const Shape& shape) {
    auto* node = createNode(OpType::INPUT);
    auto* output = createValue(shape);
    node->addOutput(output);
    return output;
}

Value* Graph::addOutput(Value* input) {
    auto* node = createNode(OpType::OUTPUT);
    node->addInput(input);
    auto* output = createValue(input->shape());
    node->addOutput(output);
    return output;
}

Value* Graph::addConv2D(Value* input, int64_t out_channels, int64_t kernel_size,
                        int64_t stride, int64_t padding) {
    auto* node = createNode(OpType::CONV2D);
    node->addInput(input);
    node->setAttr("out_channels", out_channels);
    node->setAttr("kernel_size", kernel_size);
    node->setAttr("stride", stride);
    node->setAttr("padding", padding);
    
    // compute output shape: [N, C_out, H_out, W_out]
    const auto& in_shape = input->shape();
    int64_t h_out = (in_shape.dims[2] + 2 * padding - kernel_size) / stride + 1;
    int64_t w_out = (in_shape.dims[3] + 2 * padding - kernel_size) / stride + 1;
    Shape out_shape = {in_shape.dims[0], out_channels, h_out, w_out};
    
    auto* output = createValue(out_shape);
    node->addOutput(output);
    return output;
}

Value* Graph::addMatMul(Value* a, Value* b) {
    auto* node = createNode(OpType::MATMUL);
    node->addInput(a);
    node->addInput(b);
    
    // compute output shape: [M, N] x [N, K] = [M, K]
    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();
    Shape out_shape = {a_shape.dims[0], b_shape.dims[1]};
    
    auto* output = createValue(out_shape);
    node->addOutput(output);
    return output;
}

Value* Graph::addReLU(Value* input) {
    auto* node = createNode(OpType::RELU);
    node->addInput(input);
    auto* output = createValue(input->shape());
    node->addOutput(output);
    return output;
}

Value* Graph::addAdd(Value* a, Value* b) {
    auto* node = createNode(OpType::ADD);
    node->addInput(a);
    node->addInput(b);
    auto* output = createValue(a->shape());
    node->addOutput(output);
    return output;
}

Value* Graph::addMaxPool(Value* input, int64_t kernel_size, int64_t stride) {
    auto* node = createNode(OpType::MAXPOOL);
    node->addInput(input);
    node->setAttr("kernel_size", kernel_size);
    node->setAttr("stride", stride);
    
    const auto& in_shape = input->shape();
    int64_t h_out = (in_shape.dims[2] - kernel_size) / stride + 1;
    int64_t w_out = (in_shape.dims[3] - kernel_size) / stride + 1;
    Shape out_shape = {in_shape.dims[0], in_shape.dims[1], h_out, w_out};
    
    auto* output = createValue(out_shape);
    node->addOutput(output);
    return output;
}

std::vector<Node*> Graph::getNodes() const {
    std::vector<Node*> result;
    for (const auto& node : nodes_) {
        result.push_back(node.get());
    }
    return result;
}

std::vector<Node*> Graph::getNodesInTopoOrder() const {
    std::vector<Node*> result;
    std::unordered_set<int> visited;
    
    // simple sort
    for (const auto& node : nodes_) {
        if (visited.find(node->id()) == visited.end()) {
            result.push_back(node.get());
            visited.insert(node->id());
        }
    }
    
    return result;
}

void Graph::print() const {
    std::cout << "Graph with " << nodes_.size() << " nodes, " 
              << values_.size() << " values\n";
    for (const auto& node : nodes_) {
        std::cout << "  " << node->toString() << "\n";
    }
}

}
}

