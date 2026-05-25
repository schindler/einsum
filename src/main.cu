// Minimal interactive Einstein-summation playground.
//
// The executor intentionally supports a small, explicit set of patterns.
// cuTENSOR performs GPU permutation, reduction, and contraction work.

#include <cuda_runtime.h>
#include <cutensor.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kBinaryMagic[8] = {'E', 'I', 'N', 'S', 'U', 'M', '1', '\0'};
using Clock = std::chrono::steady_clock;

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error: ") +               \
                                     cudaGetErrorString(err__));                  \
        }                                                                         \
    } while (0)

#define CHECK_CUTENSOR(call)                                                      \
    do {                                                                          \
        cutensorStatus_t err__ = (call);                                          \
        if (err__ != CUTENSOR_STATUS_SUCCESS) {                                   \
            throw std::runtime_error(std::string("cuTENSOR error: ") +           \
                                     cutensorGetErrorString(err__));              \
        }                                                                         \
    } while (0)

struct Tensor {
    std::vector<int> dims;
    std::vector<float> values;
};

struct Stats {
    std::string operation;
    std::string backend;
    std::string output_name;
    std::vector<int> output_dims;
    std::size_t bytes_read = 0;
    std::size_t bytes_written = 0;
    std::uint64_t flops = 0;
    std::uint64_t workspace_bytes = 0;
    float elapsed_ms = 0.0f;
};

enum class GpuOpKind {
    Permute,
    Reduce,
    Contract,
};

struct Pattern {
    std::string text;
    std::vector<std::string> inputs;
    std::string output;
};

struct ResolvedOp {
    GpuOpKind kind = GpuOpKind::Contract;
    Pattern pattern;
    std::vector<int> output_dims;
    std::vector<int32_t> mode_a;
    std::vector<int32_t> mode_b;
    std::vector<int32_t> mode_out;
    std::uint64_t flops = 0;
};

const std::set<std::string>& supportedPatterns() {
    static const std::set<std::string> patterns = {
        "ij->ji",
        "ij->",
        "ij->j",
        "ij->i",
        "ij,jk->ik",
        "ik,kj->ij",
        "ij,j->i",
        "ik,k->i",
        "i,i->",
        "i,j->ij",
        "abc,cde->abde",
    };
    return patterns;
}

float elapsedMs(Clock::time_point start) {
    const auto end = Clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

std::size_t elementCount(const std::vector<int>& dims) {
    if (dims.empty()) {
        return 1;  // scalar
    }
    std::size_t n = 1;
    for (int d : dims) {
        if (d <= 0) {
            throw std::runtime_error("tensor dimension must be positive");
        }
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

std::string shapeString(const std::vector<int>& dims) {
    if (dims.empty()) {
        return "[]";
    }
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i != 0) {
            out << "x";
        }
        out << dims[i];
    }
    out << "]";
    return out.str();
}

std::string trim(const std::string& text) {
    std::size_t first = 0;
    while (first < text.size() && std::isspace(static_cast<unsigned char>(text[first]))) {
        ++first;
    }
    std::size_t last = text.size();
    while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1]))) {
        --last;
    }
    return text.substr(first, last - first);
}

std::vector<std::string> splitWords(const std::string& text) {
    std::istringstream in(text);
    std::vector<std::string> words;
    std::string word;
    while (in >> word) {
        words.push_back(word);
    }
    return words;
}

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> out;
    std::string current;
    for (char ch : text) {
        if (ch == delimiter) {
            out.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    out.push_back(current);
    return out;
}

bool endsWith(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void validateModeString(const std::string& mode) {
    std::set<char> seen;
    for (char label : mode) {
        if (!std::islower(static_cast<unsigned char>(label))) {
            throw std::runtime_error("mode labels must be lowercase letters");
        }
        if (!seen.insert(label).second) {
            throw std::runtime_error("repeated labels inside one tensor are not supported");
        }
    }
}

Pattern parsePattern(const std::string& text) {
    const std::size_t arrow = text.find("->");
    if (arrow == std::string::npos) {
        throw std::runtime_error("einsum pattern must contain ->");
    }
    if (supportedPatterns().count(text) == 0) {
        throw std::runtime_error("unsupported pattern: " + text);
    }

    Pattern p;
    p.text = text;
    p.inputs = split(text.substr(0, arrow), ',');
    p.output = text.substr(arrow + 2);
    if (p.inputs.empty() || p.inputs.size() > 2) {
        throw std::runtime_error("only unary and binary patterns are supported");
    }
    for (const std::string& input : p.inputs) {
        validateModeString(input);
    }
    validateModeString(p.output);
    return p;
}

std::vector<int32_t> modes(const std::string& labels) {
    std::vector<int32_t> out;
    out.reserve(labels.size());
    for (char label : labels) {
        out.push_back(static_cast<int32_t>(label));
    }
    return out;
}

const int32_t* modePtr(const std::vector<int32_t>& m) {
    return m.empty() ? nullptr : m.data();
}

std::vector<int64_t> toExtents(const std::vector<int>& dims) {
    std::vector<int64_t> out;
    out.reserve(dims.size());
    for (int dim : dims) {
        out.push_back(dim);
    }
    return out;
}

std::vector<int64_t> rowMajorStrides(const std::vector<int>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
        strides[static_cast<std::size_t>(i)] =
            strides[static_cast<std::size_t>(i + 1)] * dims[static_cast<std::size_t>(i + 1)];
    }
    return strides;
}

const int64_t* extentPtr(const std::vector<int64_t>& values) {
    return values.empty() ? nullptr : values.data();
}

std::vector<int> parseDimsToken(const std::string& token) {
    std::vector<int> dims;
    std::string current;
    for (char ch : token) {
        if (ch == 'x' || ch == 'X' || ch == ',') {
            if (current.empty()) {
                throw std::runtime_error("invalid dimension token: " + token);
            }
            dims.push_back(std::stoi(current));
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        dims.push_back(std::stoi(current));
    }
    if (dims.empty()) {
        throw std::runtime_error("empty dimension token");
    }
    elementCount(dims);
    return dims;
}

std::vector<int> parseShapeWords(const std::vector<std::string>& words, std::size_t begin) {
    if (begin >= words.size()) {
        throw std::runtime_error("missing shape");
    }
    std::vector<int> dims;
    if (words.size() == begin + 1 &&
        (words[begin].find('x') != std::string::npos ||
         words[begin].find('X') != std::string::npos ||
         words[begin].find(',') != std::string::npos)) {
        dims = parseDimsToken(words[begin]);
    } else {
        for (std::size_t i = begin; i < words.size(); ++i) {
            dims.push_back(std::stoi(words[i]));
        }
    }
    if (dims.empty() || dims.size() > 2) {
        throw std::runtime_error("random/ones/zeros support vector or matrix shapes");
    }
    elementCount(dims);
    return dims;
}

Tensor loadTextTensor(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("could not open " + path);
    }

    std::vector<std::string> tokens;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        std::vector<std::string> words = splitWords(line);
        tokens.insert(tokens.end(), words.begin(), words.end());
    }
    if (tokens.empty()) {
        throw std::runtime_error("empty tensor file: " + path);
    }

    std::size_t pos = 0;
    if (tokens[pos] == "dims") {
        ++pos;
    }
    if (pos >= tokens.size()) {
        throw std::runtime_error("missing tensor dimensions in " + path);
    }

    Tensor t;
    if (tokens[pos] == "scalar") {
        ++pos;
    } else if (tokens[pos].find('x') != std::string::npos ||
        tokens[pos].find('X') != std::string::npos ||
        tokens[pos].find(',') != std::string::npos) {
        t.dims = parseDimsToken(tokens[pos++]);
    } else {
        while (pos < tokens.size()) {
            char* end = nullptr;
            const long value = std::strtol(tokens[pos].c_str(), &end, 10);
            if (*end != '\0' || value <= 0) {
                break;
            }
            t.dims.push_back(static_cast<int>(value));
            ++pos;
            const std::size_t needed = elementCount(t.dims);
            const std::size_t remaining = tokens.size() - pos;
            if (!t.dims.empty() && remaining == needed) {
                break;
            }
        }
    }

    const std::size_t expected = elementCount(t.dims);
    t.values.reserve(expected);
    for (; pos < tokens.size(); ++pos) {
        t.values.push_back(std::stof(tokens[pos]));
    }
    if (t.values.size() != expected) {
        std::ostringstream msg;
        msg << path << " has " << t.values.size() << " values but shape "
            << shapeString(t.dims) << " needs " << expected;
        throw std::runtime_error(msg.str());
    }
    return t;
}

std::uint32_t readU32(std::ifstream& in) {
    std::uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("truncated binary tensor");
    }
    return value;
}

std::uint64_t readU64(std::ifstream& in) {
    std::uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("truncated binary tensor");
    }
    return value;
}

Tensor loadBinaryTensor(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("could not open " + path);
    }
    char magic[sizeof(kBinaryMagic)];
    in.read(magic, sizeof(magic));
    if (!in || !std::equal(std::begin(kBinaryMagic), std::end(kBinaryMagic), magic)) {
        throw std::runtime_error("invalid binary tensor magic in " + path);
    }

    Tensor t;
    const std::uint32_t rank = readU32(in);
    if (rank > 8) {
        throw std::runtime_error("invalid tensor rank in " + path);
    }
    t.dims.resize(rank);
    for (std::uint32_t i = 0; i < rank; ++i) {
        const std::uint64_t dim = readU64(in);
        if (dim == 0 || dim > static_cast<std::uint64_t>(1 << 30)) {
            throw std::runtime_error("invalid tensor dimension in " + path);
        }
        t.dims[i] = static_cast<int>(dim);
    }

    t.values.resize(elementCount(t.dims));
    in.read(reinterpret_cast<char*>(t.values.data()),
            static_cast<std::streamsize>(t.values.size() * sizeof(float)));
    if (!in) {
        throw std::runtime_error("truncated binary tensor values in " + path);
    }
    return t;
}

Tensor loadTensor(const std::string& path) {
    if (endsWith(path, ".bin")) {
        return loadBinaryTensor(path);
    }
    return loadTextTensor(path);
}

void saveTextTensor(const Tensor& t, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("could not write " + path);
    }
    out << "dims ";
    if (t.dims.empty()) {
        out << "scalar\n";
    } else {
        for (std::size_t i = 0; i < t.dims.size(); ++i) {
            if (i != 0) {
                out << "x";
            }
            out << t.dims[i];
        }
        out << "\n";
    }
    out << std::setprecision(8);
    for (std::size_t i = 0; i < t.values.size(); ++i) {
        out << t.values[i] << ((i + 1 == t.values.size()) ? '\n' : ' ');
    }
}

void saveBinaryTensor(const Tensor& t, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("could not write " + path);
    }
    out.write(kBinaryMagic, sizeof(kBinaryMagic));
    const std::uint32_t rank = static_cast<std::uint32_t>(t.dims.size());
    out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    for (int dim : t.dims) {
        const std::uint64_t value = static_cast<std::uint64_t>(dim);
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    out.write(reinterpret_cast<const char*>(t.values.data()),
              static_cast<std::streamsize>(t.values.size() * sizeof(float)));
}

void saveTensor(const Tensor& t, const std::string& path) {
    if (endsWith(path, ".bin")) {
        saveBinaryTensor(t, path);
    } else {
        saveTextTensor(t, path);
    }
}

void printTensor(const Tensor& t) {
    std::cout << "shape=" << shapeString(t.dims) << "\n";
    if (t.dims.empty()) {
        std::cout << t.values[0] << "\n";
        return;
    }
    if (t.dims.size() == 2) {
        const int rows = t.dims[0];
        const int cols = t.dims[1];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << std::setw(10) << t.values[static_cast<std::size_t>(i) * cols + j];
            }
            std::cout << "\n";
        }
        return;
    }

    for (std::size_t i = 0; i < t.values.size(); ++i) {
        std::cout << t.values[i] << ((i + 1 == t.values.size()) ? '\n' : ' ');
    }
}

Tensor makeMatrix(const std::vector<std::string>& words) {
    if (words.size() < 4) {
        throw std::runtime_error("usage: A = matrix rows cols values...");
    }
    Tensor t;
    t.dims = {std::stoi(words[1]), std::stoi(words[2])};
    const std::size_t expected = elementCount(t.dims);
    if (words.size() != expected + 3) {
        throw std::runtime_error("matrix value count does not match rows * cols");
    }
    t.values.reserve(expected);
    for (std::size_t i = 3; i < words.size(); ++i) {
        t.values.push_back(std::stof(words[i]));
    }
    return t;
}

Tensor makeTensor(const std::vector<std::string>& words) {
    if (words.size() < 4) {
        throw std::runtime_error("usage: A = tensor rank dim... values...");
    }
    const int rank = std::stoi(words[1]);
    if (rank < 0 || rank > 8 || words.size() < static_cast<std::size_t>(2 + rank)) {
        throw std::runtime_error("invalid tensor rank");
    }

    Tensor t;
    for (int i = 0; i < rank; ++i) {
        t.dims.push_back(std::stoi(words[2 + i]));
    }
    const std::size_t expected = elementCount(t.dims);
    const std::size_t values_begin = 2 + static_cast<std::size_t>(rank);
    if (words.size() != values_begin + expected) {
        throw std::runtime_error("tensor value count does not match shape");
    }
    t.values.reserve(expected);
    for (std::size_t i = values_begin; i < words.size(); ++i) {
        t.values.push_back(std::stof(words[i]));
    }
    return t;
}

Tensor filledTensor(const std::vector<int>& dims, float value) {
    Tensor t;
    t.dims = dims;
    t.values.assign(elementCount(dims), value);
    return t;
}

Tensor randomTensor(const std::vector<int>& dims) {
    static std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor t;
    t.dims = dims;
    t.values.resize(elementCount(dims));
    for (float& value : t.values) {
        value = dist(rng);
    }
    return t;
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) {
        reset(count);
    }
    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void reset(std::size_t count) {
        if (ptr_) {
            CHECK_CUDA(cudaFree(ptr_));
            ptr_ = nullptr;
        }
        count_ = count;
        if (count_ != 0) {
            CHECK_CUDA(cudaMalloc(&ptr_, count_ * sizeof(float)));
        }
    }

    void upload(const std::vector<float>& host) {
        if (host.size() != count_) {
            throw std::runtime_error("device upload size mismatch");
        }
        CHECK_CUDA(cudaMemcpy(ptr_, host.data(), count_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    void zero() {
        CHECK_CUDA(cudaMemset(ptr_, 0, count_ * sizeof(float)));
    }

    void download(std::vector<float>& host) const {
        if (host.size() != count_) {
            host.resize(count_);
        }
        CHECK_CUDA(cudaMemcpy(host.data(), ptr_, count_ * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float* get() {
        return ptr_;
    }

private:
    float* ptr_ = nullptr;
    std::size_t count_ = 0;
};

class DeviceBytes {
public:
    explicit DeviceBytes(std::uint64_t bytes) {
        if (bytes != 0) {
            CHECK_CUDA(cudaMalloc(&ptr_, bytes));
        }
    }
    ~DeviceBytes() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;

    void* get() {
        return ptr_;
    }

private:
    void* ptr_ = nullptr;
};

class CutensorHandle {
public:
    CutensorHandle() {
        CHECK_CUTENSOR(cutensorInit(&handle_));
    }
    CutensorHandle(const CutensorHandle&) = delete;
    CutensorHandle& operator=(const CutensorHandle&) = delete;

    const cutensorHandle_t* get() const {
        return &handle_;
    }

private:
    cutensorHandle_t handle_{};
};

class TensorDescriptor {
public:
    TensorDescriptor(const cutensorHandle_t* handle, const std::vector<int>& dims) {
        extents_ = toExtents(dims);
        strides_ = rowMajorStrides(dims);
        CHECK_CUTENSOR(cutensorInitTensorDescriptor(handle,
                                                    &desc_,
                                                    static_cast<uint32_t>(dims.size()),
                                                    extentPtr(extents_),
                                                    extentPtr(strides_),
                                                    CUDA_R_32F,
                                                    CUTENSOR_OP_IDENTITY));
    }
    TensorDescriptor(const TensorDescriptor&) = delete;
    TensorDescriptor& operator=(const TensorDescriptor&) = delete;

    const cutensorTensorDescriptor_t* get() const {
        return &desc_;
    }

    std::uint32_t alignmentFor(const cutensorHandle_t* handle, const void* ptr) const {
        uint32_t alignment = 0;
        CHECK_CUTENSOR(cutensorGetAlignmentRequirement(handle, ptr, &desc_, &alignment));
        return alignment;
    }

private:
    cutensorTensorDescriptor_t desc_{};
    std::vector<int64_t> extents_;
    std::vector<int64_t> strides_;
};

ResolvedOp resolvePattern(const std::string& pattern_text, const std::vector<const Tensor*>& inputs) {
    Pattern p = parsePattern(pattern_text);
    if (p.inputs.size() != inputs.size()) {
        throw std::runtime_error("pattern input count does not match command");
    }

    std::map<char, int> extents;
    for (std::size_t tensor_idx = 0; tensor_idx < inputs.size(); ++tensor_idx) {
        const std::string& mode = p.inputs[tensor_idx];
        const Tensor& tensor = *inputs[tensor_idx];
        if (mode.size() != tensor.dims.size()) {
            throw std::runtime_error("tensor rank does not match pattern");
        }
        for (std::size_t i = 0; i < mode.size(); ++i) {
            const char label = mode[i];
            const int dim = tensor.dims[i];
            const auto existing = extents.find(label);
            if (existing != extents.end() && existing->second != dim) {
                throw std::runtime_error("shared einsum dimension mismatch");
            }
            extents[label] = dim;
        }
    }

    ResolvedOp op;
    op.pattern = p;
    op.mode_a = modes(p.inputs[0]);
    if (p.inputs.size() == 2) {
        op.mode_b = modes(p.inputs[1]);
    }
    op.mode_out = modes(p.output);

    for (char label : p.output) {
        const auto found = extents.find(label);
        if (found == extents.end()) {
            throw std::runtime_error("output label does not exist in input tensors");
        }
        op.output_dims.push_back(found->second);
    }

    if (p.inputs.size() == 1) {
        op.kind = (p.inputs[0].size() == p.output.size()) ? GpuOpKind::Permute : GpuOpKind::Reduce;
        if (op.kind == GpuOpKind::Reduce) {
            const std::size_t output_elements = elementCount(op.output_dims);
            op.flops = (output_elements == 0) ? 0 : elementCount(inputs[0]->dims) - output_elements;
        }
        return op;
    }

    op.kind = GpuOpKind::Contract;
    std::set<char> output_labels(p.output.begin(), p.output.end());
    std::set<char> all_input_labels;
    for (const std::string& input : p.inputs) {
        all_input_labels.insert(input.begin(), input.end());
    }
    std::uint64_t contracted_extent = 1;
    for (char label : all_input_labels) {
        if (output_labels.count(label) == 0) {
            contracted_extent *= static_cast<std::uint64_t>(extents[label]);
        }
    }
    const std::uint64_t output_elements = static_cast<std::uint64_t>(elementCount(op.output_dims));
    op.flops = (contracted_extent == 1) ? output_elements : (2ull * output_elements * contracted_extent);
    return op;
}

Stats runEinsum(const std::string& pattern_text,
                const std::string& output_name,
                const std::vector<const Tensor*>& inputs,
                Tensor* output) {
    const ResolvedOp op = resolvePattern(pattern_text, inputs);
    output->dims = op.output_dims;
    output->values.assign(elementCount(output->dims), 0.0f);

    DeviceBuffer d_a(inputs[0]->values.size());
    DeviceBuffer d_b(inputs.size() == 2 ? inputs[1]->values.size() : 0);
    DeviceBuffer d_out(output->values.size());
    d_a.upload(inputs[0]->values);
    if (inputs.size() == 2) {
        d_b.upload(inputs[1]->values);
    }
    d_out.zero();

    CutensorHandle cutensor;
    const cutensorHandle_t* handle = cutensor.get();
    TensorDescriptor desc_a(handle, inputs[0]->dims);
    TensorDescriptor desc_out(handle, output->dims);

    std::unique_ptr<TensorDescriptor> desc_b;
    if (inputs.size() == 2) {
        desc_b.reset(new TensorDescriptor(handle, inputs[1]->dims));
    }

    std::uint64_t workspace_bytes = 0;
    cutensorContractionDescriptor_t contraction_desc{};
    cutensorContractionFind_t contraction_find{};
    cutensorContractionPlan_t contraction_plan{};

    if (op.kind == GpuOpKind::Reduce) {
        CHECK_CUTENSOR(cutensorReductionGetWorkspace(handle,
                                                     d_a.get(),
                                                     desc_a.get(),
                                                     modePtr(op.mode_a),
                                                     d_out.get(),
                                                     desc_out.get(),
                                                     modePtr(op.mode_out),
                                                     d_out.get(),
                                                     desc_out.get(),
                                                     modePtr(op.mode_out),
                                                     CUTENSOR_OP_ADD,
                                                     CUTENSOR_COMPUTE_32F,
                                                     &workspace_bytes));
    } else if (op.kind == GpuOpKind::Contract) {
        const uint32_t alignment_a = desc_a.alignmentFor(handle, d_a.get());
        const uint32_t alignment_b = desc_b->alignmentFor(handle, d_b.get());
        const uint32_t alignment_out = desc_out.alignmentFor(handle, d_out.get());

        CHECK_CUTENSOR(cutensorInitContractionDescriptor(handle,
                                                         &contraction_desc,
                                                         desc_a.get(),
                                                         modePtr(op.mode_a),
                                                         alignment_a,
                                                         desc_b->get(),
                                                         modePtr(op.mode_b),
                                                         alignment_b,
                                                         desc_out.get(),
                                                         modePtr(op.mode_out),
                                                         alignment_out,
                                                         desc_out.get(),
                                                         modePtr(op.mode_out),
                                                         alignment_out,
                                                         CUTENSOR_COMPUTE_32F));
        CHECK_CUTENSOR(cutensorInitContractionFind(handle,
                                                   &contraction_find,
                                                   CUTENSOR_ALGO_DEFAULT));
        CHECK_CUTENSOR(cutensorContractionGetWorkspace(handle,
                                                       &contraction_desc,
                                                       &contraction_find,
                                                       CUTENSOR_WORKSPACE_RECOMMENDED,
                                                       &workspace_bytes));
        CHECK_CUTENSOR(cutensorInitContractionPlan(handle,
                                                   &contraction_plan,
                                                   &contraction_desc,
                                                   &contraction_find,
                                                   workspace_bytes));
    }

    DeviceBytes workspace(workspace_bytes);

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (op.kind == GpuOpKind::Permute) {
        CHECK_CUTENSOR(cutensorPermutation(handle,
                                           &alpha,
                                           d_a.get(),
                                           desc_a.get(),
                                           modePtr(op.mode_a),
                                           d_out.get(),
                                           desc_out.get(),
                                           modePtr(op.mode_out),
                                           CUDA_R_32F,
                                           stream));
    } else if (op.kind == GpuOpKind::Reduce) {
        CHECK_CUTENSOR(cutensorReduction(handle,
                                         &alpha,
                                         d_a.get(),
                                         desc_a.get(),
                                         modePtr(op.mode_a),
                                         &beta,
                                         d_out.get(),
                                         desc_out.get(),
                                         modePtr(op.mode_out),
                                         d_out.get(),
                                         desc_out.get(),
                                         modePtr(op.mode_out),
                                         CUTENSOR_OP_ADD,
                                         CUTENSOR_COMPUTE_32F,
                                         workspace.get(),
                                         workspace_bytes,
                                         stream));
    } else {
        CHECK_CUTENSOR(cutensorContraction(handle,
                                           &contraction_plan,
                                           &alpha,
                                           d_a.get(),
                                           d_b.get(),
                                           &beta,
                                           d_out.get(),
                                           d_out.get(),
                                           workspace.get(),
                                           workspace_bytes,
                                           stream));
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float gpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));

    d_out.download(output->values);

    std::size_t input_bytes = inputs[0]->values.size() * sizeof(float);
    if (inputs.size() == 2) {
        input_bytes += inputs[1]->values.size() * sizeof(float);
    }

    Stats stats;
    stats.operation = pattern_text;
    stats.backend = "cuTENSOR";
    stats.output_name = output_name;
    stats.output_dims = output->dims;
    stats.bytes_read = input_bytes;
    stats.bytes_written = output->values.size() * sizeof(float);
    stats.flops = op.flops;
    stats.workspace_bytes = workspace_bytes;
    stats.elapsed_ms = gpu_ms;
    return stats;
}

void printStats(const Stats& stats) {
    std::cout << "operation=" << stats.operation << "\n"
              << "backend=" << stats.backend << "\n"
              << "output=" << stats.output_name << " shape=" << shapeString(stats.output_dims) << "\n"
              << "bytes_read=" << stats.bytes_read << "\n"
              << "bytes_written=" << stats.bytes_written << "\n"
              << "flops=" << stats.flops << "\n"
              << "workspace_bytes=" << stats.workspace_bytes << "\n"
              << "elapsed_ms=" << std::fixed << std::setprecision(4) << stats.elapsed_ms << "\n";
}

Stats hostStats(const std::string& operation,
                const std::string& output_name,
                const Tensor& tensor,
                std::size_t bytes_read,
                float elapsed_ms) {
    Stats stats;
    stats.operation = operation;
    stats.backend = "host";
    stats.output_name = output_name;
    stats.output_dims = tensor.dims;
    stats.bytes_read = bytes_read;
    stats.bytes_written = tensor.values.size() * sizeof(float);
    stats.elapsed_ms = elapsed_ms;
    return stats;
}

void printHelp() {
    std::cout
        << "Commands:\n"
        << "  A = load data/A.txt\n"
        << "  A = matrix 2 3 1 2 3 4 5 6\n"
        << "  v = tensor 1 3 1 2 3\n"
        << "  A = random 2 3\n"
        << "  A = random 2x3\n"
        << "  A = ones 2 3\n"
        << "  A = zeros 2 3\n"
        << "  show A\n"
        << "  show A --output=data/A.bin\n"
        << "  show A --outpu=data/A.bin\n"
        << "  tensors\n"
        << "  help\n"
        << "  quit\n\n"
        << "Supported einsum examples:\n"
        << "  T = ij->ji A              # transpose matrix\n"
        << "  s = ij-> A                # sum all matrix values\n"
        << "  c = ij->j A               # column sum\n"
        << "  r = ij->i A               # row sum\n"
        << "  C = ij,jk->ik A B         # matrix multiplication\n"
        << "  C = ik,kj->ij A B         # matrix multiplication with alternate labels\n"
        << "  y = ij,j->i A x           # matrix-vector multiplication\n"
        << "  y = ik,k->i A x           # matrix-vector multiplication with alternate labels\n"
        << "  d = i,i-> x y             # dot product\n"
        << "  O = i,j->ij x y           # outer product\n"
        << "  D = abc,cde->abde A B     # rank-3 contraction\n"
        << "  D = einsum abc,cde->abde A B\n";
}

void listTensors(const std::map<std::string, Tensor>& tensors) {
    if (tensors.empty()) {
        std::cout << "no tensors loaded\n";
        return;
    }
    for (const auto& item : tensors) {
        std::cout << item.first << " shape=" << shapeString(item.second.dims)
                  << " elements=" << item.second.values.size() << "\n";
    }
}

std::vector<const Tensor*> getInputs(const Pattern& pattern,
                                     const std::vector<std::string>& names,
                                     const std::map<std::string, Tensor>& tensors) {
    if (names.size() != pattern.inputs.size()) {
        throw std::runtime_error("wrong number of tensors for pattern");
    }
    std::vector<const Tensor*> inputs;
    for (const std::string& name : names) {
        const auto found = tensors.find(name);
        if (found == tensors.end()) {
            throw std::runtime_error("input tensor not found: " + name);
        }
        inputs.push_back(&found->second);
    }
    return inputs;
}

void handleAssignment(const std::string& name,
                      const std::string& expression,
                      std::map<std::string, Tensor>* tensors) {
    std::vector<std::string> words = splitWords(expression);
    if (words.empty()) {
        throw std::runtime_error("empty assignment");
    }

    if (words[0] == "load") {
        if (words.size() != 2) {
            throw std::runtime_error("usage: A = load path");
        }
        const Clock::time_point start = Clock::now();
        Tensor tensor = loadTensor(words[1]);
        const float ms = elapsedMs(start);
        (*tensors)[name] = tensor;
        printStats(hostStats("load", name, (*tensors)[name], 0, ms));
        return;
    }

    if (words[0] == "matrix") {
        const Clock::time_point start = Clock::now();
        Tensor tensor = makeMatrix(words);
        const float ms = elapsedMs(start);
        (*tensors)[name] = tensor;
        printStats(hostStats("matrix", name, (*tensors)[name], 0, ms));
        return;
    }

    if (words[0] == "tensor") {
        const Clock::time_point start = Clock::now();
        Tensor tensor = makeTensor(words);
        const float ms = elapsedMs(start);
        (*tensors)[name] = tensor;
        printStats(hostStats("tensor", name, (*tensors)[name], 0, ms));
        return;
    }

    if (words[0] == "random" || words[0] == "ones" || words[0] == "zeros") {
        const Clock::time_point start = Clock::now();
        const std::vector<int> dims = parseShapeWords(words, 1);
        Tensor tensor;
        if (words[0] == "random") {
            tensor = randomTensor(dims);
        } else if (words[0] == "ones") {
            tensor = filledTensor(dims, 1.0f);
        } else {
            tensor = filledTensor(dims, 0.0f);
        }
        const float ms = elapsedMs(start);
        (*tensors)[name] = tensor;
        printStats(hostStats(words[0], name, (*tensors)[name], 0, ms));
        return;
    }

    std::string pattern_text;
    std::vector<std::string> input_names;
    if (words[0] == "einsum") {
        if (words.size() < 3) {
            throw std::runtime_error("usage: C = einsum pattern A [B]");
        }
        pattern_text = words[1];
        input_names.assign(words.begin() + 2, words.end());
    } else if (words[0].find("->") != std::string::npos) {
        pattern_text = words[0];
        input_names.assign(words.begin() + 1, words.end());
    } else {
        throw std::runtime_error("unknown assignment expression: " + expression);
    }

    const Pattern pattern = parsePattern(pattern_text);
    const std::vector<const Tensor*> inputs = getInputs(pattern, input_names, *tensors);
    Tensor output;
    const Stats stats = runEinsum(pattern_text, name, inputs, &output);
    (*tensors)[name] = output;
    printStats(stats);
}

void handleShow(const std::vector<std::string>& words, const std::map<std::string, Tensor>& tensors) {
    if (words.size() < 2) {
        throw std::runtime_error("usage: show A [--output=file]");
    }
    const std::string& name = words[1];
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("tensor not found: " + name);
    }

    std::string output_path;
    for (std::size_t i = 2; i < words.size(); ++i) {
        const std::string prefix1 = "--output=";
        const std::string prefix2 = "--outpu=";
        if (words[i].compare(0, prefix1.size(), prefix1) == 0) {
            output_path = words[i].substr(prefix1.size());
        } else if (words[i].compare(0, prefix2.size(), prefix2) == 0) {
            output_path = words[i].substr(prefix2.size());
        } else {
            throw std::runtime_error("unknown show option: " + words[i]);
        }
    }

    if (!output_path.empty()) {
        const Clock::time_point start = Clock::now();
        saveTensor(it->second, output_path);
        printStats(hostStats("save", name, it->second, it->second.values.size() * sizeof(float), elapsedMs(start)));
    } else {
        printTensor(it->second);
    }
}

void initializeLibraries() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("no CUDA device found");
    }
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDeviceProperties(&props, device));

    CutensorHandle cutensor;
    (void)cutensor.get();

    std::cout << "GPU " << device << ": " << props.name << "\n"
              << "cuTENSOR version=" << cutensorGetVersion() << "\n";
}

void runPrompt() {
    std::map<std::string, Tensor> tensors;

    initializeLibraries();
    std::cout << "Minimal Einstein-summation prompt\n";
    std::cout << "Type help for commands.\n";

    std::string line;
    while (true) {
        std::cout << "einsum> " << std::flush;
        if (!std::getline(std::cin, line)) {
            std::cout << "\n";
            return;
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        try {
            if (line == "quit" || line == "exit") {
                return;
            }
            if (line == "help") {
                printHelp();
                continue;
            }
            if (line == "tensors") {
                listTensors(tensors);
                continue;
            }

            const std::vector<std::string> words = splitWords(line);
            if (!words.empty() && words[0] == "show") {
                handleShow(words, tensors);
                continue;
            }

            const std::size_t eq = line.find('=');
            if (eq == std::string::npos) {
                throw std::runtime_error("expected assignment or command");
            }
            const std::string name = trim(line.substr(0, eq));
            const std::string expression = trim(line.substr(eq + 1));
            if (name.empty() || name.find(' ') != std::string::npos) {
                throw std::runtime_error("invalid tensor name");
            }
            handleAssignment(name, expression, &tensors);
        } catch (const std::exception& e) {
            std::cerr << "error: " << e.what() << "\n";
        }
    }
}

}  // namespace

int main() {
    try {
        runPrompt();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
