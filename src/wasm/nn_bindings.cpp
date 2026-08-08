/// WASM bindings for the neural network educational modules.
/// Exposes tokenizer, embedding, attention, matmul, softmax, activations
/// to the JavaScript frontend for step-by-step visualization.

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#endif

#include <inferx/core/tracer.h>
#include <inferx/nn/tokenizer.h>
#include <inferx/nn/embedding.h>
#include <inferx/nn/attention.h>
#include <inferx/nn/matmul.h>
#include <inferx/nn/softmax.h>
#include <inferx/nn/activations.h>

#include <vector>
#include <string>

using namespace inferx;

// Global instances
static nn::Tokenizer g_tokenizer;
static std::unique_ptr<nn::Embedding> g_embedding;
static std::unique_ptr<nn::Attention> g_attention;

static constexpr int EMBED_DIM = 32; // Small for visualization
static constexpr int NUM_HEADS = 2;

// Initialize modules
void initNN() {
    g_embedding = std::make_unique<nn::Embedding>(
        g_tokenizer.vocab_size(), EMBED_DIM);
    g_attention = std::make_unique<nn::Attention>(EMBED_DIM, NUM_HEADS);
}

// Helper: convert TraceStep vector to a format Emscripten can handle
struct JSTraceStep {
    std::string component;
    std::string title;
    std::string detail;
    std::vector<std::string> internal;
};

std::vector<JSTraceStep> getTrace() {
    auto steps = core::Tracer::instance().take();
    std::vector<JSTraceStep> result;
    for (auto& s : steps) {
        result.push_back({s.component, s.title, s.detail, s.internal});
    }
    return result;
}

// --- API Functions ---

/// Tokenize text and return trace steps
std::vector<JSTraceStep> traceTokenize(const std::string& text) {
    core::Tracer::instance().enable();
    core::Tracer::instance().clear();
    auto ids = g_tokenizer.encode(text);
    return getTrace();
}

/// Get token IDs for a text (no tracing)
std::vector<int> tokenize(const std::string& text) {
    core::Tracer::instance().disable();
    return g_tokenizer.encode(text);
}

/// Run embedding on token IDs and return trace steps
std::vector<JSTraceStep> traceEmbedding(const std::vector<int>& token_ids) {
    if (!g_embedding) initNN();
    core::Tracer::instance().enable();
    core::Tracer::instance().clear();
    g_embedding->forward(token_ids);
    return getTrace();
}

/// Run attention on embedded vectors and return trace steps
std::vector<JSTraceStep> traceAttention(
    const std::vector<float>& input, int seq_len) {
    if (!g_attention) initNN();
    core::Tracer::instance().enable();
    core::Tracer::instance().clear();
    g_attention->forward(input, seq_len);
    return getTrace();
}

/// Run matrix multiplication and return trace steps
std::vector<JSTraceStep> traceMatMul(
    const std::vector<float>& A, int M, int K,
    const std::vector<float>& B, int K2, int N) {
    core::Tracer::instance().enable();
    core::Tracer::instance().clear();
    nn::MatMul::forward(A, M, K, B, K2, N);
    return getTrace();
}

/// Run softmax and return trace steps
std::vector<JSTraceStep> traceSoftmax(const std::vector<float>& logits) {
    core::Tracer::instance().enable();
    core::Tracer::instance().clear();
    nn::Softmax::forward(logits);
    return getTrace();
}

/// Run ReLU and return trace steps
std::vector<JSTraceStep> traceReLU(const std::vector<float>& input) {
    core::Tracer::instance().enable();
    core::Tracer::instance().clear();
    nn::Activations::relu(input);
    return getTrace();
}

/// Run GELU and return trace steps
std::vector<JSTraceStep> traceGELU(const std::vector<float>& input) {
    core::Tracer::instance().enable();
    core::Tracer::instance().clear();
    nn::Activations::gelu(input);
    return getTrace();
}

/// Run full text pipeline: tokenize → embed → attention → output
std::vector<JSTraceStep> traceFullTextPipeline(const std::string& text) {
    if (!g_embedding || !g_attention) initNN();
    core::Tracer::instance().enable();
    core::Tracer::instance().clear();

    // Tokenize
    auto ids = g_tokenizer.encode(text);

    // Embed
    auto embedded = g_embedding->forward(ids);

    // Attention
    int seq_len = static_cast<int>(ids.size());
    auto attended = g_attention->forward(embedded, seq_len);

    // Simple linear projection to vocab (simulated)
    std::vector<float> logits(g_tokenizer.vocab_size(), 0.0f);
    // Use last token's representation
    for (int i = 0; i < std::min(EMBED_DIM, g_tokenizer.vocab_size()); i++) {
        logits[i] = attended[(seq_len - 1) * EMBED_DIM + (i % EMBED_DIM)];
    }

    core::Tracer::instance().record("Linear", "Final Linear Projection",
        "Project last token to vocabulary size for next-word prediction",
        {"Input: last token representation [" + std::to_string(EMBED_DIM) + "]",
         "Output: logits [" + std::to_string(g_tokenizer.vocab_size()) + "]",
         "This maps abstract features → word scores"});

    // Softmax on logits
    nn::Softmax::forward(logits);

    return getTrace();
}

int getEmbedDim() { return EMBED_DIM; }
int getVocabSize() { return g_tokenizer.vocab_size(); }

#ifdef __EMSCRIPTEN__
using namespace emscripten;

EMSCRIPTEN_BINDINGS(inferx_nn) {
    register_vector<std::string>("VectorString");
    register_vector<int>("VectorInt");
    register_vector<float>("VectorFloat");

    value_object<JSTraceStep>("TraceStep")
        .field("component", &JSTraceStep::component)
        .field("title", &JSTraceStep::title)
        .field("detail", &JSTraceStep::detail)
        .field("internal", &JSTraceStep::internal);

    register_vector<JSTraceStep>("VectorTraceStep");

    function("initNN", &initNN);
    function("traceTokenize", &traceTokenize);
    function("tokenize", &tokenize);
    function("traceEmbedding", &traceEmbedding);
    function("traceAttention", &traceAttention);
    function("traceMatMul", &traceMatMul);
    function("traceSoftmax", &traceSoftmax);
    function("traceReLU", &traceReLU);
    function("traceGELU", &traceGELU);
    function("traceFullTextPipeline", &traceFullTextPipeline);
    function("getEmbedDim", &getEmbedDim);
    function("getVocabSize", &getVocabSize);
}
#endif
