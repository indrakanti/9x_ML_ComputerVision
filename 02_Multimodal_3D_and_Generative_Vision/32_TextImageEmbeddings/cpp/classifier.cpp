#include "classifier.hpp"

#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cv9x::module32 {
namespace {
constexpr int kSide = 32;
constexpr int kDim = 32;
constexpr int kTokens = 32;
constexpr float kScale = 10.0F;

const std::array<std::string, 10> kLabels{
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
};

std::string sha256(const std::filesystem::path& path) {
    std::ifstream source(path, std::ios::binary);
    if (!source) throw std::runtime_error("missing ONNX model: " + path.string());
    using DigestPtr = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
    DigestPtr digest(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!digest || EVP_DigestInit_ex(digest.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("SHA256 initialization failed");
    std::array<char, 65536> buffer{};
    while (source) {
        source.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = source.gcount();
        if (count > 0 && EVP_DigestUpdate(
                digest.get(), buffer.data(), static_cast<std::size_t>(count)) != 1)
            throw std::runtime_error("SHA256 update failed");
    }
    if (source.bad()) throw std::runtime_error("cannot read ONNX model");
    std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(digest.get(), hash.data(), &length) != 1 || length != 32)
        throw std::runtime_error("SHA256 finalization failed");
    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < length; ++i) hex << std::setw(2) << int(hash[i]);
    return hex.str();
}

void check_bundle(const std::filesystem::path& directory) {
    std::ifstream input(directory / "manifest.json");
    if (!input) throw std::runtime_error("manifest.json is missing");
    nlohmann::json meta;
    input >> meta;
    if (meta.at("schema_version") != 1 || meta.at("format") != "onnx" ||
        meta.at("opset") != 17 ||
        meta.at("image_shape") != nlohmann::json::array({1, 3, 32, 32}) ||
        meta.at("text_shape") != nlohmann::json::array({10, 32}) ||
        meta.at("embedding_width") != 32 ||
        meta.at("logit_scale") != kScale ||
        meta.at("image_preprocessing") !=
            "RGB; resize to 32x32; float32 CHW in [0,1]" ||
        meta.at("tokenizer") !=
            "printable ASCII codepoint minus 31; pad 0 to 32")
        throw std::runtime_error("incompatible Module 32 artifact schema");
    if (meta.at("labels") != nlohmann::json(kLabels))
        throw std::runtime_error("class label order does not match the executable");
    nlohmann::json prompts = nlohmann::json::array();
    for (const auto& label : kLabels) prompts.push_back("a photo of a " + label);
    if (meta.at("prompts") != prompts)
        throw std::runtime_error("text prompts do not match the native tokenizer");
    const auto& hashes = meta.at("files_sha256");
    if (!hashes.is_object() || hashes.size() != 2)
        throw std::runtime_error("unexpected ONNX model-file set");
    for (const char* name : {"image.onnx", "text.onnx"}) {
        if (!hashes.contains(name) || !hashes.at(name).is_string() ||
            hashes.at(name).get<std::string>() != sha256(directory / name))
            throw std::runtime_error(std::string("SHA256 mismatch: ") + name);
    }
}

std::array<int64_t, kTokens> tokenize(const std::string& text) {
    if (text.empty() || text.size() > kTokens)
        throw std::invalid_argument("prompt outside 1..32 ASCII characters");
    std::array<int64_t, kTokens> tokens{};
    for (std::size_t index = 0; index < text.size(); ++index) {
        const auto code = static_cast<unsigned char>(text[index]);
        if (code < 32 || code > 126)
            throw std::invalid_argument("prompt contains non-printable ASCII");
        tokens[index] = static_cast<int64_t>(code) - 31;
    }
    return tokens;
}

std::vector<float> load_image(const std::string& path) {
    const cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("cannot decode image: " + path);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(kSide, kSide), 0, 0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat floating;
    rgb.convertTo(floating, CV_32FC3, 1.0 / 255.0);
    std::vector<float> nchw(3 * kSide * kSide);
    for (int row = 0; row < kSide; ++row)
        for (int column = 0; column < kSide; ++column) {
            const auto pixel = floating.at<cv::Vec3f>(row, column);
            for (int channel = 0; channel < 3; ++channel)
                nchw[static_cast<std::size_t>(channel * kSide * kSide +
                                              row * kSide + column)] = pixel[channel];
        }
    return nchw;
}

std::vector<float> infer(Ort::Session& session, const char* input_name,
                         const char* output_name, Ort::Value& input,
                         std::size_t expected_count) {
    const char* inputs[] = {input_name};
    const char* outputs[] = {output_name};
    auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
    if (result.size() != 1 || !result.front().IsTensor() ||
        result.front().GetTensorTypeAndShapeInfo().GetElementCount() != expected_count)
        throw std::runtime_error("unexpected ONNX output count or tensor shape");
    const float* data = result.front().GetTensorData<float>();
    return std::vector<float>(data, data + expected_count);
}
}  // namespace

std::string validated_directory(const std::string& directory) {
    check_bundle(directory);
    return directory;
}

class Classifier::Impl {
public:
    explicit Impl(const std::string& dir)
        : directory_(validated_directory(dir)),
          environment_(ORT_LOGGING_LEVEL_WARNING, "cv9x-module32"),
          image_session_(environment_, (std::filesystem::path(directory_) / "image.onnx").c_str(),
                         options()),
          text_session_(environment_, (std::filesystem::path(directory_) / "text.onnx").c_str(),
                        options()) {
        const auto image_input = image_session_.GetInputNameAllocated(0, allocator_);
        const auto image_output = image_session_.GetOutputNameAllocated(0, allocator_);
        const auto text_input = text_session_.GetInputNameAllocated(0, allocator_);
        const auto text_output = text_session_.GetOutputNameAllocated(0, allocator_);
        image_input_ = image_input.get();
        image_output_ = image_output.get();
        text_input_ = text_input.get();
        text_output_ = text_output.get();

        std::vector<int64_t> text_ids;
        text_ids.reserve(kLabels.size() * kTokens);
        for (const auto& label : kLabels) {
            const auto tokens = tokenize("a photo of a " + label);
            text_ids.insert(text_ids.end(), tokens.begin(), tokens.end());
        }
        std::array<int64_t, 2> shape{static_cast<int64_t>(kLabels.size()), kTokens};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<int64_t>(
            memory, text_ids.data(), text_ids.size(), shape.data(), shape.size());
        text_embeddings_ = infer(text_session_, text_input_.c_str(),
                                 text_output_.c_str(), tensor, kLabels.size() * kDim);
    }

    std::vector<RankedClass> predict(const std::string& image_path) {
        auto image = load_image(image_path);
        std::array<int64_t, 4> shape{1, 3, kSide, kSide};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(
            memory, image.data(), image.size(), shape.data(), shape.size());
        const auto embedding = infer(image_session_, image_input_.c_str(),
                                     image_output_.c_str(), tensor, kDim);
        std::array<float, 10> logits{};
        for (std::size_t cls = 0; cls < kLabels.size(); ++cls)
            for (int index = 0; index < kDim; ++index)
                logits[cls] += kScale * embedding[static_cast<std::size_t>(index)] *
                               text_embeddings_[cls * kDim + static_cast<std::size_t>(index)];
        const auto highest = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0F;
        for (auto& score : logits) {
            score = std::exp(score - highest);
            sum += score;
        }
        std::vector<std::size_t> ranking(kLabels.size());
        std::iota(ranking.begin(), ranking.end(), 0);
        std::sort(ranking.begin(), ranking.end(),
                  [&](std::size_t a, std::size_t b) {
                      if (logits[a] != logits[b]) return logits[a] > logits[b];
                      return a < b;
                  });
        std::vector<RankedClass> result;
        for (const auto cls : ranking)
            result.push_back({kLabels[cls], logits[cls] / sum});
        return result;
    }

private:
    static Ort::SessionOptions options() {
        Ort::SessionOptions result;
        result.SetIntraOpNumThreads(1);
        return result;
    }

    std::string directory_;
    Ort::Env environment_;
    Ort::Session image_session_;
    Ort::Session text_session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string image_input_;
    std::string image_output_;
    std::string text_input_;
    std::string text_output_;
    std::vector<float> text_embeddings_;
};

Classifier::Classifier(const std::string& directory)
    : impl_(std::make_unique<Impl>(directory)) {}
Classifier::~Classifier() = default;
Classifier::Classifier(Classifier&&) noexcept = default;
Classifier& Classifier::operator=(Classifier&&) noexcept = default;
std::vector<RankedClass> Classifier::predict(const std::string& image) const {
    return impl_->predict(image);
}
}  // namespace cv9x::module32
