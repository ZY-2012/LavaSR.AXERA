#include "lavasr.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Wav {
    int sr = 0;
    std::vector<float> data;
};

bool read_wav(const char* path, Wav& wav) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), {});
    if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) != 0 ||
        std::memcmp(b.data() + 8, "WAVE", 4) != 0) {
        std::fprintf(stderr, "not a RIFF/WAVE file: %s\n", path);
        return false;
    }
    int channels = 1, bits = 16, fmt = 1;
    size_t data_off = 0, data_size = 0;
    size_t p = 12;
    while (p + 8 <= b.size()) {
        const uint32_t sz = *reinterpret_cast<const uint32_t*>(&b[p + 4]);
        if (std::memcmp(&b[p], "fmt ", 4) == 0 && p + 8 + 16 <= b.size()) {
            fmt = *reinterpret_cast<const uint16_t*>(&b[p + 8]);
            channels = *reinterpret_cast<const uint16_t*>(&b[p + 10]);
            wav.sr = *reinterpret_cast<const int32_t*>(&b[p + 12]);
            bits = *reinterpret_cast<const uint16_t*>(&b[p + 22]);
        } else if (std::memcmp(&b[p], "data", 4) == 0) {
            data_off = p + 8;
            data_size = sz;
        }
        p += 8 + sz + (sz & 1);
    }
    if (!data_off || !channels || !wav.sr) {
        std::fprintf(stderr, "invalid wav chunks: %s\n", path);
        return false;
    }
    const int bytes = bits / 8;
    const size_t frames = data_size / (bytes * channels);
    wav.data.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
        double acc = 0.0;
        for (int ch = 0; ch < channels; ++ch) {
            const uint8_t* q = &b[data_off + (i * channels + ch) * bytes];
            if (fmt == 3 && bits == 32) {
                float v;
                std::memcpy(&v, q, 4);
                acc += v;
            } else if (bits == 16) {
                int16_t v;
                std::memcpy(&v, q, 2);
                acc += v / 32768.0;
            } else if (bits == 24) {
                int32_t v = q[0] | (q[1] << 8) | (q[2] << 16);
                if (v & 0x800000) v -= 0x1000000;
                acc += v / 8388608.0;
            } else if (bits == 32) {
                int32_t v;
                std::memcpy(&v, q, 4);
                acc += v / 2147483648.0;
            } else {
                std::fprintf(stderr, "unsupported bits=%d\n", bits);
                return false;
            }
        }
        wav.data[i] = static_cast<float>(acc / channels);
    }
    return true;
}

void write_wav(const char* path, const std::vector<float>& data, int sr) {
    std::ofstream f(path, std::ios::binary);
    const int32_t n = static_cast<int32_t>(data.size());
    const int32_t byte_rate = sr * 2, data_bytes = n * 2, chunk = 36 + data_bytes;
    const int16_t one = 1, two = 2, sixteen = 16;
    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char*>(&chunk), 4);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    int32_t sz = 16;
    f.write(reinterpret_cast<const char*>(&sz), 4);
    f.write(reinterpret_cast<const char*>(&one), 2);
    f.write(reinterpret_cast<const char*>(&one), 2);
    f.write(reinterpret_cast<const char*>(&sr), 4);
    f.write(reinterpret_cast<const char*>(&byte_rate), 4);
    f.write(reinterpret_cast<const char*>(&two), 2);
    f.write(reinterpret_cast<const char*>(&sixteen), 2);
    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_bytes), 4);
    for (float v : data) {
        const int16_t s = static_cast<int16_t>((v > 1 ? 1 : (v < -1 ? -1 : v)) * 32767);
        f.write(reinterpret_cast<const char*>(&s), 2);
    }
}

float rms(const std::vector<float>& x) {
    double s = 0.0;
    for (float v : x) s += double(v) * v;
    return x.empty() ? 0.0f : static_cast<float>(std::sqrt(s / x.size()));
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s <den_encoder.axmodel> <den_decoder.axmodel> <enhancer.axmodel> "
                 "<input.wav> [output.wav] [--no-denoise] [--bench]\n", argv0);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        usage(argv[0]);
        return 1;
    }
    const std::string enc = argv[1], dec = argv[2], enh = argv[3];
    const char* input = argv[4];
    const char* output = (argc > 5 && argv[5][0] != '-') ? argv[5] : "output.wav";
    bool denoise = true, bench = false;
    for (int i = 5; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-denoise") == 0) denoise = false;
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;
    }
    Wav wav;
    if (!read_wav(input, wav)) return 1;
    lavasr::LavaSR model;
    if (!model.load(enc, dec, enh)) return 1;
    const auto t0 = std::chrono::steady_clock::now();
    auto y = model.enhance(wav.data, wav.sr, denoise);
    const auto t1 = std::chrono::steady_clock::now();
    if (y.empty()) {
        std::fprintf(stderr, "inference failed\n");
        return 1;
    }
    double dt = std::chrono::duration<double>(t1 - t0).count();
    if (bench) {
        const int N = 3;
        const auto b0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) y = model.enhance(wav.data, wav.sr, denoise);
        const auto b1 = std::chrono::steady_clock::now();
        dt = std::chrono::duration<double>(b1 - b0).count() / N;
    }
    write_wav(output, y, 48000);
    const double dur = double(wav.data.size()) / wav.sr;
    std::printf("LavaSR v2: %d Hz -> 48000 Hz, %.2fs audio\n", wav.sr, dur);
    std::printf("  denoise : %s\n", denoise ? "true" : "false");
    std::printf("  time    : %.3fs   RTF %.4f\n", dt, dt / std::max(dur, 1e-9));
    std::printf("  rms in/out: %.4f / %.4f\n", rms(wav.data), rms(y));
    std::printf("  output  : %s\n", output);
    return 0;
}
