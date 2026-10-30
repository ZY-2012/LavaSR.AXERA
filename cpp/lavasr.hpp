#pragma once
// LavaSR v2 AX650 runtime: UL-UNAS denoiser + Vocos BWE, NPU cores.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

#include "assets.h"
#include "dpgrnn.hpp"
#include "dsp.hpp"
#include "model_runner.hpp"

namespace lavasr {

class LavaSR {
public:
    bool load(const std::string& encoder, const std::string& decoder, const std::string& enhancer) {
        return enc_.load(encoder) && dec_.load(decoder) && enh_.load(enhancer);
    }

    std::vector<float> enhance(const std::vector<float>& wav, int sr, bool denoise) {
        const bool prof = std::getenv("LAVASR_PROFILE") != nullptr;
        auto t0 = std::chrono::steady_clock::now();
        auto x16 = resample(wav, sr, 16000);
        auto t1 = std::chrono::steady_clock::now();
        if (denoise) x16 = denoise_16k(x16);
        auto t2 = std::chrono::steady_clock::now();
        auto x48 = resample(x16, 16000, 48000);
        auto t3 = std::chrono::steady_clock::now();
        auto gen = enhance_48k(x48);
        auto t4 = std::chrono::steady_clock::now();
        const float cutoff = static_cast<float>(std::min(sr, 16000)) / 2.0f;
        std::vector<float> orig(x48.begin(), x48.begin() + gen.size());
        auto y = lr_merge(gen, orig, cutoff);
        auto t5 = std::chrono::steady_clock::now();
        if (prof) {
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            std::fprintf(stderr, "profile: resample16=%.0fms denoise=%.0fms resample48=%.0fms enhance48=%.0fms merge=%.0fms\n",
                         ms(t0,t1), ms(t1,t2), ms(t2,t3), ms(t3,t4), ms(t4,t5));
        }
        return y;
    }

private:
    AxModel enc_, dec_, enh_;
    DpGrnn dp_;

    static void extend_reflect(std::vector<float>& x, int pad) {
        if (pad <= 0) return;
        const int n = static_cast<int>(x.size());
        if (n < 2) {
            x.resize(n + pad, 0.0f);
            return;
        }
        x.resize(n + pad);
        for (int i = 0; i < pad; ++i) x[n + i] = x[std::max(0, n - 2 - i)];
    }

    static std::vector<int> starts(int length, int chunk, int hop, int& pad) {
        std::vector<int> out;
        if (length <= chunk) {
            pad = 0;
            out.push_back(0);
            return out;
        }
        pad = (hop - (length - chunk) % hop) % hop;
        for (int s = 0; s <= length + pad - chunk; s += hop) out.push_back(s);
        if (out.back() + chunk < length + pad) out.push_back(length + pad - chunk);
        return out;
    }

    std::vector<float> denoise_chunk(std::vector<float> c) {
        if (static_cast<int>(c.size()) < kDenChunk) {
            const int n = static_cast<int>(c.size());
            c.resize(kDenChunk);
            for (int i = n; i < kDenChunk; ++i) c[i] = c[std::max(0, n - 2 - (i - n))];
        }
        const auto spec = den_stft_d(c, std::vector<float>(kDenWindow, kDenWindow + kDenWin));
        auto feat = den_features_d(spec, kErbFc);
        if (!enc_.set_input("erb_feat", feat.data(), feat.size()) || !enc_.run()) return {};
        std::vector<float> f, en0, en1, en2, en3, en4;
        if (!enc_.get_output("feat", f, 1 * 16 * kDenFrames * 33) ||
            !enc_.get_output("en0", en0, 1 * 12 * kDenFrames * 65) ||
            !enc_.get_output("en1", en1, 1 * 24 * kDenFrames * 33) ||
            !enc_.get_output("en2", en2, 1 * 24 * kDenFrames * 33) ||
            !enc_.get_output("en3", en3, 1 * 32 * kDenFrames * 33) ||
            !enc_.get_output("en4", en4, 1 * 16 * kDenFrames * 33)) return {};
        auto dp = dp_.forward(f, 1, 16, kDenFrames, 33);
        if (!dec_.set_input("feat", dp.data(), dp.size()) ||
            !dec_.set_input("en0", en0.data(), en0.size()) ||
            !dec_.set_input("en1", en1.data(), en1.size()) ||
            !dec_.set_input("en2", en2.data(), en2.size()) ||
            !dec_.set_input("en3", en3.data(), en3.size()) ||
            !dec_.set_input("en4", en4.data(), en4.size()) || !dec_.run()) return {};
        std::vector<float> mask;
        if (!dec_.get_output("mask", mask, 1 * 1 * kDenFrames * kErbBins)) return {};
        auto full = den_mask_expand(mask.data(), kDenFrames, kIerbFc);
        std::vector<std::complex<float>> masked(static_cast<size_t>(kDenFrames) * kDenBins);
        for (size_t i = 0; i < masked.size(); ++i) {
            const float m = full[i];
            masked[i] = std::complex<float>(static_cast<float>(spec[i].real()) * m,
                                            static_cast<float>(spec[i].imag()) * m);
        }
        return den_istft(masked, std::vector<float>(kDenWindow, kDenWindow + kDenWin), kDenChunk);
    }

    std::vector<float> denoise_16k(const std::vector<float>& x) {
        const int hop = static_cast<int>(kDenChunk * (1.0f - kDenOverlap));
        int pad = 0;
        const auto st = starts(static_cast<int>(x.size()), kDenChunk, hop, pad);
        auto xp = x;
        extend_reflect(xp, pad);
        std::vector<std::vector<float>> chunks;
        for (int s : st) {
            std::vector<float> c(xp.begin() + s, xp.begin() + std::min<int>(s + kDenChunk, xp.size()));
            chunks.push_back(denoise_chunk(c));
            if (chunks.back().empty()) return {};
        }
        return crossfade(chunks, hop, kDenChunk, static_cast<int>(x.size()));
    }

    std::vector<float> enhance_chunk(std::vector<float> c) {
        if (static_cast<int>(c.size()) < kBweChunk) {
            const int n = static_cast<int>(c.size());
            c.resize(kBweChunk);
            for (int i = n; i < kBweChunk; ++i) c[i] = c[std::max(0, n - 2 - (i - n))];
        }
        const auto mel = mel_features(c, kMelFb, std::vector<float>(kMelWindow, kMelWindow + kBweWin));
        if (!enh_.set_input("mel", mel.data(), mel.size()) || !enh_.run()) return {};
        std::vector<float> head;
        if (!enh_.get_output("head_out", head, 1 * (2 * kBweBins) * kBweFrames)) return {};
        std::vector<std::complex<float>> spec(static_cast<size_t>(kBweFrames) * kBweBins);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < kBweFrames; ++t) {
            for (int k = 0; k < kBweBins; ++k) {
                const float m = expf(std::min(head[static_cast<size_t>(k) * kBweFrames + t], 1e3f));
                const float p = head[static_cast<size_t>(kBweBins + k) * kBweFrames + t];
                float sn, cs;
                __builtin_sincosf(p, &sn, &cs);
                spec[static_cast<size_t>(t) * kBweBins + k] = std::complex<float>(m * cs, m * sn);
            }
        }
        return bwe_istft(spec, std::vector<float>(kMelWindow, kMelWindow + kBweWin));
    }

    std::vector<float> enhance_48k(const std::vector<float>& x) {
        const int hop = static_cast<int>(kBweChunk * (1.0f - kBweOverlap));
        const int out_len = (static_cast<int>(x.size()) / 512) * 512;
        int pad = 0;
        const auto st = starts(static_cast<int>(x.size()), kBweChunk, hop, pad);
        auto xp = x;
        extend_reflect(xp, pad);
        std::vector<std::vector<float>> chunks;
        for (int s : st) {
            std::vector<float> c(xp.begin() + s, xp.begin() + std::min<int>(s + kBweChunk, xp.size()));
            chunks.push_back(enhance_chunk(c));
            if (chunks.back().empty()) return {};
        }
        auto y = crossfade(chunks, hop, kBweChunk, static_cast<int>(xp.size()));
        y.resize(out_len);
        return y;
    }
};

}  // namespace lavasr
