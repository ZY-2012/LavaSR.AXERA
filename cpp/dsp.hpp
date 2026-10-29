#pragma once
// Host DSP for LavaSR: FFT / STFT / mel / ERB / iSTFT / resample / LR merge.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <vector>

#include "third_party/kissfft/kiss_fftr.h"

namespace lavasr {

constexpr int kDenNfft = 512;
constexpr int kDenHop = 256;
constexpr int kDenWin = 512;
constexpr int kDenChunk = 76800;   // 4.8 s @ 16 kHz
constexpr int kDenFrames = 301;
constexpr float kDenOverlap = 0.25f;

constexpr int kBweNfft = 2048;
constexpr int kBweHop = 512;
constexpr int kBweWin = 2048;
constexpr int kBweChunk = 230400;  // 4.8 s @ 48 kHz
constexpr int kBweFrames = 450;
constexpr float kBweOverlap = 0.5f;

constexpr int kDenBins = 257;
constexpr int kErbLow = 65;
constexpr int kErbHigh = 64;
constexpr int kErbBins = 129;
constexpr int kBweBins = 1025;
constexpr int kMels = 80;

inline void fft_c2c(std::vector<std::complex<float>>& a, bool inverse) {
    const int n = static_cast<int>(a.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const double pi = 3.14159265358979323846;
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = (inverse ? 2.0 : -2.0) * pi / len;
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                const std::complex<float> u = a[i + j];
                const std::complex<float> v = a[i + j + len / 2] *
                    std::complex<float>(static_cast<float>(w.real()), static_cast<float>(w.imag()));
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        const float inv = 1.0f / static_cast<float>(n);
        for (auto& x : a) x *= inv;
    }
}

inline void fft_c2c_d(std::vector<std::complex<double>>& a, bool inverse) {
    const int n = static_cast<int>(a.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const double pi = 3.14159265358979323846;
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = (inverse ? 2.0 : -2.0) * pi / len;
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                const auto u = a[i + j];
                const auto v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) for (auto& x : a) x /= static_cast<double>(n);
}

inline std::vector<std::complex<double>> rfft_d(const float* x, int n) {
    std::vector<std::complex<double>> a(n);
    for (int i = 0; i < n; ++i) a[i] = std::complex<double>(x[i], 0.0);
    fft_c2c_d(a, false);
    return std::vector<std::complex<double>>(a.begin(), a.begin() + n / 2 + 1);
}

inline std::vector<std::complex<float>> rfft(const float* x, int n) {
    std::vector<std::complex<float>> a(n);
    for (int i = 0; i < n; ++i) a[i] = std::complex<float>(x[i], 0.0f);
    fft_c2c(a, false);
    return std::vector<std::complex<float>>(a.begin(), a.begin() + n / 2 + 1);
}

inline std::vector<float> irfft(const std::vector<std::complex<float>>& spec, int n) {
    std::vector<std::complex<float>> a(n, std::complex<float>(0.0f, 0.0f));
    const int h = n / 2;
    for (int k = 0; k <= h; ++k) a[k] = spec[k];
    for (int k = 1; k < h; ++k) a[n - k] = std::conj(spec[k]);
    fft_c2c(a, true);
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) out[i] = a[i].real();
    return out;
}

inline std::vector<float> hann(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i) {
        w[i] = 0.5f - 0.5f * static_cast<float>(std::cos(2.0 * 3.14159265358979323846 * i / (n - 1)));
    }
    return w;
}

class RealFFT {
public:
    explicit RealFFT(int n) : n_(n) {
        fwd_ = kiss_fftr_alloc(n, 0, nullptr, nullptr);
        inv_ = kiss_fftr_alloc(n, 1, nullptr, nullptr);
    }
    ~RealFFT() {
        free(fwd_);
        free(inv_);
    }
    void forward(const float* x, std::vector<std::complex<float>>& out) {
        out.resize(n_ / 2 + 1);
        kiss_fftr(fwd_, x, reinterpret_cast<kiss_fft_cpx*>(out.data()));
    }
    std::vector<float> inverse(const std::vector<std::complex<float>>& spec) {
        std::vector<float> y(n_);
        kiss_fftri(inv_, reinterpret_cast<const kiss_fft_cpx*>(spec.data()), y.data());
        const float inv = 1.0f / static_cast<float>(n_);
        for (auto& v : y) v *= inv;
        return y;
    }

private:
    int n_;
    kiss_fftr_cfg fwd_ = nullptr;
    kiss_fftr_cfg inv_ = nullptr;
};


inline std::vector<float> reflect_pad(const std::vector<float>& x, int left, int right) {
    const int n = static_cast<int>(x.size());
    std::vector<float> out(n + left + right);
    for (int i = 0; i < left; ++i) out[i] = x[std::min(left - i, n - 1)];
    std::copy(x.begin(), x.end(), out.begin() + left);
    for (int i = 0; i < right; ++i) out[left + n + i] = x[std::max(0, n - 2 - i)];
    return out;
}

inline std::vector<std::complex<float>> den_stft(const std::vector<float>& x,
                                                 const std::vector<float>& win) {
    const auto padded = reflect_pad(x, kDenWin / 2, kDenWin / 2);
    const int frames = (static_cast<int>(padded.size()) - kDenWin) / kDenHop + 1;
    std::vector<std::complex<float>> spec(static_cast<size_t>(frames) * kDenBins);
    std::vector<float> frame(kDenWin);
    for (int t = 0; t < frames; ++t) {
        const int off = t * kDenHop;
        for (int i = 0; i < kDenWin; ++i) frame[i] = padded[off + i] * win[i];
        const auto s = rfft(frame.data(), kDenWin);
        std::copy(s.begin(), s.end(), spec.begin() + static_cast<size_t>(t) * kDenBins);
    }
    return spec;
}

inline std::vector<float> den_istft(const std::vector<std::complex<float>>& spec,
                                    const std::vector<float>& win, int out_len) {
    const int frames = static_cast<int>(spec.size()) / kDenBins;
    const int total = (frames - 1) * kDenHop + kDenWin;
    std::vector<float> y(total, 0.0f), env(total, 0.0f);
    for (int t = 0; t < frames; ++t) {
        const auto frame = irfft(std::vector<std::complex<float>>(
            spec.begin() + static_cast<size_t>(t) * kDenBins,
            spec.begin() + static_cast<size_t>(t + 1) * kDenBins), kDenWin);
        const int off = t * kDenHop;
        for (int i = 0; i < kDenWin; ++i) {
            y[off + i] += frame[i] * win[i];
            env[off + i] += win[i] * win[i];
        }
    }
    const int pad = kDenHop;
    y = std::vector<float>(y.begin() + pad, y.end() - pad);
    env = std::vector<float>(env.begin() + pad, env.end() - pad);
    for (size_t i = 0; i < y.size(); ++i) y[i] /= std::max(env[i], 1e-12f);
    if (static_cast<int>(y.size()) < out_len) y.resize(out_len, 0.0f);
    y.resize(out_len);
    return y;
}

inline std::vector<float> den_features(const std::vector<std::complex<float>>& spec,
                                       const float* erb_fc) {
    const int frames = static_cast<int>(spec.size()) / kDenBins;
    std::vector<float> feat(static_cast<size_t>(frames) * kErbBins);
    std::vector<float> logmag(kDenBins);
    for (int t = 0; t < frames; ++t) {
        const auto* s = spec.data() + static_cast<size_t>(t) * kDenBins;
        for (int k = 0; k < kDenBins; ++k) {
            const float m = std::max(std::abs(s[k]), 1e-12f);
            logmag[k] = std::log10(m);
        }
        float* out = feat.data() + static_cast<size_t>(t) * kErbBins;
        std::copy(logmag.begin(), logmag.begin() + kErbLow, out);
        for (int m = 0; m < kErbHigh; ++m) {
            float acc = 0.0f;
            const float* row = erb_fc + static_cast<size_t>(m) * (kDenBins - kErbLow);
            for (int k = 0; k < kDenBins - kErbLow; ++k) acc += logmag[kErbLow + k] * row[k];
            out[kErbLow + m] = acc;
        }
    }
    return feat;
}

inline std::vector<std::complex<double>> den_stft_d(const std::vector<float>& x,
                                                    const std::vector<float>& win) {
    const auto padded = reflect_pad(x, kDenWin / 2, kDenWin / 2);
    const int frames = (static_cast<int>(padded.size()) - kDenWin) / kDenHop + 1;
    std::vector<std::complex<double>> spec(static_cast<size_t>(frames) * kDenBins);
    std::vector<float> frame(kDenWin);
    for (int t = 0; t < frames; ++t) {
        const int off = t * kDenHop;
        for (int i = 0; i < kDenWin; ++i) frame[i] = padded[off + i] * win[i];
        const auto s = rfft_d(frame.data(), kDenWin);
        std::copy(s.begin(), s.end(), spec.begin() + static_cast<size_t>(t) * kDenBins);
    }
    return spec;
}

inline std::vector<float> den_features_d(const std::vector<std::complex<double>>& spec,
                                         const float* erb_fc) {
    const int frames = static_cast<int>(spec.size()) / kDenBins;
    std::vector<float> feat(static_cast<size_t>(frames) * kErbBins);
    std::vector<double> logmag(kDenBins);
    for (int t = 0; t < frames; ++t) {
        const auto* s = spec.data() + static_cast<size_t>(t) * kDenBins;
        for (int k = 0; k < kDenBins; ++k) {
            const double m = std::max(std::abs(s[k]), 1e-12);
            logmag[k] = std::log10(m);
        }
        float* out = feat.data() + static_cast<size_t>(t) * kErbBins;
        for (int k = 0; k < kErbLow; ++k) out[k] = static_cast<float>(logmag[k]);
        for (int m = 0; m < kErbHigh; ++m) {
            double acc = 0.0;
            const float* row = erb_fc + static_cast<size_t>(m) * (kDenBins - kErbLow);
            for (int k = 0; k < kDenBins - kErbLow; ++k) acc += logmag[kErbLow + k] * row[k];
            out[kErbLow + m] = static_cast<float>(acc);
        }
    }
    return feat;
}

inline std::vector<float> den_mask_expand(const float* mask, int frames, const float* ierb_fc) {
    std::vector<float> out(static_cast<size_t>(frames) * kDenBins);
    for (int t = 0; t < frames; ++t) {
        const float* m = mask + static_cast<size_t>(t) * kErbBins;
        float* o = out.data() + static_cast<size_t>(t) * kDenBins;
        std::copy(m, m + kErbLow, o);
        for (int k = 0; k < kDenBins - kErbLow; ++k) {
            float acc = 0.0f;
            const float* row = ierb_fc + static_cast<size_t>(k) * kErbHigh;
            for (int j = 0; j < kErbHigh; ++j) acc += m[kErbLow + j] * row[j];
            o[kErbLow + k] = acc;
        }
    }
    return out;
}

inline std::vector<float> mel_features(const std::vector<float>& x, const float* fb,
                                       const std::vector<float>& win) {
    const int pad = (kBweNfft - kBweHop) / 2;
    const auto padded = reflect_pad(x, pad, pad);
    const int frames = (static_cast<int>(padded.size()) - kBweNfft) / kBweHop + 1;
    std::vector<float> feat(static_cast<size_t>(kMels) * frames);
    std::vector<float> frame(kBweNfft), mag(kBweBins);
    static RealFFT fft(kBweNfft);
    static std::vector<float> fb_t;
    if (fb_t.empty()) {
        fb_t.resize(static_cast<size_t>(kMels) * kBweBins);
        for (int k = 0; k < kBweBins; ++k)
            for (int m = 0; m < kMels; ++m)
                fb_t[static_cast<size_t>(m) * kBweBins + k] = fb[static_cast<size_t>(k) * kMels + m];
    }
    std::vector<std::complex<float>> spec;
    for (int t = 0; t < frames; ++t) {
        const int off = t * kBweHop;
        for (int i = 0; i < kBweNfft; ++i) frame[i] = padded[off + i] * win[i];
        fft.forward(frame.data(), spec);
        for (int k = 0; k < kBweBins; ++k) { const float re = spec[k].real(), im = spec[k].imag(); mag[k] = std::sqrt(re * re + im * im); }
        for (int m = 0; m < kMels; ++m) {
            const float* row = fb_t.data() + static_cast<size_t>(m) * kBweBins;
            float acc = 0.0f;
            for (int k = 0; k < kBweBins; ++k) acc += mag[k] * row[k];
            feat[static_cast<size_t>(m) * frames + t] = logf(std::max(acc, 1e-5f));
        }
    }
    return feat;
}

inline std::vector<float> mel_features_d(const std::vector<float>& x, const float* fb,
                                         const std::vector<float>& win) {
    const int pad = (kBweNfft - kBweHop) / 2;
    const auto padded = reflect_pad(x, pad, pad);
    const int frames = (static_cast<int>(padded.size()) - kBweNfft) / kBweHop + 1;
    std::vector<float> feat(static_cast<size_t>(kMels) * frames);
    std::vector<float> frame(kBweNfft);
    for (int t = 0; t < frames; ++t) {
        const int off = t * kBweHop;
        for (int i = 0; i < kBweNfft; ++i) frame[i] = padded[off + i] * win[i];
        const auto s = rfft_d(frame.data(), kBweNfft);
        for (int m = 0; m < kMels; ++m) {
            double acc = 0.0;
            for (int k = 0; k < kBweBins; ++k) acc += std::abs(s[k]) * fb[static_cast<size_t>(k) * kMels + m];
            feat[static_cast<size_t>(m) * frames + t] = static_cast<float>(std::log(std::max(acc, 1e-5)));
        }
    }
    return feat;
}

inline std::vector<float> bwe_istft(const std::vector<std::complex<float>>& spec,
                                    const std::vector<float>& win) {
    const int frames = static_cast<int>(spec.size()) / kBweBins;
    const int total = (frames - 1) * kBweHop + kBweWin;
    std::vector<float> y(total, 0.0f), env(total, 0.0f);
    static RealFFT fft(kBweNfft);
    for (int t = 0; t < frames; ++t) {
        const std::vector<std::complex<float>> one(
            spec.begin() + static_cast<size_t>(t) * kBweBins,
            spec.begin() + static_cast<size_t>(t + 1) * kBweBins);
        const auto frame = fft.inverse(one);
        const int off = t * kBweHop;
        for (int i = 0; i < kBweWin; ++i) {
            y[off + i] += frame[i] * win[i];
            env[off + i] += win[i] * win[i];
        }
    }
    const int pad = (kBweWin - kBweHop) / 2;
    y = std::vector<float>(y.begin() + pad, y.end() - pad);
    env = std::vector<float>(env.begin() + pad, env.end() - pad);
    for (size_t i = 0; i < y.size(); ++i) y[i] /= std::max(env[i], 1e-12f);
    return y;
}

inline std::vector<float> resample(const std::vector<float>& x, int orig, int target) {
    if (orig == target) return x;
    int g = std::gcd(orig, target);
    const int o = orig / g, n = target / g;
    const double pi = 3.14159265358979323846;
    const double rolloff = 0.99, width_scale = 6.0;
    const double base = std::min(o, n) * rolloff;
    const int width = static_cast<int>(std::ceil(width_scale * o / base));
    const int ksize = 2 * width + o;
    std::vector<float> kernel(static_cast<size_t>(n) * ksize);
    for (int p = 0; p < n; ++p) {
        for (int k = 0; k < ksize; ++k) {
            double t = (static_cast<double>(k - width) / o) - static_cast<double>(p) / n;
            t = std::max(-width_scale, std::min(width_scale, t * base));
            const double window = std::pow(std::cos(t * pi / width_scale / 2.0), 2.0);
            const double sinc = (std::abs(t) < 1e-12) ? 1.0 : std::sin(pi * t) / (pi * t);
            kernel[static_cast<size_t>(p) * ksize + k] = static_cast<float>(sinc * window * base / o);
        }
    }
    std::vector<float> padded(width + x.size() + width + o, 0.0f);
    std::copy(x.begin(), x.end(), padded.begin() + width);
    const int n_out = (static_cast<int>(padded.size()) - ksize) / o + 1;
    std::vector<float> y(static_cast<size_t>(n_out) * n);
    for (int p = 0; p < n; ++p) {
        for (int j = 0; j < n_out; ++j) {
            float acc = 0.0f;
            const int off = j * o;
            const float* row = kernel.data() + static_cast<size_t>(p) * ksize;
            for (int k = 0; k < ksize; ++k) acc += padded[off + k] * row[k];
            y[static_cast<size_t>(j) * n + p] = acc;
        }
    }
    const int target_len = static_cast<int>(std::ceil(static_cast<double>(n) * x.size() / o));
    y.resize(std::min<int>(y.size(), target_len));
    return y;
}

inline float smoothstep(float x) {
    return x * x * (3.0f - 2.0f * x);
}

inline std::vector<float> lr_mask(int bins, float cutoff, int transition, int sr) {
    const int cutoff_bin = static_cast<int>((cutoff / (sr / 2.0f)) * bins);
    std::vector<float> mask(bins, 1.0f);
    const int half = transition / 2;
    const int start = std::max(0, cutoff_bin - half);
    const int end = std::min(bins, cutoff_bin + half);
    for (int i = 0; i < start; ++i) mask[i] = 0.0f;
    for (int j = 0; j < end - start; ++j) {
        const float x = -1.0f + 2.0f * static_cast<float>(j) / std::max(1, transition - 1);
        mask[start + j] = smoothstep((x + 1.0f) / 2.0f);
    }
    return mask;
}

inline int next_pow2(int n) {
    int m = 1;
    while (m < n) m <<= 1;
    return m;
}


inline std::vector<float> lr_merge(const std::vector<float>& gen,
                                   const std::vector<float>& orig,
                                   float cutoff, int transition = 1024, int sr = 48000) {
    const int n = static_cast<int>(gen.size());
    const int m = next_pow2(n + 16384);
    const int tb = std::max(2, static_cast<int>(std::lround(transition * static_cast<double>(m) / n)) / 2 * 2);
    const auto mask = lr_mask(m / 2 + 1, cutoff, tb, sr);
    std::vector<float> a(m, 0.0f), b(m, 0.0f);
    std::copy(gen.begin(), gen.end(), a.begin());
    std::copy(orig.begin(), orig.end(), b.begin());
    static std::map<int, RealFFT*> plans;
    auto it = plans.find(m);
    if (it == plans.end()) it = plans.emplace(m, new RealFFT(m)).first;
    std::vector<std::complex<float>> sa, sb;
    it->second->forward(a.data(), sa);
    it->second->forward(b.data(), sb);
    std::vector<std::complex<float>> merged(sa.size());
    for (size_t i = 0; i < sa.size(); ++i) merged[i] = sb[i] + (sa[i] - sb[i]) * mask[i];
    const auto y = it->second->inverse(merged);
    return std::vector<float>(y.begin(), y.begin() + n);
}

inline std::vector<float> crossfade(const std::vector<std::vector<float>>& chunks,
                                    int hop, int chunk_len, int out_len) {
    const int total = chunk_len + hop * (static_cast<int>(chunks.size()) - 1);
    std::vector<float> out(total, 0.0f), wsum(total, 0.0f);
    const auto win = hann(chunk_len);
    for (size_t i = 0; i < chunks.size(); ++i) {
        const int s = static_cast<int>(i) * hop;
        for (int j = 0; j < chunk_len; ++j) {
            float w = win[j];
            if (i == 0 && j < hop) w = 1.0f;
            if (i + 1 == chunks.size() && j >= chunk_len - hop) w = 1.0f;
            out[s + j] += chunks[i][j] * w;
            wsum[s + j] += w;
        }
    }
    out.resize(out_len);
    wsum.resize(out_len);
    for (int i = 0; i < out_len; ++i) out[i] /= std::max(wsum[i], 1e-12f);
    return out;
}

}  // namespace lavasr
