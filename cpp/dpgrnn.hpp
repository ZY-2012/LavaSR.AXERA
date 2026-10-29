#pragma once
// Host DPGRNN for UL-UNAS (Pulsar2 cannot lower its large-batch GRU reliably).
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "assets.h"

namespace lavasr {

class DpGrnn {
public:
    DpGrnn() {
        for (int i = 0; i < kDpAssetCount; ++i)
            m_[kDpAssets[i].name] = {kDpAssets[i].data, kDpAssets[i].size};
    }

    std::vector<float> debug_rnn(const std::vector<float>& x, int b, int t, int half,
                                 const std::string& p) { return run_rnn(x, b, t, half, p); }
    std::vector<float> debug_grnn(const std::vector<float>& x, int b, int t, int c,
                                  const std::string& p) { return grnn(x, b, t, c, p); }
    std::vector<float> debug_linear(const std::vector<float>& x, int rows, int c,
                                    const std::string& p) { return linear(x, rows, c, p); }
    void debug_ln(std::vector<float>& x, int rows, int cols, const std::string& p) { layernorm(x, rows, cols, p); }

    // input  (B,C,T,F), output (B,C,T,F)
    std::vector<float> forward(const std::vector<float>& x, int b, int c, int t, int f) {
        std::vector<float> cur = x;
        for (int block = 0; block < 2; ++block) {
            const std::string pre = "b" + std::to_string(block) + ".";
            std::vector<float> intra(static_cast<size_t>(b) * t * f * c);
            for (int bi = 0; bi < b; ++bi)
                for (int ti = 0; ti < t; ++ti)
                    for (int fi = 0; fi < f; ++fi)
                        for (int ci = 0; ci < c; ++ci)
                            intra[((static_cast<size_t>(bi) * t + ti) * f + fi) * c + ci] =
                                cur[((static_cast<size_t>(bi) * c + ci) * t + ti) * f + fi];
            auto y = grnn(intra, b * t, f, c, pre + "intra_rnn");
            y = linear(y, b * t * f, c, pre + "intra_fc");
            layernorm(y, b * t, f * c, pre + "intra_ln");
            for (size_t i = 0; i < intra.size(); ++i) intra[i] += y[i];

            std::vector<float> inter(static_cast<size_t>(b) * f * t * c);
            for (int bi = 0; bi < b; ++bi)
                for (int fi = 0; fi < f; ++fi)
                    for (int ti = 0; ti < t; ++ti)
                        for (int ci = 0; ci < c; ++ci)
                            inter[((static_cast<size_t>(bi) * f + fi) * t + ti) * c + ci] =
                                intra[((static_cast<size_t>(bi) * t + ti) * f + fi) * c + ci];
            auto z = grnn(inter, b * f, t, c, pre + "inter_rnn");
            z = linear(z, b * f * t, c, pre + "inter_fc");
            std::vector<float> z4(static_cast<size_t>(b) * t * f * c);
            for (int bi = 0; bi < b; ++bi)
                for (int fi = 0; fi < f; ++fi)
                    for (int ti = 0; ti < t; ++ti)
                        for (int ci = 0; ci < c; ++ci)
                            z4[((static_cast<size_t>(bi) * t + ti) * f + fi) * c + ci] =
                                z[((static_cast<size_t>(bi) * f + fi) * t + ti) * c + ci];
            layernorm(z4, b * t, f * c, pre + "inter_ln");
            for (size_t i = 0; i < intra.size(); ++i) intra[i] += z4[i];
            std::vector<float> next(x.size());
            for (int bi = 0; bi < b; ++bi)
                for (int ti = 0; ti < t; ++ti)
                    for (int fi = 0; fi < f; ++fi)
                        for (int ci = 0; ci < c; ++ci)
                            next[((static_cast<size_t>(bi) * c + ci) * t + ti) * f + fi] =
                                intra[((static_cast<size_t>(bi) * t + ti) * f + fi) * c + ci];
            cur = next;
        }
        return cur;
    }

private:
    struct Entry { const float* data; int size; };
    std::unordered_map<std::string, Entry> m_;

    const float* get(const std::string& name) {
        auto it = m_.find(name);
        return it == m_.end() ? nullptr : it->second.data;
    }
    int size(const std::string& name) {
        auto it = m_.find(name);
        return it == m_.end() ? 0 : it->second.size;
    }
    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    static int hidden(const std::string& rnn, int n) {
        return static_cast<int>(std::lround(std::sqrt(n / 3.0)));
    }

    void gru(const std::vector<float>& x, int b, int t, int in, int h,
             const std::string& p, bool reverse, std::vector<float>& out,
             int out_off, int out_stride) {
        const float* w_ih = get(p + ".weight_ih_l0" + (reverse ? "_reverse" : ""));
        const float* w_hh = get(p + ".weight_hh_l0" + (reverse ? "_reverse" : ""));
        const float* b_ih = get(p + ".bias_ih_l0" + (reverse ? "_reverse" : ""));
        const float* b_hh = get(p + ".bias_hh_l0" + (reverse ? "_reverse" : ""));
        const int g = 3 * h;
        std::vector<float> gx(static_cast<size_t>(b) * t * g);
        for (int bi = 0; bi < b; ++bi) {
            for (int ti = 0; ti < t; ++ti) {
                const float* xp = x.data() + (static_cast<size_t>(bi) * t + ti) * in;
                float* gp = gx.data() + (static_cast<size_t>(bi) * t + ti) * g;
                for (int gi = 0; gi < g; ++gi) {
                    float acc = b_ih[gi];
                    const float* row = w_ih + static_cast<size_t>(gi) * in;
                    for (int k = 0; k < in; ++k) acc += xp[k] * row[k];
                    gp[gi] = acc;
                }
            }
        }
        std::vector<float> hs(static_cast<size_t>(b) * h, 0.0f);
        std::vector<float> nh(h);
        for (int step = 0; step < t; ++step) {
            const int ti = reverse ? (t - 1 - step) : step;
            for (int bi = 0; bi < b; ++bi) {
                const float* gp = gx.data() + (static_cast<size_t>(bi) * t + ti) * g;
                float* hp = hs.data() + static_cast<size_t>(bi) * h;
                for (int hi = 0; hi < h; ++hi) {
                    float gr = b_hh[hi], gz = b_hh[h + hi], gn = b_hh[2 * h + hi];
                    const float* rr = w_hh + static_cast<size_t>(hi) * h;
                    const float* rz = w_hh + static_cast<size_t>(h + hi) * h;
                    const float* rn = w_hh + static_cast<size_t>(2 * h + hi) * h;
                    for (int k = 0; k < h; ++k) {
                        gr += hp[k] * rr[k];
                        gz += hp[k] * rz[k];
                        gn += hp[k] * rn[k];
                    }
                    const float r = sigmoid(gp[hi] + gr);
                    const float z = sigmoid(gp[h + hi] + gz);
                    const float n = std::tanh(gp[2 * h + hi] + r * gn);
                    nh[hi] = (1.0f - z) * n + z * hp[hi];
                }
                for (int hi = 0; hi < h; ++hi) hp[hi] = nh[hi];
                float* op = out.data() + (static_cast<size_t>(bi) * t + ti) * out_stride + out_off;
                for (int hi = 0; hi < h; ++hi) op[hi] = nh[hi];
            }
        }
    }

    std::vector<float> run_rnn(const std::vector<float>& x, int b, int t, int half,
                               const std::string& p) {
        const bool bidir = get(p + ".weight_ih_l0_reverse") != nullptr;
        const int h = hidden(p, size(p + ".weight_hh_l0"));
        const int os = bidir ? 2 * h : h;
        std::vector<float> out(static_cast<size_t>(b) * t * os);
        gru(x, b, t, half, h, p, false, out, 0, os);
        if (bidir) gru(x, b, t, half, h, p, true, out, h, os);
        return out;
    }

    std::vector<float> grnn(const std::vector<float>& x, int b, int t, int c,
                            const std::string& p) {
        const int half = c / 2;
        const bool bidir1 = get(p + ".rnn1.weight_ih_l0_reverse") != nullptr;
        const bool bidir2 = get(p + ".rnn2.weight_ih_l0_reverse") != nullptr;
        const int h1 = hidden(p + ".rnn1", size(p + ".rnn1.weight_hh_l0"));
        const int h2 = hidden(p + ".rnn2", size(p + ".rnn2.weight_hh_l0"));
        const int o1 = bidir1 ? 2 * h1 : h1;
        const int o2 = bidir2 ? 2 * h2 : h2;
        std::vector<float> x1(static_cast<size_t>(b) * t * half);
        std::vector<float> x2(static_cast<size_t>(b) * t * half);
        for (int i = 0; i < b * t; ++i) {
            std::copy(x.begin() + static_cast<size_t>(i) * c,
                      x.begin() + static_cast<size_t>(i) * c + half,
                      x1.begin() + static_cast<size_t>(i) * half);
            std::copy(x.begin() + static_cast<size_t>(i) * c + half,
                      x.begin() + static_cast<size_t>(i) * c + c,
                      x2.begin() + static_cast<size_t>(i) * half);
        }
        auto y1 = run_rnn(x1, b, t, half, p + ".rnn1");
        auto y2 = run_rnn(x2, b, t, half, p + ".rnn2");
        std::vector<float> out(static_cast<size_t>(b) * t * (o1 + o2));
        for (int i = 0; i < b * t; ++i) {
            std::copy(y1.begin() + static_cast<size_t>(i) * o1,
                      y1.begin() + static_cast<size_t>(i + 1) * o1,
                      out.begin() + static_cast<size_t>(i) * (o1 + o2));
            std::copy(y2.begin() + static_cast<size_t>(i) * o2,
                      y2.begin() + static_cast<size_t>(i + 1) * o2,
                      out.begin() + static_cast<size_t>(i) * (o1 + o2) + o1);
        }
        return out;
    }

    std::vector<float> linear(const std::vector<float>& x, int rows, int c,
                              const std::string& p) {
        const float* w = get(p + ".weight");
        const float* bias = get(p + ".bias");
        const int oc = size(p + ".weight") / c;
        std::vector<float> y(static_cast<size_t>(rows) * oc);
        for (int r = 0; r < rows; ++r) {
            for (int o = 0; o < oc; ++o) {
                float acc = bias[o];
                const float* row = w + static_cast<size_t>(o) * c;
                for (int k = 0; k < c; ++k) acc += x[static_cast<size_t>(r) * c + k] * row[k];
                y[static_cast<size_t>(r) * oc + o] = acc;
            }
        }
        return y;
    }

    void layernorm(std::vector<float>& x, int rows, int cols, const std::string& p) {
        const float* w = get(p + ".weight");
        const float* bias = get(p + ".bias");
        for (int r = 0; r < rows; ++r) {
            float* q = x.data() + static_cast<size_t>(r) * cols;
            float mean = 0.0f, var = 0.0f;
            for (int i = 0; i < cols; ++i) mean += q[i];
            mean /= cols;
            for (int i = 0; i < cols; ++i) {
                const float d = q[i] - mean;
                var += d * d;
            }
            var /= cols;
            const float inv = 1.0f / std::sqrt(var + 1e-8f);
            for (int i = 0; i < cols; ++i) q[i] = (q[i] - mean) * inv * w[i] + bias[i];
        }
    }
};

}  // namespace lavasr
