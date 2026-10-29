#pragma once
#include <cstddef>

extern const float kMelFb[82000];
extern const float kMelWindow[2048];
extern const float kDenWindow[512];
extern const float kErbFc[12288];
extern const float kIerbFc[12288];
struct DpAsset { const char* name; const float* data; int size; };
extern const DpAsset kDpAssets[];
extern const int kDpAssetCount;
