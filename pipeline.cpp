/*
g++ -std=c++17 pipeline.cpp -O2 -pthread -o pipeline

  ОДИН эффект через конвейер (reader -> processor -> writer):
    ./pipeline -i input.jpg -o out.png -negativ -threads 4
    ./pipeline -i input.jpg -o out.png -rotate 90 -threads 6
    ./pipeline -i input.jpg -o out.png -blur 5 -threads 8
    ./pipeline -i input.jpg -o out_negativ.png -negativ -threads 8
    ./pipeline -i input.jpg -o out_negativ.png -negativ -threads 8
    ./pipeline -i input.jpg -o out_any.png -rotate_any 37.5 -threads 8
    ./pipeline -i input.jpg -o out_cont.png -contrast 30 -threads 6
    ./pipeline -i input.jpg -o out_aff_s.png -affine_scale 1.5 1.5 -threads 10
    ./pipeline -i input.jpg -o out_aff_s.png -affine_scale 1.5 1.5 -threads 10



  ПОСЛЕДОВАТЕЛЬНОЕ применение нескольих эффектов к ОДНОМУ изображению (-seq):
    ./pipeline -seq -i input.jpg -negativ -blur 5 -rotate_any 37.5 -brightness 40 -threads 8
    ./pipeline -seq -i input.jpg -negativ -blur 5 -rotate_any 37.5 -brightness 40 -threads 4
    ./pipeline -seq -i input.jpg -negativ -blur 5 -rotate_any 37.5 -brightness 40 -threads 2



 ┌──────────┐      ┌──────────────┐      ┌─────────┐
 │  Reader  │ ---> │  Processor   │ ---> │ Writer  │
 | 1 thread |      | 1+N theread  |      |1 thread |  
 └──────────┘      └──────────────┘      └─────────┘
 default N = hardware_concurrency() - 2
 Load image
    ↓
    Effect 1 → save step 1
    ↓
    Effect 2 → save step 2
    ↓
    Effect 3 → save step 3
    ↓
    ...
    ↓
    save final.png
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct Image {
    int width = 0, height = 0, channels = 0;
    std::vector<unsigned char> data;
};

// ================= ОБЩИЕ ФУНКЦИИ ЗАГРУЗКИ / СОХРАНЕНИЯ =================

bool loadImage(const std::string &filename, Image &img) {
    unsigned char *pixels = stbi_load(filename.c_str(),
                                      &img.width,
                                      &img.height,
                                      &img.channels,
                                      3); // принудительно RGB
    if (!pixels) return false;

    img.channels = 3;
    img.data.assign(pixels, pixels + img.width * img.height * 3);
    stbi_image_free(pixels);
    return true;
}

bool saveImage(const std::string &filename, const Image &img) {
    auto pos = filename.find_last_of('.');
    if (pos == std::string::npos) return false;

    std::string ext = filename.substr(pos + 1);
    for (auto &c : ext) c = std::tolower(static_cast<unsigned char>(c));

    if (ext == "png")
        return stbi_write_png(filename.c_str(),
                              img.width, img.height,
                              3, img.data.data(),
                              img.width * 3);

    if (ext == "bmp")
        return stbi_write_bmp(filename.c_str(),
                              img.width, img.height,
                              3, img.data.data());

    if (ext == "jpg" || ext == "jpeg")
        return stbi_write_jpg(filename.c_str(),
                              img.width, img.height,
                              3, img.data.data(),
                              95);

    if (ext == "tga")
        return stbi_write_tga(filename.c_str(),
                              img.width, img.height,
                              3, img.data.data());

    std::cerr << "Unsupported output format: " << ext << "\n";
    return false;
}

// число потоков обработки, если пользователь не указал
int resolveWorkerThreads(int requested) {
    if (requested > 0) return requested;
    unsigned int total = std::thread::hardware_concurrency();
    if (total == 0) total = 4;
    if (total < 3) total = 3;
    int worker = static_cast<int>(total) - 2; // вариант А
    if (worker < 1) worker = 1;
    return worker;
}

// =========================== NEGATIVE ===========================

void invert_chunk(std::vector<unsigned char> &d, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i)
        d[i] = 255 - d[i];
}

void invert_multithread(Image &img, int requestedThreads) {
    int threadCount = resolveWorkerThreads(requestedThreads);
    size_t total = img.data.size();
    if (total == 0) return;
    size_t chunk = total / threadCount;
    if (chunk == 0) {
        invert_chunk(img.data, 0, total);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(threadCount);
    for (int t = 0; t < threadCount; ++t) {
        size_t start = t * chunk;
        size_t end = (t == threadCount - 1) ? total : start + chunk;
        pool.emplace_back(invert_chunk, std::ref(img.data), start, end);
    }
    for (auto &th : pool) th.join();
}

// =========================== BLUR (Gaussian 1D по X) ===========================

std::vector<float> build_kernel(int radius) {
    int size = radius * 2 + 1;
    std::vector<float> kernel(size);
    float sigma = radius / 2.0f;
    if (sigma <= 0.0f) sigma = 1.0f;
    float sum = 0.0f;

    for (int i = 0; i < size; ++i) {
        int x = i - radius;
        float v = std::exp(-(x * x) / (2 * sigma * sigma));
        kernel[i] = v;
        sum += v;
    }
    for (float &v : kernel) v /= sum;
    return kernel;
}

void blur_chunk(const Image &src, Image &dst,
                const std::vector<float> &kernel,
                int radius, int yStart, int yEnd) {
    int w = src.width;
    for (int y = yStart; y < yEnd; ++y) {
        for (int x = 0; x < w; ++x) {
            float r = 0, g = 0, b = 0;
            for (int k = -radius; k <= radius; ++k) {
                int xx = std::min(std::max(x + k, 0), w - 1);
                int idx = (y * w + xx) * 3;
                float kv = kernel[k + radius];
                r += src.data[idx] * kv;
                g += src.data[idx + 1] * kv;
                b += src.data[idx + 2] * kv;
            }
            int outIdx = (y * w + x) * 3;
            dst.data[outIdx]     = static_cast<unsigned char>(r);
            dst.data[outIdx + 1] = static_cast<unsigned char>(g);
            dst.data[outIdx + 2] = static_cast<unsigned char>(b);
        }
    }
}

void gaussian_blur(Image &img, int radius, int requestedThreads) {
    if (radius < 1) return;
    Image temp = img;
    std::vector<float> kernel = build_kernel(radius);

    int threadCount = resolveWorkerThreads(requestedThreads);
    int rowsPerThread = img.height / threadCount;
    if (rowsPerThread == 0) {
        blur_chunk(temp, img, kernel, radius, 0, img.height);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(threadCount);
    for (int t = 0; t < threadCount; ++t) {
        int yStart = t * rowsPerThread;
        int yEnd = (t == threadCount - 1) ? img.height : yStart + rowsPerThread;
        pool.emplace_back(blur_chunk, std::cref(temp), std::ref(img),
                          std::cref(kernel), radius, yStart, yEnd);
    }
    for (auto &th : pool) th.join();
}

// =========================== BRIGHTNESS ===========================

static inline unsigned char clampByteInt(int v) {
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return static_cast<unsigned char>(v);
}

void brightness_chunk(std::vector<unsigned char> &data,
                      size_t start, size_t end,
                      int brightness) {
    for (size_t i = start; i < end; ++i) {
        int v = static_cast<int>(data[i]) + brightness;
        data[i] = clampByteInt(v);
    }
}

void adjust_brightness(Image &img, int brightness, int requestedThreads) {
    if (brightness == 0) return;
    if (brightness < -255) brightness = -255;
    if (brightness >  255) brightness =  255;

    int threadCount = resolveWorkerThreads(requestedThreads);
    size_t total = img.data.size();
    if (total == 0) return;
    size_t chunk = total / threadCount;
    if (chunk == 0) {
        brightness_chunk(img.data, 0, total, brightness);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(threadCount);
    for (int t = 0; t < threadCount; ++t) {
        size_t start = static_cast<size_t>(t) * chunk;
        size_t end   = (t == threadCount - 1) ? total : start + chunk;
        pool.emplace_back(brightness_chunk,
                          std::ref(img.data), start, end, brightness);
    }
    for (auto &th : pool) th.join();
}

// =========================== CONTRAST ===========================

static inline unsigned char clampByteFloat(float v) {
    if (v < 0.0f)   v = 0.0f;
    if (v > 255.0f) v = 255.0f;
    return static_cast<unsigned char>(v);
}

void contrast_chunk(std::vector<unsigned char> &data,
                    size_t start, size_t end,
                    float alpha) {
    for (size_t i = start; i < end; ++i) {
        float v = static_cast<float>(data[i]);
        v = (v - 128.0f) * alpha + 128.0f;
        data[i] = clampByteFloat(v);
    }
}

void adjust_contrast(Image &img, int contrast, int requestedThreads) {
    if (contrast < -100) contrast = -100;
    if (contrast >  100) contrast =  100;

    float alpha = 1.0f + static_cast<float>(contrast) / 100.0f;
    if (std::fabs(alpha - 1.0f) < 1e-6f) return;

    int threadCount = resolveWorkerThreads(requestedThreads);
    size_t total = img.data.size();
    if (total == 0) return;
    size_t chunk = total / threadCount;
    if (chunk == 0) {
        contrast_chunk(img.data, 0, total, alpha);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(threadCount);
    for (int t = 0; t < threadCount; ++t) {
        size_t start = static_cast<size_t>(t) * chunk;
        size_t end   = (t == threadCount - 1) ? total : start + chunk;
        pool.emplace_back(contrast_chunk,
                          std::ref(img.data), start, end, alpha);
    }
    for (auto &th : pool) th.join();
}

// =========================== ROTATE 0/90/180/270 ===========================

void rotate180_chunk(const std::vector<unsigned char> &src,
                     std::vector<unsigned char> &dst,
                     size_t start, size_t end) {
    size_t pixelCount = src.size() / 3;
    for (size_t i = start; i < end; ++i) {
        size_t rev = pixelCount - 1 - i;
        dst[rev * 3]     = src[i * 3];
        dst[rev * 3 + 1] = src[i * 3 + 1];
        dst[rev * 3 + 2] = src[i * 3 + 2];
    }
}

void rotate90_270_chunk(const Image &src, Image &dst,
                        size_t startRow, size_t endRow,
                        bool clockwise) {
    int w = src.width;
    int h = src.height;
    for (size_t y = startRow; y < endRow; ++y) {
        for (int x = 0; x < w; ++x) {
            int nx, ny;
            if (clockwise) {
                nx = h - 1 - static_cast<int>(y);
                ny = x;
            } else {
                nx = static_cast<int>(y);
                ny = w - 1 - x;
            }
            size_t srcIdx = (y * w + x) * 3;
            size_t dstIdx = (ny * dst.width + nx) * 3;
            dst.data[dstIdx]     = src.data[srcIdx];
            dst.data[dstIdx + 1] = src.data[srcIdx + 1];
            dst.data[dstIdx + 2] = src.data[srcIdx + 2];
        }
    }
}

void rotateImage(Image &img, int angle, int requestedThreads) {
    if (angle == 0) return;
    if (!(angle == 90 || angle == 180 || angle == 270)) {
        std::cerr << "Unsupported angle (use 0,90,180,270)\n";
        return;
    }

    int threadCount = resolveWorkerThreads(requestedThreads);
    Image out;

    if (angle == 180) {
        out.width = img.width;
        out.height = img.height;
        out.channels = 3;
        out.data.resize(img.data.size());

        size_t pixels = img.data.size() / 3;
        size_t chunk = pixels / threadCount;
        if (chunk == 0) {
            rotate180_chunk(img.data, out.data, 0, pixels);
            img = out;
            return;
        }

        std::vector<std::thread> pool;
        pool.reserve(threadCount);
        for (int t = 0; t < threadCount; ++t) {
            size_t start = t * chunk;
            size_t end = (t == threadCount - 1) ? pixels : start + chunk;
            pool.emplace_back(rotate180_chunk,
                              std::cref(img.data), std::ref(out.data),
                              start, end);
        }
        for (auto &th : pool) th.join();
        img = out;
        return;
    }

    bool cw = (angle == 90);
    out.width = img.height;
    out.height = img.width;
    out.channels = 3;
    out.data.resize(img.data.size());

    size_t rows = img.height;
    size_t chunk = rows / threadCount;
    if (chunk == 0) {
        rotate90_270_chunk(img, out, 0, rows, cw);
        img = out;
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(threadCount);
    for (int t = 0; t < threadCount; ++t) {
        size_t start = t * chunk;
        size_t end = (t == threadCount - 1) ? rows : start + chunk;
        pool.emplace_back(rotate90_270_chunk,
                          std::cref(img), std::ref(out),
                          start, end, cw);
    }
    for (auto &th : pool) th.join();
    img = out;
}

// =========================== ROTATE_ANY (любой угол) ===========================

void getPixelBilinear(const Image &img, float x, float y, unsigned char *out) {
    if (x < 0 || y < 0 || x >= img.width - 1 || y >= img.height - 1) {
        out[0] = out[1] = out[2] = 0;
        return;
    }

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    float dx = x - x0;
    float dy = y - y0;

    int idx = (y0 * img.width + x0) * 3;

    unsigned char p00[3] = {
        img.data[idx], img.data[idx + 1], img.data[idx + 2]
    };
    unsigned char p10[3] = {
        img.data[idx + 3], img.data[idx + 4], img.data[idx + 5]
    };
    unsigned char p01[3] = {
        img.data[idx + img.width * 3],
        img.data[idx + img.width * 3 + 1],
        img.data[idx + img.width * 3 + 2]
    };
    unsigned char p11[3] = {
        img.data[idx + img.width * 3 + 3],
        img.data[idx + img.width * 3 + 4],
        img.data[idx + img.width * 3 + 5]
    };

    for (int c = 0; c < 3; ++c) {
        float a = p00[c] * (1.0f - dx) + p10[c] * dx;
        float b = p01[c] * (1.0f - dx) + p11[c] * dx;
        out[c] = static_cast<unsigned char>(a * (1.0f - dy) + b * dy);
    }
}

void rotateChunkAny(const Image &src, Image &dst,
                    float cosA, float sinA,
                    float cx, float cy,
                    float ncx, float ncy,
                    int yStart, int yEnd) {
    for (int y = yStart; y < yEnd; ++y) {
        for (int x = 0; x < dst.width; ++x) {
            float dx = x - ncx;
            float dy = y - ncy;

            float sx =  cosA * dx + sinA * dy + cx;
            float sy = -sinA * dx + cosA * dy + cy;

            unsigned char rgb[3];
            getPixelBilinear(src, sx, sy, rgb);

            int idx = (y * dst.width + x) * 3;
            dst.data[idx]   = rgb[0];
            dst.data[idx+1] = rgb[1];
            dst.data[idx+2] = rgb[2];
        }
    }
}

void rotateAnyAngle(Image &img, float angleDeg, int requestedThreads) {
    float angle = angleDeg * 3.14159265358979323846f / 180.0f;
    float cosA = std::cos(angle);
    float sinA = std::sin(angle);

    int w = img.width;
    int h = img.height;

    int nw = static_cast<int>(std::fabs(w * cosA) + std::fabs(h * sinA));
    int nh = static_cast<int>(std::fabs(w * sinA) + std::fabs(h * cosA));
    if (nw < 1) nw = 1;
    if (nh < 1) nh = 1;

    Image out;
    out.width = nw;
    out.height = nh;
    out.channels = 3;
    out.data.resize(nw * nh * 3);

    float cx = w / 2.0f;
    float cy = h / 2.0f;
    float ncx = nw / 2.0f;
    float ncy = nh / 2.0f;

    int threadCount = resolveWorkerThreads(requestedThreads);
    int chunk = nh / threadCount;
    if (chunk == 0) {
        rotateChunkAny(img, out, cosA, sinA, cx, cy, ncx, ncy, 0, nh);
        img = out;
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(threadCount);
    for (int t = 0; t < threadCount; ++t) {
        int ys = t * chunk;
        int ye = (t == threadCount - 1) ? nh : ys + chunk;
        pool.emplace_back(rotateChunkAny,
                          std::cref(img), std::ref(out),
                          cosA, sinA,
                          cx, cy,
                          ncx, ncy,
                          ys, ye);
    }
    for (auto &th : pool) th.join();
    img = out;
}

// =========================== АФФИННЫЕ ПРЕОБРАЗОВАНИЯ ===========================

struct AffineMatrix {
    double m[2][3];
    AffineMatrix() {
        m[0][0] = 1; m[0][1] = 0; m[0][2] = 0;
        m[1][0] = 0; m[1][1] = 1; m[1][2] = 0;
    }
};

AffineMatrix createRotationMatrix(double angleDeg) {
    AffineMatrix mat;
    double rad = angleDeg * 3.14159265358979323846 / 180.0;
    double c = std::cos(rad);
    double s = std::sin(rad);
    mat.m[0][0] = c;  mat.m[0][1] = -s; mat.m[0][2] = 0;
    mat.m[1][0] = s;  mat.m[1][1] =  c; mat.m[1][2] = 0;
    return mat;
}

AffineMatrix createScaleMatrix(double sx, double sy) {
    AffineMatrix mat;
    mat.m[0][0] = sx; mat.m[0][1] = 0;  mat.m[0][2] = 0;
    mat.m[1][0] = 0;  mat.m[1][1] = sy; mat.m[1][2] = 0;
    return mat;
}

AffineMatrix createShearMatrix(double shx, double shy) {
    AffineMatrix mat;
    mat.m[0][0] = 1;   mat.m[0][1] = shx; mat.m[0][2] = 0;
    mat.m[1][0] = shy; mat.m[1][1] = 1;   mat.m[1][2] = 0;
    return mat;
}

AffineMatrix createTranslationMatrix(double tx, double ty) {
    AffineMatrix mat;
    mat.m[0][0] = 1; mat.m[0][1] = 0; mat.m[0][2] = tx;
    mat.m[1][0] = 0; mat.m[1][1] = 1; mat.m[1][2] = ty;
    return mat;
}

void applyAffineToPixel(const Image &src, Image &dst,
                        int x, int y, const AffineMatrix &mat) {
    double centerX = dst.width / 2.0;
    double centerY = dst.height / 2.0;
    double relX = x - centerX;
    double relY = y - centerY;

    double srcX = mat.m[0][0] * relX + mat.m[0][1] * relY + mat.m[0][2] + src.width / 2.0;
    double srcY = mat.m[1][0] * relX + mat.m[1][1] * relY + mat.m[1][2] + src.height / 2.0;

    int x0 = static_cast<int>(std::floor(srcX));
    int y0 = static_cast<int>(std::floor(srcY));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    double dx = srcX - x0;
    double dy = srcY - y0;

    if (x0 < 0 || x1 >= src.width || y0 < 0 || y1 >= src.height) {
        for (int c = 0; c < 3; ++c)
            dst.data[(y * dst.width + x) * 3 + c] = 0;
        return;
    }

    for (int c = 0; c < 3; ++c) {
        double top =
            src.data[(y0 * src.width + x0) * 3 + c] * (1.0 - dx) +
            src.data[(y0 * src.width + x1) * 3 + c] * dx;
        double bottom =
            src.data[(y1 * src.width + x0) * 3 + c] * (1.0 - dx) +
            src.data[(y1 * src.width + x1) * 3 + c] * dx;
        double value = top * (1.0 - dy) + bottom * dy;
        value = std::max(0.0, std::min(255.0, value));
        dst.data[(y * dst.width + x) * 3 + c] = static_cast<unsigned char>(value);
    }
}

void affine_chunk(const Image &src, Image &dst,
                  const AffineMatrix &mat,
                  int startY, int endY) {
    for (int y = startY; y < endY; ++y) {
        if (y < 0 || y >= dst.height) continue;
        for (int x = 0; x < dst.width; ++x)
            applyAffineToPixel(src, dst, x, y, mat);
    }
}

void affineTransform(const Image &src,
                     Image &dst,
                     const AffineMatrix &mat,
                     int newWidth,
                     int newHeight,
                     int requestedThreads) {
    if (newWidth < 1) newWidth = 1;
    if (newHeight < 1) newHeight = 1;

    dst.width = newWidth;
    dst.height = newHeight;
    dst.channels = 3;
    dst.data.assign(newWidth * newHeight * 3, 0);

    int threadCount = resolveWorkerThreads(requestedThreads);
    int rowsPerThread = (newHeight + threadCount - 1) / threadCount;

    std::vector<std::thread> pool;
    pool.reserve(threadCount);
    for (int t = 0; t < threadCount; ++t) {
        int startY = t * rowsPerThread;
        int endY   = std::min(startY + rowsPerThread, newHeight);
        if (startY >= endY) break;
        pool.emplace_back(affine_chunk,
                          std::cref(src), std::ref(dst),
                          std::cref(mat), startY, endY);
    }
    for (auto &th : pool) th.join();
}

// =========================== ОПИСАНИЕ ЭФФЕКТОВ ===========================

enum class EffectType {
    NONE,
    NEGATIVE,
    ROTATE_FIXED,
    ROTATE_ANY,
    BLUR,
    BRIGHTNESS,
    CONTRAST,
    AFFINE_ROTATE,
    AFFINE_SCALE,
    AFFINE_SHEAR,
    AFFINE_TRANSLATE
};

struct EffectConfig {
    EffectType type = EffectType::NONE;
    int rotateAngle = 0;
    float rotateAny = 0.0f;
    int blurRadius = 0;
    int brightness = 0;
    int contrast = 0;
    double p1 = 0.0, p2 = 0.0;
};

std::string effectName(const EffectConfig &cfg) {
    switch (cfg.type) {
        case EffectType::NEGATIVE:         return "negativ";
        case EffectType::ROTATE_FIXED:     return "rotate";
        case EffectType::ROTATE_ANY:       return "rotate_any";
        case EffectType::BLUR:             return "blur";
        case EffectType::BRIGHTNESS:       return "brightness";
        case EffectType::CONTRAST:         return "contrast";
        case EffectType::AFFINE_ROTATE:    return "affine_rotate";
        case EffectType::AFFINE_SCALE:     return "affine_scale";
        case EffectType::AFFINE_SHEAR:     return "affine_shear";
        case EffectType::AFFINE_TRANSLATE: return "affine_translate";
        case EffectType::NONE:             return "none";
    }
    return "unknown";
}

void applyEffect(Image &img, const EffectConfig &cfg, int workerThreads) {
    switch (cfg.type) {
        case EffectType::NEGATIVE:
            std::cout << "Effect: NEGATIVE\n";
            invert_multithread(img, workerThreads);
            break;
        case EffectType::BLUR:
            std::cout << "Effect: BLUR radius=" << cfg.blurRadius << "\n";
            gaussian_blur(img, cfg.blurRadius, workerThreads);
            break;
        case EffectType::BRIGHTNESS:
            std::cout << "Effect: BRIGHTNESS value=" << cfg.brightness << "\n";
            adjust_brightness(img, cfg.brightness, workerThreads);
            break;
        case EffectType::CONTRAST:
            std::cout << "Effect: CONTRAST value=" << cfg.contrast << "\n";
            adjust_contrast(img, cfg.contrast, workerThreads);
            break;
        case EffectType::ROTATE_FIXED:
            std::cout << "Effect: ROTATE_FIXED angle=" << cfg.rotateAngle << "\n";
            rotateImage(img, cfg.rotateAngle, workerThreads);
            break;
        case EffectType::ROTATE_ANY:
            std::cout << "Effect: ROTATE_ANY angle=" << cfg.rotateAny << "\n";
            rotateAnyAngle(img, cfg.rotateAny, workerThreads);
            break;
        case EffectType::AFFINE_ROTATE: {
            std::cout << "Effect: AFFINE_ROTATE angle=" << cfg.p1 << "\n";
            Image srcCopy = img;
            Image dst;
            AffineMatrix mat = createRotationMatrix(cfg.p1);
            affineTransform(srcCopy, dst, mat, srcCopy.width, srcCopy.height, workerThreads);
            img = std::move(dst);
            break;
        }
        case EffectType::AFFINE_SCALE: {
            std::cout << "Effect: AFFINE_SCALE sx=" << cfg.p1 << " sy=" << cfg.p2 << "\n";
            Image srcCopy = img;
            Image dst;
            AffineMatrix mat = createScaleMatrix(cfg.p1, cfg.p2);
            int newW = static_cast<int>(srcCopy.width * cfg.p1);
            int newH = static_cast<int>(srcCopy.height * cfg.p2);
            if (newW < 1) newW = 1;
            if (newH < 1) newH = 1;
            affineTransform(srcCopy, dst, mat, newW, newH, workerThreads);
            img = std::move(dst);
            break;
        }
        case EffectType::AFFINE_SHEAR: {
            std::cout << "Effect: AFFINE_SHEAR shx=" << cfg.p1 << " shy=" << cfg.p2 << "\n";
            Image srcCopy = img;
            Image dst;
            AffineMatrix mat = createShearMatrix(cfg.p1, cfg.p2);
            affineTransform(srcCopy, dst, mat, srcCopy.width, srcCopy.height, workerThreads);
            img = std::move(dst);
            break;
        }
        case EffectType::AFFINE_TRANSLATE: {
            std::cout << "Effect: AFFINE_TRANSLATE tx=" << cfg.p1 << " ty=" << cfg.p2 << "\n";
            Image srcCopy = img;
            Image dst;
            AffineMatrix mat = createTranslationMatrix(cfg.p1, cfg.p2);
            int newW = srcCopy.width + static_cast<int>(std::fabs(cfg.p1)) + 10;
            int newH = srcCopy.height + static_cast<int>(std::fabs(cfg.p2)) + 10;
            affineTransform(srcCopy, dst, mat, newW, newH, workerThreads);
            img = std::move(dst);
            break;
        }
        case EffectType::NONE:
        default:
            std::cerr << "No effect selected\n";
            break;
    }
}

// =========================== ПАРСИНГ АРГУМЕНТОВ ===========================

void printUsage() {
    std::cout <<
        "Usage (single effect, конвейер):\n"
        "  pipeline -i <input> -o <output> [effect] -threads <N>\n\n"
        "Usage (sequence mode, один входной файл, много эффектов):\n"
        "  pipeline -seq -i <input> [effects...] -threads <N>\n\n"
        "Effects (стиль 1):\n"
        "  -negativ\n"
        "  -rotate <0|90|180|270>\n"
        "  -rotate_any <angle_deg>\n"
        "  -blur <radius>\n"
        "  -brightness <value>          ([-255;255])\n"
        "  -contrast <value>            ([-100;100])\n"
        "  -affine_rotate <angle_deg>\n"
        "  -affine_scale <sx> <sy>\n"
        "  -affine_shear <shx> <shy>\n"
        "  -affine_translate <tx> <ty>\n\n"
        "Дополнительно:\n"
        "  -threads <N>   — число потоков обработки (по умолчанию hw_concurrency-2).\n"
        "  В режиме -seq выходные файлы кладутся в папку <input_basename>_pipeline/.\n";
}

bool parseArgs(int argc, char **argv,
               std::string &input,
               std::string &output,
               bool &seqMode,
               EffectConfig &singleCfg,
               std::vector<EffectConfig> &seqCfg,
               int &workerThreads) {
    seqMode = false;
    workerThreads = 0;
    bool singleEffectSet = false;

    // первый проход: узнаём -seq, -i, -o, -threads
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-seq")) {
            seqMode = true;
        } else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) {
            input = argv[++i];
        } else if (!std::strcmp(argv[i], "-o") && i + 1 < argc) {
            output = argv[++i];
        } else if (!std::strcmp(argv[i], "-threads") && i + 1 < argc) {
            workerThreads = std::atoi(argv[++i]);
        }
    }

    // второй проход — парсим эффекты
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-seq") || !std::strcmp(argv[i], "-i") ||
            !std::strcmp(argv[i], "-o") || !std::strcmp(argv[i], "-threads")) {
            if (!std::strcmp(argv[i], "-i") || !std::strcmp(argv[i], "-o") || !std::strcmp(argv[i], "-threads")) {
                ++i; // пропускаем значение
            }
            continue;
        }

        auto pushEffect = [&](const EffectConfig &cfg) {
            if (seqMode) {
                seqCfg.push_back(cfg);
            } else {
                if (singleEffectSet) {
                    std::cerr << "Only one effect allowed in non-seq mode\n";
                    return false;
                }
                singleCfg = cfg;
                singleEffectSet = true;
            }
            return true;
        };

        if (!std::strcmp(argv[i], "-negativ")) {
            EffectConfig cfg;
            cfg.type = EffectType::NEGATIVE;
            if (!pushEffect(cfg)) return false;
        } else if (!std::strcmp(argv[i], "-rotate") && i + 1 < argc) {
            EffectConfig cfg;
            cfg.type = EffectType::ROTATE_FIXED;
            cfg.rotateAngle = std::atoi(argv[++i]);
            if (!pushEffect(cfg)) return false;
        } else if (!std::strcmp(argv[i], "-rotate_any") && i + 1 < argc) {
            EffectConfig cfg;
            cfg.type = EffectType::ROTATE_ANY;
            cfg.rotateAny = std::stof(argv[++i]);
            if (!pushEffect(cfg)) return false;
        } else if (!std::strcmp(argv[i], "-blur") && i + 1 < argc) {
            EffectConfig cfg;
            cfg.type = EffectType::BLUR;
            cfg.blurRadius = std::atoi(argv[++i]);
            if (!pushEffect(cfg)) return false;
        } else if (!std::strcmp(argv[i], "-brightness") && i + 1 < argc) {
            EffectConfig cfg;
            cfg.type = EffectType::BRIGHTNESS;
            cfg.brightness = std::atoi(argv[++i]);
            if (!pushEffect(cfg)) return false;
        } else if (!std::strcmp(argv[i], "-contrast") && i + 1 < argc) {
            EffectConfig cfg;
            cfg.type = EffectType::CONTRAST;
            cfg.contrast = std::atoi(argv[++i]);
            if (!pushEffect(cfg)) return false;
        } else if (!std::strcmp(argv[i], "-affine_rotate") && i + 1 < argc) {
            EffectConfig cfg;
            cfg.type = EffectType::AFFINE_ROTATE;
            cfg.p1 = std::atof(argv[++i]);
            if (!pushEffect(cfg)) return false;
        } else if (!std::strcmp(argv[i], "-affine_scale") && i + 2 < argc) {
            EffectConfig cfg;
            cfg.type = EffectType::AFFINE_SCALE;
            cfg.p1 = std::atof(argv[++i]);
            cfg.p2 = std::atof(argv[++i]);
            if (!pushEffect(cfg)) return false;
        } else if (!std::strcmp(argv[i], "-affine_shear") && i + 2 < argc) {
            EffectConfig cfg;
            cfg.type = EffectType::AFFINE_SHEAR;
            cfg.p1 = std::atof(argv[++i]);
            cfg.p2 = std::atof(argv[++i]);
            if (!pushEffect(cfg)) return false;
        } else if (!std::strcmp(argv[i], "-affine_translate") && i + 2 < argc) {
            EffectConfig cfg;
            cfg.type = EffectType::AFFINE_TRANSLATE;
            cfg.p1 = std::atof(argv[++i]);
            cfg.p2 = std::atof(argv[++i]);
            if (!pushEffect(cfg)) return false;
        } else {
            std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
            return false;
        }
    }

    if (input.empty()) {
        std::cerr << "Input not specified (-i).\n";
        return false;
    }

    if (!seqMode && output.empty()) {
        std::cerr << "Output not specified (-o).\n";
        return false;
    }

    if (seqMode) {
        if (seqCfg.empty()) {
            std::cerr << "No effects specified for -seq mode.\n";
            return false;
        }
    } else {
        if (!singleEffectSet || singleCfg.type == EffectType::NONE) {
            std::cerr << "No effect specified.\n";
            return false;
        }
    }

    return true;
}

// =========================== РЕЖИМ SEQ (ОДНО ФОТО → МНОГО ЭФФЕКТОВ) ===========================

void runSequentialMode(const std::string &inputPath,
                       const std::vector<EffectConfig> &seqCfg,
                       int workerThreads) {
    Image img;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!loadImage(inputPath, img)) {
        std::cerr << "Unable to load image: " << inputPath << "\n";
        return;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double loadMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::filesystem::path inPath(inputPath);
    std::string base = inPath.stem().string();
    std::string outDirName = base + "_pipeline";
    std::filesystem::create_directories(outDirName);

    std::cout << "Sequential mode: input=" << inputPath
              << ", out dir=" << outDirName << "\n";

    Image current = img;
    double totalProc = 0.0;

    for (size_t i = 0; i < seqCfg.size(); ++i) {
        const EffectConfig &cfg = seqCfg[i];
        std::string name = effectName(cfg);
        std::cout << "--- Step " << (i + 1) << " : " << name << " ---\n";

        auto s0 = std::chrono::high_resolution_clock::now();
        applyEffect(current, cfg, workerThreads);
        auto s1 = std::chrono::high_resolution_clock::now();
        double stepMs = std::chrono::duration<double, std::milli>(s1 - s0).count();
        totalProc += stepMs;

        std::string outFile = outDirName + "/step_" + std::to_string(i + 1) + "_" + name + ".png";
        std::filesystem::path outPath = std::filesystem::path(outFile);
        if (!saveImage(outPath.string(), current)) {
            std::cerr << "Unable to save step image: " << outPath << "\n";
        } else {
            std::cout << "Saved: " << outPath << "\n";
        }
        std::cout << "Step time: " << stepMs << " ms\n";
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::string finalFile = outDirName + "/final.png";
    if (!saveImage(finalFile, current)) {
        std::cerr << "Unable to save final image: " << finalFile << "\n";
    } else {
        std::cout << "Final image: " << finalFile << "\n";
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    double saveFinalMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double totalMs = std::chrono::duration<double, std::milli>(t3 - t0).count();

    std::cout << "=== Timing (seq mode) ===\n";
    std::cout << "Load:       " << loadMs      << " ms\n";
    std::cout << "Process sum:" << totalProc   << " ms\n";
    std::cout << "Final save: " << saveFinalMs << " ms\n";
    std::cout << "Total:      " << totalMs     << " ms\n";
}

// =========================== КОНВЕЙЕР (ОДИН ЭФФЕКТ) ===========================

Image g_inputImage;
Image g_outputImage;

std::mutex g_m1;
std::condition_variable g_cv1;
bool g_hasInput = false;
bool g_inputError = false;

std::mutex g_m2;
std::condition_variable g_cv2;
bool g_hasOutput = false;
bool g_outputError = false;

double g_readMs = 0.0;
double g_procMs = 0.0;
double g_writeMs = 0.0;

void readerThreadFunc(const std::string &inputPath) {
    auto t0 = std::chrono::high_resolution_clock::now();
    Image local;
    bool ok = loadImage(inputPath, local);
    auto t1 = std::chrono::high_resolution_clock::now();
    g_readMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    {
        std::lock_guard<std::mutex> lk(g_m1);
        if (!ok) {
            g_inputError = true;
        } else {
            g_inputImage = std::move(local);
            g_inputError = false;
        }
        g_hasInput = true;
    }
    g_cv1.notify_one();
}

void processorThreadFunc(const EffectConfig cfg, int workerThreads) {
    std::unique_lock<std::mutex> lk(g_m1);
    g_cv1.wait(lk, [] { return g_hasInput; });

    if (g_inputError) {
        {
            std::lock_guard<std::mutex> lk2(g_m2);
            g_outputError = true;
            g_hasOutput = true;
        }
        g_cv2.notify_one();
        return;
    }

    Image local = std::move(g_inputImage);
    lk.unlock();

    auto t0 = std::chrono::high_resolution_clock::now();
    applyEffect(local, cfg, workerThreads);
    auto t1 = std::chrono::high_resolution_clock::now();
    g_procMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    {
        std::lock_guard<std::mutex> lk2(g_m2);
        g_outputImage = std::move(local);
        g_outputError = false;
        g_hasOutput = true;
    }
    g_cv2.notify_one();
}

void writerThreadFunc(const std::string &outputPath) {
    std::unique_lock<std::mutex> lk(g_m2);
    g_cv2.wait(lk, [] { return g_hasOutput; });

    if (g_outputError) {
        std::cerr << "Processing failed, nothing to write.\n";
        return;
    }

    Image local = std::move(g_outputImage);
    lk.unlock();

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = saveImage(outputPath, local);
    auto t1 = std::chrono::high_resolution_clock::now();
    g_writeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ok) {
        std::cerr << "Unable to save image: " << outputPath << "\n";
    }
}

// =========================== MAIN ===========================

int main(int argc, char **argv) {
    std::string inputPath, outputPath;
    bool seqMode = false;
    EffectConfig singleCfg;
    std::vector<EffectConfig> seqCfg;
    int workerThreads = 0;

    if (!parseArgs(argc, argv, inputPath, outputPath,
                   seqMode, singleCfg, seqCfg, workerThreads)) {
        printUsage();
        return 1;
    }

    workerThreads = resolveWorkerThreads(workerThreads);
    std::cout << "Worker threads (processing): " << workerThreads << "\n";

    if (seqMode) {
        runSequentialMode(inputPath, seqCfg, workerThreads);
        return 0;
    }

    auto tTotal0 = std::chrono::high_resolution_clock::now();

    std::thread reader(readerThreadFunc, inputPath);
    std::thread processor(processorThreadFunc, singleCfg, workerThreads);
    std::thread writer(writerThreadFunc, outputPath);

    reader.join();
    processor.join();
    writer.join();

    auto tTotal1 = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(tTotal1 - tTotal0).count();

    std::cout << "=== Timing (pipeline mode) ===\n";
    std::cout << "Read:       " << g_readMs  << " ms\n";
    std::cout << "Process:    " << g_procMs  << " ms\n";
    std::cout << "Write:      " << g_writeMs << " ms\n";
    std::cout << "Total:      " << totalMs   << " ms\n";

    return 0;
}
