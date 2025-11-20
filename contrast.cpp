// g++ contrast.cpp -O2 -pthread -o contrast
// ./contrast -i input.jpg -o out.png -c 30

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <cmath>

struct Image {
    int width, height, channels;
    std::vector<unsigned char> data;
};

// ---- Load ----
bool loadImage(const std::string &filename, Image &img) {
    unsigned char *pixels = stbi_load(
        filename.c_str(),
        &img.width,
        &img.height,
        &img.channels,
        3 // принудительно RGB
    );

    if (!pixels) {
        return false;
    }

    img.channels = 3;
    img.data.assign(pixels, pixels + img.width * img.height * 3);

    stbi_image_free(pixels);
    return true;
}

// ---- Save ----
bool saveImage(const std::string &filename, const Image &img) {
    auto pos = filename.find_last_of('.');
    if (pos == std::string::npos) return false;

    std::string ext = filename.substr(pos + 1);
    for (auto &c : ext) c = std::tolower(c);

    if (ext == "png")
        return stbi_write_png(
            filename.c_str(),
            img.width,
            img.height,
            3,
            img.data.data(),
            img.width * 3
        );

    if (ext == "bmp")
        return stbi_write_bmp(
            filename.c_str(),
            img.width,
            img.height,
            3,
            img.data.data()
        );

    if (ext == "jpg" || ext == "jpeg")
        return stbi_write_jpg(
            filename.c_str(),
            img.width,
            img.height,
            3,
            img.data.data(),
            95
        );

    if (ext == "tga")
        return stbi_write_tga(
            filename.c_str(),
            img.width,
            img.height,
            3,
            img.data.data()
        );

    std::cerr << "Unsupported output format: " << ext << "\n";
    return false;
}

// ---- Вспомогательная функция: обрезка к [0;255] ----
static inline unsigned char clampByte(float v) {
    if (v < 0.0f)   v = 0.0f;
    if (v > 255.0f) v = 255.0f;
    return static_cast<unsigned char>(v);
}

// ---- Обработка куска данных: изменение контраста ----
void contrast_chunk(std::vector<unsigned char> &data,
                    size_t start, size_t end,
                    float alpha)
{
    for (size_t i = start; i < end; ++i) {
        float v = static_cast<float>(data[i]);
        // смещаем к центру 128, растягиваем/сжимаем, возвращаем назад
        v = (v - 128.0f) * alpha + 128.0f;
        data[i] = clampByte(v);
    }
}

// ---- Многопоточная регулировка контраста ----
void adjust_contrast(Image &img, int contrast) {
    // contrast в [-100; 100]
    float alpha = 1.0f + static_cast<float>(contrast) / 100.0f;

    // если по факту контраст не меняется — выходим
    if (std::fabs(alpha - 1.0f) < 1e-6f) {
        return;
    }

    int totalThreads = std::thread::hardware_concurrency();
    if (totalThreads < 3) totalThreads = 3;

    int threadCount = totalThreads - 2;
    if (threadCount < 1) threadCount = 1;

    size_t total = img.data.size();
    size_t chunk = total / threadCount;

    std::vector<std::thread> pool;
    pool.reserve(threadCount);

    for (int t = 0; t < threadCount; ++t) {
        size_t start = static_cast<size_t>(t) * chunk;
        size_t end   = (t == threadCount - 1) ? total : start + chunk;

        pool.emplace_back(
            contrast_chunk,
            std::ref(img.data),
            start,
            end,
            alpha
        );
    }

    for (auto &th : pool) {
        th.join();
    }
}

// ---- Разбор аргументов ----
bool parseArgs(int argc, char **argv,
               std::string &input,
               std::string &output,
               int &contrast)
{
    contrast = 0; // по умолчанию без изменения

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-i") && i + 1 < argc) {
            input = argv[i + 1];
        }
        if (!std::strcmp(argv[i], "-o") && i + 1 < argc) {
            output = argv[i + 1];
        }
        if (!std::strcmp(argv[i], "-c") && i + 1 < argc) {
            contrast = std::atoi(argv[i + 1]);
        }
    }

    if (contrast < -100) contrast = -100;
    if (contrast >  100) contrast =  100;

    return !input.empty() && !output.empty();
}

// ---- Main ----
int main(int argc, char **argv) {
    std::string in, out;
    int contrast = 0;

    if (!parseArgs(argc, argv, in, out, contrast)) {
        std::cout << "Usage: contrast -i <input> -o <output> -c <value>\n";
        std::cout << "  contrast: [-100; 100], положительное — усилить контраст\n";
        return 1;
    }

    Image img;
    if (!loadImage(in, img)) {
        std::cerr << "Unable to load image: " << in << "\n";
        return 1;
    }

    adjust_contrast(img, contrast);

    if (!saveImage(out, img)) {
        std::cerr << "Unable to save image: " << out << "\n";
        return 1;
    }

    std::cout << "Done! Saved to " << out << "\n";
    return 0;
}

