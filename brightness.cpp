// g++ brightness.cpp -O2 -pthread -o brightness
// ./brightness -i input.jpg -o out.png -b 30

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>

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
static inline unsigned char clampByte(int v) {
    if (v < 0)   v = 0;
    if (v > 255) v = 255;
    return static_cast<unsigned char>(v);
}

// ---- Обработка куска данных: изменение яркости ----
void brightness_chunk(std::vector<unsigned char> &data,
                      size_t start, size_t end,
                      int brightness)
{
    for (size_t i = start; i < end; ++i) {
        int v = static_cast<int>(data[i]) + brightness;
        data[i] = clampByte(v);
    }
}

// ---- Многопоточная регулировка яркости ----
void adjust_brightness(Image &img, int brightness) {
    if (brightness == 0) {
        return; // нет изменений
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
            brightness_chunk,
            std::ref(img.data),
            start,
            end,
            brightness
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
               int &brightness)
{
    brightness = 0; // по умолчанию без изменения

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-i") && i + 1 < argc) {
            input = argv[i + 1];
        }
        if (!std::strcmp(argv[i], "-o") && i + 1 < argc) {
            output = argv[i + 1];
        }
        if (!std::strcmp(argv[i], "-b") && i + 1 < argc) {
            brightness = std::atoi(argv[i + 1]);
        }
    }

    // Легкая нормализация (формально не обязательно, но безопасно)
    if (brightness < -255) brightness = -255;
    if (brightness >  255) brightness =  255;

    return !input.empty() && !output.empty();
}

// ---- Main ----
int main(int argc, char **argv) {
    std::string in, out;
    int brightness = 0;

    if (!parseArgs(argc, argv, in, out, brightness)) {
        std::cout << "Usage: brightness -i <input> -o <output> -b <value>\n";
        std::cout << "  brightness: [-255; 255], положительное — сделать светлее\n";
        return 1;
    }

    Image img;
    if (!loadImage(in, img)) {
        std::cerr << "Unable to load image: " << in << "\n";
        return 1;
    }

    adjust_brightness(img, brightness);

    if (!saveImage(out, img)) {
        std::cerr << "Unable to save image: " << out << "\n";
        return 1;
    }

    std::cout << "Done! Saved to " << out << "\n";
    return 0;
}
