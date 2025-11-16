// g++ rotate.cpp -O2 -pthread -o rotate
// ./rotate -i input.jpg -o out.png -r 90

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <algorithm>

struct Image {
    int width, height, channels;
    std::vector<unsigned char> data;
};

// ---- Load image ----
bool loadImage(const std::string &filename, Image &img) {
    unsigned char *pixels = stbi_load(filename.c_str(),
                                      &img.width,
                                      &img.height,
                                      &img.channels,
                                      3); // force RGB

    if (!pixels) return false;

    img.channels = 3;
    img.data.assign(pixels, pixels + img.width * img.height * 3);
    stbi_image_free(pixels);
    return true;
}

// ---- Save image by extension ----
bool saveImage(const std::string &filename, const Image &img) {
    auto extPos = filename.find_last_of('.');
    if (extPos == std::string::npos) return false;

    std::string ext = filename.substr(extPos + 1);
    for (auto &c : ext) c = std::tolower(c);

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


// ---- Rotate 180° (easy, same size) ----
void rotate180_chunk(const std::vector<unsigned char> &src,
                     std::vector<unsigned char> &dst,
                     size_t start, size_t end)
{
    size_t pixelCount = src.size() / 3;
    for (size_t i = start; i < end; i++) {
        size_t rev = pixelCount - 1 - i;
        dst[rev * 3]     = src[i * 3];
        dst[rev * 3 + 1] = src[i * 3 + 1];
        dst[rev * 3 + 2] = src[i * 3 + 2];
    }
}

// ---- Rotate 90° & 270° (requires XY transform) ----
void rotate90_270_chunk(const Image &src,
                        Image &dst,
                        size_t startRow,
                        size_t endRow,
                        bool clockwise)
{
    int w = src.width;
    int h = src.height;

    for (size_t y = startRow; y < endRow; y++) {
        for (int x = 0; x < w; x++) {
            int nx, ny;

            if (clockwise) {
                nx = h - 1 - y;
                ny = x;
            } else { // 270° CCW
                nx = y;
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

// ---- Manager function ----
void rotateImage(Image &img, int angle) {
    int totalThreads = std::thread::hardware_concurrency();
    if (totalThreads < 3) totalThreads = 3;

    int threadCount = totalThreads - 2;

    Image out;
    if (angle == 0) return;

    if (angle == 180) {
        out.width = img.width;
        out.height = img.height;
        out.channels = 3;
        out.data.resize(img.data.size());

        size_t pixels = img.data.size() / 3;
        size_t chunk = pixels / threadCount;

        std::vector<std::thread> pool;
        for (int t = 0; t < threadCount; t++) {
            size_t start = t * chunk;
            size_t end = (t == threadCount - 1) ? pixels : start + chunk;

            pool.emplace_back(rotate180_chunk,
                              std::cref(img.data),
                              std::ref(out.data),
                              start, end);
        }
        for (auto &th : pool) th.join();

        img = out;
        return;
    }

    if (angle == 90 || angle == 270) {
        bool cw = (angle == 90);

        out.width = img.height;
        out.height = img.width;
        out.channels = 3;
        out.data.resize(img.data.size());

        size_t rows = img.height;
        size_t chunk = rows / threadCount;

        std::vector<std::thread> pool;
        for (int t = 0; t < threadCount; t++) {
            size_t start = t * chunk;
            size_t end = (t == threadCount - 1) ? rows : start + chunk;

            pool.emplace_back(rotate90_270_chunk,
                              std::cref(img),
                              std::ref(out),
                              start, end, cw);
        }
        for (auto &th : pool) th.join();

        img = out;
        return;
    }

    std::cerr << "Unsupported angle: " << angle << "\n";
}

// ---- Parse args ----
bool parseArgs(int argc, char **argv,
               std::string &input,
               std::string &output,
               int &angle)
{
    angle = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) input = argv[i + 1];
        if (!strcmp(argv[i], "-o") && i + 1 < argc) output = argv[i + 1];
        if (!strcmp(argv[i], "-r") && i + 1 < argc) angle = std::stoi(argv[i + 1]);
    }

    return !input.empty() && !output.empty();
}

// ---- Main ----
int main(int argc, char **argv) {
    std::string in, out;
    int angle = 0;

    if (!parseArgs(argc, argv, in, out, angle)) {
        std::cout << "Usage: program -i <input> -o <output> -r <0|90|180|270>\n";
        return 1;
    }

    if (!(angle == 0 || angle == 90 || angle == 180 || angle == 270)) {
        std::cerr << "Error: angle must be 0, 90, 180 or 270\n";
        return 1;
    }

    Image img;
    if (!loadImage(in, img)) {
        std::cerr << "Unable to load: " << in << "\n";
        return 1;
    }

    rotateImage(img, angle);

    if (!saveImage(out, img)) {
        std::cerr << "Unable to save: " << out << "\n";
        return 1;
    }

    std::cout << "Done! Saved to " << out << "\n";
    return 0;
}
