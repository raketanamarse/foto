// g++ blur.cpp -O2 -pthread -o blur
// ./blur -i input.jpg -o out.png -b 5

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <cstring>

struct Image {
    int width, height, channels;
    std::vector<unsigned char> data;
};

// ---- Load ----
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

// ---- Save ----
bool saveImage(const std::string &filename, const Image &img) {
    auto pos = filename.find_last_of('.');
    if (pos == std::string::npos) return false;

    std::string ext = filename.substr(pos + 1);
    for (auto &c : ext) c = std::tolower(c);

    if (ext == "png")
        return stbi_write_png(filename.c_str(), img.width, img.height, 3, img.data.data(), img.width * 3);
    if (ext == "bmp")
        return stbi_write_bmp(filename.c_str(), img.width, img.height, 3, img.data.data());
    if (ext == "jpg" || ext == "jpeg")
        return stbi_write_jpg(filename.c_str(), img.width, img.height, 3, img.data.data(), 95);
    if (ext == "tga")
        return stbi_write_tga(filename.c_str(), img.width, img.height, 3, img.data.data());

    std::cerr << "Unsupported output format: " << ext << "\n";
    return false;
}

// ---- Build Gaussian kernel ----
std::vector<float> build_kernel(int radius) {
    int size = radius * 2 + 1;
    std::vector<float> kernel(size);
    float sigma = radius / 2.0f;
    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        int x = i - radius;
        float v = std::exp(-(x*x)/(2*sigma*sigma));
        kernel[i] = v;
        sum += v;
    }

    for (float &v : kernel) v /= sum;

    return kernel;
}

// ---- Gaussian blur chunk ----
void blur_chunk(const Image &src, Image &dst,
                const std::vector<float> &kernel,
                int radius, int yStart, int yEnd)
{
    int w = src.width;
    int h = src.height;

    for (int y = yStart; y < yEnd; y++) {
        for (int x = 0; x < w; x++) {
            float r = 0, g = 0, b = 0;

            // Convolution
            for (int k = -radius; k <= radius; k++) {
                int xx = std::min(std::max(x + k, 0), w - 1);
                int idx = (y * w + xx) * 3;

                float kv = kernel[k + radius];
                r += src.data[idx] * kv;
                g += src.data[idx + 1] * kv;
                b += src.data[idx + 2] * kv;
            }

            int outIdx = (y * w + x) * 3;
            dst.data[outIdx] = (unsigned char)r;
            dst.data[outIdx + 1] = (unsigned char)g;
            dst.data[outIdx + 2] = (unsigned char)b;
        }
    }
}

// ---- Multi-threaded Gaussian ----
void gaussian_blur(Image &img, int radius) {
    if (radius < 1) return;

    Image temp = img;
    std::vector<float> kernel = build_kernel(radius);

    int totalThreads = std::thread::hardware_concurrency();
    if (totalThreads < 3) totalThreads = 3;

    int threads = totalThreads - 2;

    std::vector<std::thread> pool;
    pool.reserve(threads);

    int rowsPerThread = img.height / threads;

    for (int t = 0; t < threads; t++) {
        int yStart = t * rowsPerThread;
        int yEnd = (t == threads - 1) ? img.height : yStart + rowsPerThread;

        pool.emplace_back(blur_chunk,
                          std::cref(temp),
                          std::ref(img),
                          std::cref(kernel),
                          radius,
                          yStart, yEnd);
    }

    for (auto &th : pool) th.join();
}

// ---- Args ----
bool parseArgs(int argc, char **argv, std::string &input, std::string &output, int &blurRadius) {
    blurRadius = 3; // default

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) input = argv[i + 1];
        if (!strcmp(argv[i], "-o") && i + 1 < argc) output = argv[i + 1];
        if (!strcmp(argv[i], "-b") && i + 1 < argc) blurRadius = std::atoi(argv[i + 1]);
    }
    return !input.empty() && !output.empty();
}

// ---- Main ----
int main(int argc, char **argv) {
    std::string in, out;
    int radius = 3;

    if (!parseArgs(argc, argv, in, out, radius)) {
        std::cout << "Usage: program -i <input> -o <output> -b <radius>\n";
        return 1;
    }

    Image img;
    if (!loadImage(in, img)) {
        std::cerr << "Unable to load image: " << in << "\n";
        return 1;
    }

    gaussian_blur(img, radius);

    if (!saveImage(out, img)) {
        std::cerr << "Unable to save image: " << out << "\n";
        return 1;
    }

    std::cout << "Done! Saved to " << out << "\n";
    return 0;
}
