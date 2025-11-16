//g++ negativ.cpp -O2 -pthread -o negativ
//./negativ -i input.jpg -o out.png


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

// ---- Load image (any format) ----
bool loadImage(const std::string &filename, Image &img) {
    unsigned char *pixels = stbi_load(filename.c_str(),
                                      &img.width,
                                      &img.height,
                                      &img.channels,
                                      3); // force to RGB

    if (!pixels) return false;

    img.channels = 3;
    img.data.assign(pixels, pixels + img.width * img.height * 3);

    stbi_image_free(pixels);
    return true;
}

// ---- Save image (auto format by extension) ----
bool saveImage(const std::string &filename, const Image &img) {
    auto extPos = filename.find_last_of('.');
    if (extPos == std::string::npos) return false;

    std::string ext = filename.substr(extPos + 1);
    for (auto &c : ext) c = std::tolower(c);

    if (ext == "png")
        return stbi_write_png(filename.c_str(),
                              img.width,
                              img.height,
                              3,
                              img.data.data(),
                              img.width * 3);

    if (ext == "bmp")
        return stbi_write_bmp(filename.c_str(),
                              img.width,
                              img.height,
                              3,
                              img.data.data());

    if (ext == "jpg" || ext == "jpeg")
        return stbi_write_jpg(filename.c_str(),
                               img.width,
                               img.height,
                               3,
                               img.data.data(),
                               95);

    if (ext == "tga")
        return stbi_write_tga(filename.c_str(),
                              img.width,
                              img.height,
                              3,
                              img.data.data());

    std::cerr << "Unsupported output format: " << ext << "\n";
    return false;
}

// ---- Single chunk invert ----
void invert_chunk(std::vector<unsigned char> &d, size_t start, size_t end) {
    for (size_t i = start; i < end; i++)
        d[i] = 255 - d[i];
}

// ---- Multithreaded inversion ----
void invert_multithread(Image &img) {
    int totalThreads = std::thread::hardware_concurrency();
    if (totalThreads < 3) totalThreads = 3;

    int threadCount = totalThreads - 2;

    size_t total = img.data.size();
    size_t chunk = total / threadCount;

    std::vector<std::thread> pool;
    pool.reserve(threadCount);

    for (int t = 0; t < threadCount; t++) {
        size_t start = t * chunk;
        size_t end = (t == threadCount - 1) ? total : start + chunk;

        pool.emplace_back(invert_chunk, std::ref(img.data), start, end);
    }

    for (auto &th : pool) th.join();
}

// ---- Args ----
bool parseArgs(int argc, char **argv, std::string &input, std::string &output) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) input = argv[i + 1];
        if (!strcmp(argv[i], "-o") && i + 1 < argc) output = argv[i + 1];
    }
    return !input.empty() && !output.empty();
}

// ---- Main ----
int main(int argc, char **argv) {
    std::string in, out;
    if (!parseArgs(argc, argv, in, out)) {
        std::cout << "Usage: program -i <input> -o <output>\n";
        return 1;
    }

    Image img;
    if (!loadImage(in, img)) {
        std::cerr << "Unable to load image: " << in << "\n";
        return 1;
    }

    invert_multithread(img);

    if (!saveImage(out, img)) {
        std::cerr << "Unable to save image: " << out << "\n";
        return 1;
    }

    std::cout << "Done! Saved to " << out << "\n";
    return 0;
}
