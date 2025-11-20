// g++ rotate_any.cpp -O2 -pthread -o rotate_any
// ./rotate_any -i input.jpg -o out.png -r 37.5

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

// ---- Load image ----
bool loadImage(const std::string &filename, Image &img) {
    unsigned char *pixels = stbi_load(filename.c_str(),
                                      &img.width,
                                      &img.height,
                                      &img.channels,
                                      3);

    if (!pixels) return false;

    img.channels = 3;
    img.data.assign(pixels, pixels + img.width * img.height * 3);
    stbi_image_free(pixels);
    return true;
}

// ---- Save image ----
bool saveImage(const std::string &filename, const Image &img) {
    std::string ext;
    size_t dot = filename.find_last_of('.');
    if (dot == std::string::npos) return false;

    ext = filename.substr(dot + 1);
    for (auto &c : ext) c = std::tolower(c);

    if (ext == "png") {
        return stbi_write_png(filename.c_str(),
                              img.width, img.height, 3,
                              img.data.data(), img.width * 3);
    }
    if (ext == "bmp") return stbi_write_bmp(filename.c_str(),
                                            img.width, img.height, 3,
                                            img.data.data());
    if (ext == "jpg" || ext == "jpeg")
        return stbi_write_jpg(filename.c_str(),
                              img.width, img.height, 3,
                              img.data.data(), 95);
    if (ext == "tga")
        return stbi_write_tga(filename.c_str(),
                              img.width, img.height, 3,
                              img.data.data());

    std::cerr << "Unsupported format: " << ext << "\n";
    return false;
}

// ---- Bilinear Interpolation ----
void getPixelBilinear(const Image& img, float x, float y, unsigned char* out) {
    if (x < 0 || y < 0 || x >= img.width - 1 || y >= img.height - 1) {
        out[0] = out[1] = out[2] = 0;
        return;
    }

    int x0 = int(x);
    int y0 = int(y);
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

    for (int c = 0; c < 3; c++) {
        float a = p00[c] * (1 - dx) + p10[c] * dx;
        float b = p01[c] * (1 - dx) + p11[c] * dx;
        out[c] = (unsigned char)(a * (1 - dy) + b * dy);
    }
}

// ---- Worker thread rotate ----
void rotateChunk(const Image &src, Image &dst,
                 float cosA, float sinA,
                 float cx, float cy,
                 float ncx, float ncy,
                 int yStart, int yEnd)
{
    for (int y = yStart; y < yEnd; y++) {
        for (int x = 0; x < dst.width; x++) {

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

// ---- Rotate arbitrary angle ----
void rotateAnyAngle(Image &img, float angleDeg) {
    float angle = angleDeg * M_PI / 180.0f;
    float cosA = std::cos(angle);
    float sinA = std::sin(angle);

    int w = img.width;
    int h = img.height;

    int nw = int(std::abs(w * cosA) + std::abs(h * sinA));
    int nh = int(std::abs(w * sinA) + std::abs(h * cosA));

    Image out;
    out.width = nw;
    out.height = nh;
    out.channels = 3;
    out.data.resize(nw * nh * 3);

    float cx = w / 2.0f;
    float cy = h / 2.0f;
    float ncx = nw / 2.0f;
    float ncy = nh / 2.0f;

    int threads = std::max(2u, std::thread::hardware_concurrency() - 2);
    int chunk = nh / threads;

    std::vector<std::thread> pool;

    for (int t = 0; t < threads; t++) {
        int ys = t * chunk;
        int ye = (t == threads - 1) ? nh : ys + chunk;

        pool.emplace_back(rotateChunk,
                          std::cref(img), std::ref(out),
                          cosA, sinA,
                          cx, cy,
                          ncx, ncy,
                          ys, ye);
    }

    for (auto &th : pool) th.join();

    img = out;
}

// ---- Args ----
bool parseArgs(int argc, char **argv,
               std::string &input,
               std::string &output,
               float &angle)
{
    angle = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i+1 < argc) input = argv[i+1];
        if (!strcmp(argv[i], "-o") && i+1 < argc) output = argv[i+1];
        if (!strcmp(argv[i], "-r") && i+1 < argc) angle = std::stof(argv[i+1]);
    }
    return !input.empty() && !output.empty();
}

// ---- Main ----
int main(int argc, char **argv) {
    std::string in, out;
    float angle;

    if (!parseArgs(argc, argv, in, out, angle)) {
        std::cout << "Usage:\n"
                  << "  ./rotate_any -i <input> -o <output> -r <angle>\n";
        return 1;
    }

    Image img;
    if (!loadImage(in, img)) {
        std::cerr << "Cannot load: " << in << "\n";
        return 1;
    }

    rotateAnyAngle(img, angle);

    if (!saveImage(out, img)) {
        std::cerr << "Cannot save: " << out << "\n";
        return 1;
    }

    std::cout << "Done! Saved to: " << out << "\n";
    return 0;
}
