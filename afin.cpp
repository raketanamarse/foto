#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cctype>
#include <cstdlib>
#include <utility>
#include <string>
#include <functional>

const double MY_PI = 3.14159265358979323846;

struct Image {
    int width, height, channels;
    std::vector<unsigned char> data;
};

// Матрица аффинного преобразования 2x3
struct AffineMatrix {
    double m[2][3]; // [a, b, c]
    // [d, e, f]

    AffineMatrix() {
        // Единичная матрица по умолчанию
        m[0][0] = 1; m[0][1] = 0; m[0][2] = 0;
        m[1][0] = 0; m[1][1] = 1; m[1][2] = 0;
    }
};

// ---- Load image (any format) ----
bool loadImage(const std::string& filename, Image& img) {
    unsigned char* pixels = stbi_load(filename.c_str(),
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
bool saveImage(const std::string& filename, const Image& img) {
    auto extPos = filename.find_last_of('.');
    if (extPos == std::string::npos) return false;

    std::string ext = filename.substr(extPos + 1);
    for (auto& c : ext) c = std::tolower(static_cast<unsigned char>(c));

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

// ---- Создание матрицы поворота ----
AffineMatrix createRotationMatrix(double angle) {
    AffineMatrix mat;
    double rad = angle * MY_PI / 180.0;
    double cosA = std::cos(rad);
    double sinA = std::sin(rad);

    mat.m[0][0] = cosA;  mat.m[0][1] = -sinA; mat.m[0][2] = 0;
    mat.m[1][0] = sinA;  mat.m[1][1] = cosA;  mat.m[1][2] = 0;

    return mat;
}

// ---- Создание матрицы масштабирования ----
AffineMatrix createScaleMatrix(double scaleX, double scaleY) {
    AffineMatrix mat;
    mat.m[0][0] = scaleX; mat.m[0][1] = 0;      mat.m[0][2] = 0;
    mat.m[1][0] = 0;      mat.m[1][1] = scaleY; mat.m[1][2] = 0;

    return mat;
}

// ---- Создание матрицы сдвига (shear) ----
AffineMatrix createShearMatrix(double shearX, double shearY) {
    AffineMatrix mat;
    mat.m[0][0] = 1;      mat.m[0][1] = shearX; mat.m[0][2] = 0;
    mat.m[1][0] = shearY; mat.m[1][1] = 1;      mat.m[1][2] = 0;

    return mat;
}

// ---- Создание матрицы переноса ----
AffineMatrix createTranslationMatrix(double tx, double ty) {
    AffineMatrix mat;
    mat.m[0][0] = 1; mat.m[0][1] = 0; mat.m[0][2] = tx;
    mat.m[1][0] = 0; mat.m[1][1] = 1; mat.m[1][2] = ty;

    return mat;
}

// ---- Применение аффинного преобразования к одному пикселю ----
void applyAffineToPixel(const Image& src, Image& dst, int x, int y, const AffineMatrix& mat) {
    // Преобразование координат относительно центра
    double centerX = dst.width / 2.0;
    double centerY = dst.height / 2.0;

    double relX = x - centerX;
    double relY = y - centerY;

    // Применяем преобразование и переносим к центру исходного
    double srcX = mat.m[0][0] * relX + mat.m[0][1] * relY + mat.m[0][2] + src.width / 2.0;
    double srcY = mat.m[1][0] * relX + mat.m[1][1] * relY + mat.m[1][2] + src.height / 2.0;

    // Билинейная интерполяция
    int x0 = static_cast<int>(std::floor(srcX));
    int y0 = static_cast<int>(std::floor(srcY));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    double dx = srcX - x0;
    double dy = srcY - y0;

    // Проверка границ
    if (x0 < 0 || x1 >= src.width || y0 < 0 || y1 >= src.height) {
        // Если вышли за границы - черный пиксель
        for (int c = 0; c < 3; c++) {
            dst.data[(y * dst.width + x) * 3 + c] = 0;
        }
        return;
    }

    // Интерполяция для каждого канала
    for (int c = 0; c < 3; c++) {
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

// ---- Обработка части изображения аффинным преобразованием ----
void affine_chunk(const Image& src, Image& dst, const AffineMatrix& mat, int startY, int endY) {
    for (int y = startY; y < endY; y++) {
        if (y < 0 || y >= dst.height) continue;
        for (int x = 0; x < dst.width; x++) {
            applyAffineToPixel(src, dst, x, y, mat);
        }
    }
}

// ---- Многопоточное аффинное преобразование ----
void affineTransform(const Image& src,
                     Image& dst,
                     const AffineMatrix& mat,
                     int& newWidth,
                     int& newHeight) {
    // Если размеры не указаны (0,0), автоматически рассчитываем bounding box
    if (newWidth == 0 && newHeight == 0) {
        // Создаем углы исходного изображения
        std::vector<std::pair<double, double>> corners = {
            {0.0, 0.0},
            {static_cast<double>(src.width), 0.0},
            {static_cast<double>(src.width), static_cast<double>(src.height)},
            {0.0, static_cast<double>(src.height)}
        };

        double minX = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double minY = std::numeric_limits<double>::max();
        double maxY = std::numeric_limits<double>::lowest();

        for (const auto& corner : corners) {
            double x = corner.first;
            double y = corner.second;

            // Применяем аффинное преобразование
            double newX = mat.m[0][0] * x + mat.m[0][1] * y + mat.m[0][2];
            double newY = mat.m[1][0] * x + mat.m[1][1] * y + mat.m[1][2];

            minX = std::min(minX, newX);
            maxX = std::max(maxX, newX);
            minY = std::min(minY, newY);
            maxY = std::max(maxY, newY);
        }

        // Рассчитываем новые размеры с небольшим запасом
        newWidth = static_cast<int>(maxX - minX) + 10;
        newHeight = static_cast<int>(maxY - minY) + 10;

        if (newWidth < 1) newWidth = 1;
        if (newHeight < 1) newHeight = 1;

        // Корректируем матрицу для учета смещения
        AffineMatrix adjustedMat = mat;
        adjustedMat.m[0][2] -= minX - 5.0;  // добавляем небольшой отступ
        adjustedMat.m[1][2] -= minY - 5.0;

        // Используем скорректированную матрицу
        dst.width = newWidth;
        dst.height = newHeight;
        dst.channels = 3;
        dst.data.assign(newWidth * newHeight * 3, 0);

        std::cout << "Auto-calculated size: " << newWidth << "x" << newHeight << std::endl;

        unsigned int totalThreads = std::thread::hardware_concurrency();
        if (totalThreads == 0) totalThreads = 4;
        if (totalThreads < 3) totalThreads = 3;
        int threadCount = static_cast<int>(totalThreads) - 2;
        if (threadCount < 1) threadCount = 1;

        int rowsPerThread = (newHeight + threadCount - 1) / threadCount;

        std::vector<std::thread> pool;
        pool.reserve(threadCount);

        for (int t = 0; t < threadCount; t++) {
            int startY = t * rowsPerThread;
            int endY = std::min(startY + rowsPerThread, newHeight);
            if (startY >= endY) break;

            pool.emplace_back(
                affine_chunk,
                std::cref(src),
                std::ref(dst),
                std::cref(adjustedMat),
                startY,
                endY
            );
        }

        for (auto& th : pool) th.join();
    } else {
        // Используем указанные размеры
        if (newWidth < 1) newWidth = 1;
        if (newHeight < 1) newHeight = 1;

        dst.width = newWidth;
        dst.height = newHeight;
        dst.channels = 3;
        dst.data.assign(newWidth * newHeight * 3, 0);

        unsigned int totalThreads = std::thread::hardware_concurrency();
        if (totalThreads == 0) totalThreads = 4;
        if (totalThreads < 3) totalThreads = 3;
        int threadCount = static_cast<int>(totalThreads) - 2;
        if (threadCount < 1) threadCount = 1;

        int rowsPerThread = (newHeight + threadCount - 1) / threadCount;

        std::vector<std::thread> pool;
        pool.reserve(threadCount);

        for (int t = 0; t < threadCount; t++) {
            int startY = t * rowsPerThread;
            int endY = std::min(startY + rowsPerThread, newHeight);
            if (startY >= endY) break;

            pool.emplace_back(
                affine_chunk,
                std::cref(src),
                std::ref(dst),
                std::cref(mat),
                startY,
                endY
            );
        }

        for (auto& th : pool) th.join();
    }
}

// ---- Args ----
bool parseArgs(int argc, char** argv,
               std::string& input,
               std::string& output,
               std::string& transform,
               double& param1,
               double& param2) {
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-i") && i + 1 < argc) input = argv[i + 1];
        if (!std::strcmp(argv[i], "-o") && i + 1 < argc) output = argv[i + 1];
        if (!std::strcmp(argv[i], "-t") && i + 1 < argc) transform = argv[i + 1];
        if (!std::strcmp(argv[i], "-p1") && i + 1 < argc) param1 = std::atof(argv[i + 1]);
        if (!std::strcmp(argv[i], "-p2") && i + 1 < argc) param2 = std::atof(argv[i + 1]);
    }
    return !input.empty() && !output.empty() && !transform.empty();
}

int main(int argc, char** argv) {
    std::string in, out, transform;
    double param1 = 0.0, param2 = 0.0;

    if (!parseArgs(argc, argv, in, out, transform, param1, param2)) {
        std::cout << "Usage: affine -i <input> -o <output> -t <transform> [-p1 param1] [-p2 param2]\n";
        std::cout << "Transforms: rotate, scale, shear, translate\n";
        std::cout << "Examples:\n";
        std::cout << "  -t rotate -p1 45            (rotate 45 degrees)\n";
        std::cout << "  -t scale -p1 1.5 -p2 1.5    (scale 1.5x)\n";
        std::cout << "  -t shear -p1 0.3 -p2 0.2    (shear x=0.3, y=0.2)\n";
        std::cout << "  -t translate -p1 50 -p2 -30 (translate x=50, y=-30)\n";
        return 1;
    }

    Image src;
    if (!loadImage(in, src)) {
        std::cerr << "Unable to load image: " << in << "\n";
        return 1;
    }

    AffineMatrix mat;
    int newWidth = src.width;
    int newHeight = src.height;

    // Выбор преобразования
    if (transform == "rotate") {
        mat = createRotationMatrix(param1);

        // Правильный расчет размеров для поворота - bounding box
        double rad = param1 * MY_PI / 180.0;
        double cosA = std::cos(rad);
        double sinA = std::sin(rad);

        // Углы исходного изображения
        std::vector<std::pair<double, double>> corners = {
            {0.0, 0.0},
            {static_cast<double>(src.width), 0.0},
            {static_cast<double>(src.width), static_cast<double>(src.height)},
            {0.0, static_cast<double>(src.height)}
        };

        double minX = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double minY = std::numeric_limits<double>::max();
        double maxY = std::numeric_limits<double>::lowest();

        for (const auto& corner : corners) {
            double x = corner.first - src.width / 2.0;   // относительно центра
            double y = corner.second - src.height / 2.0;

            // Поворачиваем
            double newX = cosA * x - sinA * y;
            double newY = sinA * x + cosA * y;

            // Обратно к абсолютным координатам
            newX += src.width / 2.0;
            newY += src.height / 2.0;

            minX = std::min(minX, newX);
            maxX = std::max(maxX, newX);
            minY = std::min(minY, newY);
            maxY = std::max(maxY, newY);
        }

        newWidth = static_cast<int>(maxX - minX) + 2;   // +2 запас
        newHeight = static_cast<int>(maxY - minY) + 2;

        // Корректируем матрицу для центрирования
        mat.m[0][2] += (newWidth - src.width) / 2.0;
        mat.m[1][2] += (newHeight - src.height) / 2.0;

    } else if (transform == "scale") {
        mat = createScaleMatrix(param1, param2);
        newWidth = static_cast<int>(src.width * param1);
        newHeight = static_cast<int>(src.height * param2);

        if (newWidth < 1) newWidth = 1;
        if (newHeight < 1) newHeight = 1;

        // Центрируем масштабированное изображение
        mat.m[0][2] = (newWidth - src.width * param1) / 2.0;
        mat.m[1][2] = (newHeight - src.height * param2) / 2.0;

    } else if (transform == "shear") {
        mat = createShearMatrix(param1, param2);

        // Расчет размеров для shear
        std::vector<std::pair<double, double>> corners = {
            {0.0, 0.0},
            {static_cast<double>(src.width), 0.0},
            {static_cast<double>(src.width), static_cast<double>(src.height)},
            {0.0, static_cast<double>(src.height)}
        };

        double minX = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double minY = std::numeric_limits<double>::max();
        double maxY = std::numeric_limits<double>::lowest();

        for (const auto& corner : corners) {
            double x = corner.first;
            double y = corner.second;

            double newX = mat.m[0][0] * x + mat.m[0][1] * y + mat.m[0][2];
            double newY = mat.m[1][0] * x + mat.m[1][1] * y + mat.m[1][2];

            minX = std::min(minX, newX);
            maxX = std::max(maxX, newX);
            minY = std::min(minY, newY);
            maxY = std::max(maxY, newY);
        }

        newWidth = static_cast<int>(maxX - minX) + 2;
        newHeight = static_cast<int>(maxY - minY) + 2;

        if (newWidth < 1) newWidth = 1;
        if (newHeight < 1) newHeight = 1;

        // Корректируем матрицу для центрирования
        mat.m[0][2] -= minX - 1.0;
        mat.m[1][2] -= minY - 1.0;

    } else if (transform == "translate") {
        mat = createTranslationMatrix(param1, param2);

        // Для переноса увеличиваем размер чтобы вместить смещенное изображение
        newWidth = src.width + std::abs(static_cast<int>(param1)) + 10;
        newHeight = src.height + std::abs(static_cast<int>(param2)) + 10;

        // Корректируем смещение для центрирования
        mat.m[0][2] += (newWidth - src.width) / 2.0;
        mat.m[1][2] += (newHeight - src.height) / 2.0;

    } else {
        std::cerr << "Unknown transform: " << transform << "\n";
        return 1;
    }

    // Гарантируем минимальный размер
    if (newWidth < 10) newWidth = 10;
    if (newHeight < 10) newHeight = 10;

    std::cout << "Original size: " << src.width << "x" << src.height << "\n";
    std::cout << "New size: " << newWidth << "x" << newHeight << "\n";

    Image dst;
    affineTransform(src, dst, mat, newWidth, newHeight);

    if (!saveImage(out, dst)) {
        std::cerr << "Unable to save image: " << out << "\n";
        return 1;
    }

    std::cout << "Done! Saved to " << out << "\n";
    return 0;
}
