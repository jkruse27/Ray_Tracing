#include <vector>
#include "imagem.hpp"
#include "vec3.hpp"

int main() {

    // Image

    const int image_width = 256;
    const int image_height = 256;

    vector<vector<color>> matrix(image_height, vector<color>(image_width));

    for (int j = image_height-1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            auto r = double(i) / (image_width-1);
            auto g = double(j) / (image_height-1);
            auto b = 0.25;

            matrix[i][j] = color(r, g, b);
        }
    }

    Imagem image = Imagem(matrix);
    image.salvar_imagem("exemplo.ppm");
}