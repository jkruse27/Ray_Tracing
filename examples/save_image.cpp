#include "Imagem.hpp"
#include <vector>

int main() {

    // Image

    const int image_width = 256;
    const int image_height = 256;

    vector<vector<vector<int>>> matrix(image_height, vector<vector<int>>(image_width, vector<int>(3, 0)));

    for (int j = image_height-1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            auto r = double(i) / (image_width-1);
            auto g = double(j) / (image_height-1);
            auto b = 0.25;

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            matrix[i][j][0] = ir;
            matrix[i][j][1] = ig;
            matrix[i][j][2] = ib;
        }
    }

    Imagem image = Imagem(matrix);
    image.salvar_imagem("exemplo.ppm");
}