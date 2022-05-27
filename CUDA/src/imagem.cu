#include "imagem.cuh"

__host__ Imagem::Imagem(color* matriz, int height, int width){
    this->matriz = matriz;
    this->altura = height;
    this->largura = width;
}

__host__ Imagem::~Imagem(){
    checkCudaErrors(cudaFree(this->matriz));
}

__host__ int Imagem::get_altura(){
    return this->altura;
}

__host__ int Imagem::get_largura(){
    return this->largura;
}

__host__ color* Imagem::get_matriz(){
    return this->matriz;
}

__host__ void Imagem::set_altura(int altura){
    this->altura = altura;
}

__host__ void Imagem::set_largura(int largura){
    this->largura = largura;
}

__host__ void Imagem::set_matriz(color* matriz){
    this->matriz = matriz;
} 

__host__ void Imagem::salvar_imagem(const char* arquivo){
    ofstream myFile(arquivo);

    myFile << "P3\n" << this->largura << ' ' << this->altura << "\n255\n";

    for (int i = 0; i < this->altura; i++) {
        for (int j = 0; j < this->largura; j++) {
            myFile << this->matriz[(this->altura-i-1)*this->altura+j].color_text() << ' ';
        }
        myFile << '\n';
    }

    myFile.close();
}