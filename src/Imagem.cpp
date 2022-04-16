#include "imagem.hpp"

Imagem::Imagem(vector<vector<vector<int>>> matriz){
    this->matriz = matriz;
    this->altura = matriz.size();
    this->largura = matriz[0].size();
}

int Imagem::get_altura(){
    return this->altura;
}

int Imagem::get_largura(){
    return this->largura;
}

vector<vector<vector<int>>> Imagem::get_matriz(){
    return this->matriz;
}

void Imagem::set_altura(int altura){
    this->altura = altura;
}

void Imagem::set_largura(int largura){
    this->largura = largura;
}

void Imagem::set_matriz(vector<vector<vector<int>>> matriz){
    this->matriz = matriz;
} 

void Imagem::salvar_imagem(const char* arquivo){
    ofstream myFile(arquivo);

    myFile << "P3\n" << this->largura << ' ' << this->altura << "\n255\n";

    for (int i = 0; i < this->altura; i++) {
        for (int j = 0; j < this->largura; j++) {
            myFile << this->matriz[i][j][0] << ' ' << this->matriz[i][j][1] << ' ' << this->matriz[i][j][2] << ' ';
        }
        myFile << '\n';
    }

    myFile.close();
}