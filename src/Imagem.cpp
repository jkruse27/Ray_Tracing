#include "imagem.hpp"

Imagem::Imagem(vector<vector<color>> matriz){
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

vector<vector<color>> Imagem::get_matriz(){
    return this->matriz;
}

void Imagem::set_altura(int altura){
    this->altura = altura;
}

void Imagem::set_largura(int largura){
    this->largura = largura;
}

void Imagem::set_matriz(vector<vector<color>> matriz){
    this->matriz = matriz;
} 

void Imagem::salvar_imagem(const char* arquivo){
    ofstream myFile(arquivo);

    myFile << "P3\n" << this->largura << ' ' << this->altura << "\n255\n";

    for (int i = 0; i < this->altura; i++) {
        for (int j = 0; j < this->largura; j++) {
            myFile << this->matriz[this->altura-i-1][j].color_text() << ' ';
        }
        myFile << '\n';
    }

    myFile.close();
}