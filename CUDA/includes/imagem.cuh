#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.cuh"
#include "color.cuh"
#include "cuda_parameters.cuh"

using namespace std;

/*! \brief Classe Imagem
*
* Classe responsável por representar uma imagem, guardando seus valores,
* dimensões e possibilitando abrir e salvar arquivos.
*
* */

class Imagem{
	private:
		int altura; /*!< Altura da imagem (número de linhas)*/
        int largura; /*!< Largura da imagem (número de colunas)*/
		color* matriz; /*!< Matriz contendo a imagem*/
	public:
		/*! \brief Construtor da Imagem
		*
		* Recebe uma matriz correspondendo a imagem
		*
		* \param color** matriz: Matriz representando a imagem
		* \return Nada (este é um construtor!)
		* */
        Imagem(int alt, int larg)
			: altura(alt), largura(larg)
		{};

		/*! \brief Construtor da Imagem
		*
		* Recebe uma matriz correspondendo a imagem
		*
		* \param color** matriz: Matriz representando a imagem
		* \return Nada (este é um construtor!)
		* */
        Imagem(color* matriz, int height, int width);

		/*! \brief Construtor da Imagem
		*
		* Instancia um objeto imagem vazio
		*
		* \return Nada (este é um construtor!)
		* */
        Imagem(){};

        int get_altura();
		int get_largura();
        color* get_matriz();
        void set_altura(int altura);
		void set_largura(int largura);
        void set_matriz(color* matriz);        

		/*! \brief Salva a imagem no caminho fornecido
		*
		* Salva a imagem usando o formato PPM no caminho determinado.
		*
		* \param const char* arquivo: Nome do arquivo para salvar a imagem
		* \return Nada
		* */
		void salvar_imagem(const char* arquivo);

		~Imagem();
}; 








__host__ inline Imagem::Imagem(color* matriz, int height, int width){
    this->matriz = matriz;
    this->altura = height;
    this->largura = width;
}

__host__ inline Imagem::~Imagem(){
    checkCudaErrors(cudaFree(this->matriz));
}

__host__ inline int Imagem::get_altura(){
    return this->altura;
}

__host__ inline int Imagem::get_largura(){
    return this->largura;
}

__host__ inline color* Imagem::get_matriz(){
    return this->matriz;
}

__host__ inline void Imagem::set_altura(int altura){
    this->altura = altura;
}

__host__ inline void Imagem::set_largura(int largura){
    this->largura = largura;
}

__host__ inline void Imagem::set_matriz(color* matriz){
    this->matriz = matriz;
} 

__host__ inline void Imagem::salvar_imagem(const char* arquivo){
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