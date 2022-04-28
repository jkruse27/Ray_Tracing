#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include "vec3.hpp"
#include "color.hpp"

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
		vector<vector<color>> matriz; /*!< Matriz contendo a imagem*/
	public:
		/*! \brief Construtor da Imagem
		*
		* Recebe uma matriz correspondendo a imagem
		*
		* \param vector<vector<vector<color>>> matriz: Matriz representando a imagem
		* \return Nada (este é um construtor!)
		* */
        Imagem(int alt, int larg)
			: altura(alt), largura(larg)
		{};

		/*! \brief Construtor da Imagem
		*
		* Recebe uma matriz correspondendo a imagem
		*
		* \param vector<vector<vector<color>>> matriz: Matriz representando a imagem
		* \return Nada (este é um construtor!)
		* */
        Imagem(vector<vector<color>> matriz);

		/*! \brief Construtor da Imagem
		*
		* Instancia um objeto imagem vazio
		*
		* \return Nada (este é um construtor!)
		* */
        Imagem(){};

        int get_altura();
		int get_largura();
        vector<vector<color>> get_matriz();
        void set_altura(int altura);
		void set_largura(int largura);
        void set_matriz(vector<vector<color>> matriz);        

		/*! \brief Salva a imagem no caminho fornecido
		*
		* Salva a imagem usando o formato PPM no caminho determinado.
		*
		* \param const char* arquivo: Nome do arquivo para salvar a imagem
		* \return Nada
		* */
		void salvar_imagem(const char* arquivo);
}; 