#pragma once
#include <vector>
#include <iostream>
#include <fstream>

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
		vector<vector<vector<int>>> matriz; /*!< Matriz contendo a imagem*/
	public:
		/*! \brief Construtor da Imagem
		*
		* Recebe uma matriz correspondendo a imagem
		*
		* \param vector<vector<vector<int>>> matriz Matriz representando a imagem
		* \return Nada (este é um construtor!)
		* */
        Imagem(vector<vector<vector<int>>> matriz);

		/*! \brief Construtor da Imagem
		*
		* Instancia um objeto imagem vazio
		*
		* \return Nada (este é um construtor!)
		* */
        Imagem(){};

        int get_altura();
		int get_largura();
        vector<vector<vector<int>>> get_matriz();
        void set_altura(int altura);
		void set_largura(int largura);
        void set_matriz(vector<vector<vector<int>>> matriz);        

		/*! \brief Salva a imagem no caminho fornecido
		*
		* Salva a imagem usando o formato PPM no caminho determinado.
		*
		* \param const char* arquivo Nome do arquivo para salvar a imagem
		* \return Nada
		* */
		void salvar_imagem(const char* arquivo);
}; 