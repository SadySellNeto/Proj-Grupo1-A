/**
 * Universidade de Sao Paulo - USP
 * Instituto de Ciencias Matematicas e Computacao - ICMC
 * Departamento de Sistemas de Computacao - SSC
 * Programacao Concorrente - SSC0143
 * Professor Doutor Julio Cezar Estrella
 * 
 * Nomes / Nos. USP:
 *   Loys Henrique Sacomano Gibertoni - 8532377
 *   Sady Sell Neto - 8532418
 * Data: 6 de dezembro de 2015
 * 
 * Copyright (C) 2015 Loys Henrique Sacomano Gibertoni, Sady Sell Neto
 */

/**
 * Simple_Smooth.cpp:
 * Um dos arquivos da biblioteca para manipulacao de imagens .pgm e .ppm
 * e para aplicacao do estencil smooth.
 */

// Impede que o arquivo seja incluido mais de uma vez.
// (Equivalente a #ifndef ... #define ... #endif, porem mais poderoso).
#pragma once

#include <string> // string
#include <vector> // vector
#include <tuple> // tuple

#include "Netpbm_Smooth.cpp" // Netpbm_Smooth

// Usando namespace padrao, para evitar qualificao completa de nomes.
using namespace std;

/* Namespace ConcurrentProgramming:
 * Todas as funcionalidades deste trabalho estarao dentro deste namespace,
 * para melhorar a organizacao do codigo.
 * Definir namespaces eh particularmente interessante quando o objetivo
 * eh a criacao de bibliotecas.
 */
namespace ConcurrentProgramming {
	
	/* Classe Simple_Smooth:
	 * Classe que aplica o estencil smooth de maneira sequencial.
	 * Deriva de Netpbm_Smooth e implementa seus metodos abstratos.
	 */
	class Simple_Smooth :
		public Netpbm_Smooth {
		
	protected:
		
		/* Metodo get_control:
		 * Recupera uma funcao de controle, que sera usada para permitir/
		 *   negar operacoes de saida pelo programa.
		 * Implementa o metodo abstrato de mesmo nome na classe Netpbm_Smooth.
		 * Este metodo nao possui parametros.
		 * Retorno:
		 *   funcao a ser aplicada; nesse caso, como nenhum controle eh
		 *   necessario, retorna sempre verdadeiro.
		 */
		virtual control_function get_control() const {
			return [](void*) { return true; };
		}
		
		/* Metodo _smooth:
		 * Realiza, efetivamente, a suavizacao da imagem, de maneira
		 *   sequencial.
		 * Implementa o metodo abstrato de mesmo nome na classe Netpbm_Smooth.
		 * Esse metodo nao possui parametros.
		 * Retorno:
		 *   ponteiro para imagem suavizada.
		 */
		Netpbm_Smooth* _smooth() {
			
			// Cria uma nova imagem com os mesmos parâmetros da original;
			Simple_Smooth* smoothed = new Simple_Smooth(this->width, this->height, this->max_value, this->dimension);
			
			// Calcula o número de pixels a serem calculados pelo algoritmo;
			dword size = this->height * this->width;
			
			// Laco para percorrer todos os pixels da imagem;
			for (dword i = 0; i < size; i++) { 
				
				byte count = 0; // Contagem de pixels utilizados na media;
				 // Vetor das somas dos valores das cores de cada pixel;
				vector<word> sums(this->dimension, 0);
				
				// Transforma o i, que eh a posicao absoluta de um valor de
				// um pixel em uma tupla que contem a linha e a coluna do
				// pixel ao qual o valor pertence;
				tuple<dword, dword> pos = this->map_position_2d(i);
				long int row = get<0>(pos);
				long int col = get<1>(pos);
				
				// Lacos que percorrem a janela 5x5 cuja media sera utilizada
				// como valor do pixel;
				for (long int x = row - windows_radius; x <= row + windows_radius; x++) {
					for (long int y = col - windows_radius; y <= col + windows_radius; y++) {
						
						// Se o pixel atual da janela estiver dentro da area
						// da imagem;
						if (x >= 0 && x < this->height && y >= 0 && y < this->width) {
							
							// Eele deve ser considerado no calculo da media,
							// entao o contador eh incrementado;
							count++;
							
							// E o valor de cada cor do pixel eh somado a sua
							// respectiva soma;
							for (byte k = 0; k < this->dimension; k++) {
								sums[k] += this->map_access(x, y, k);
							}
							
						}
						
					}
				}
				
				// Para cada cor do pixel atual, escreve a media calculada da
				// cor correspondente;
				for (byte k = 0; k < this->dimension; k++) {
					smoothed->map_access(row, col, k) = sums[k] / count;
				}
				
			}
			
			// Retorna a imagem resultante.
			return smoothed;
			
		}
		
		/* Construtor (sobrecarregado):
		 * Constroi um novo objeto com dimensoes suficientes para comportar
		 *   uma certa demanda de dados, porem com vetor nao-inicializado.
		 *   Assim o faz delegando esta tarefa para o construtor da base.
		 * Parametros:
		 *   width: largura do objeto;
		 *   height: altura do objeto;
		 *   max_value: valor maximo de cor do objeto;
		 *   dimension: dimensao (numero de cores) do objeto;
		 */
		Simple_Smooth(dword width, dword height, byte max_value, byte dimension)
			: Netpbm_Smooth(width, height, max_value, dimension) {
		}
		
	public:
		
		/* Construtor (sobrecarregado):
		 * Constroi um novo objeto com seus dados obtidos a partir
		 *   de um arquivo.
		 *   Assim o faz delegando esta tarefa para o construtor da base.
		 * Parametros:
		 *   file_name: nome do arquivo cujos dados serao lidos.
		 */
		Simple_Smooth(const string& file_name) 
			: Netpbm_Smooth(file_name) {
		}
		
		/* Metodo parallelism_type:
		 * Recupera uma string de acordo com o tipo de paralelismo
		 *   empregado, para ser usada como sufixo em nomes de arquivo.
		 * Implementa o metodo abstrato de mesmo nome na classe Netpbm_Smooth.
		 * Este metodo nao possui parametros.
		 * Retorno:
		 *   string correspondente ao tipo de paralelismo;
		 *   para o smooth sequencial, essa string eh "seq".
		 */
		string parallelism_type() const {
			return "seq";
		}
		
	};
	
}
