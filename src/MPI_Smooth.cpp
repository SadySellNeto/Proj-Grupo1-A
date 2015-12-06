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
 * MPI_Smooth.cpp:
 * Um dos arquivos da biblioteca para manipulacao de imagens .pgm e .ppm
 * e para aplicacao do estencil smooth.
 */

// Impede que o arquivo seja incluido mais de uma vez.
// (Equivalente a #ifndef ... #define ... #endif, porem mais poderoso).
#pragma once

#include <string> // string
#include <vector> // vector
#include <tuple> // tuple
#include <mpi.h> // MPI_Init, MPI_Comm_rank, MPI_Comm_size, MPI_Scatter,
 // MPI_Gather, MPI_Barrier, MPI_Finalize

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
	
	/* Classe MPI_Smooth:
	 * Classe que aplica o estencil smooth usando OpenMP e OpenMPI.
	 * Deriva de Netpbm_Smooth e implementa seus metodos abstratos.
	 */
	class MPI_Smooth :
		public Netpbm_Smooth {
		
	private:
		
		/* Campos:
		 * rank: identificador do processo.
		 */
		int rank;
		
		/* Metodo __smooth:
		 * Realiza, efetivamente, a suavizacao da imagem, de maneira
		 *   paralela, usando OpenMP.
		 * Implementa o metodo abstrato de mesmo nome na classe Netpbm_Smooth.
		 * Esse metodo nao possui parametros.
		 * Retorno:
		 *   ponteiro para imagem suavizada.
		 */
		inline MPI_Smooth* __smooth() {
			
			// Cria uma nova imagem com os mesmos parâmetros da original;
			MPI_Smooth* smoothed = new MPI_Smooth(this->width, this->height, this->max_value, this->dimension);
			
			// Calcula o número de pixels a serem calculados pelo algoritmo;
			dword size = this->height * this->width;
			
			// Diretiva omp parallel for, já que este FOR pode ser executado
			// em pararelo, possuindo as seguintes cáusulas:
			// default(none): por padrao declara como erro o uso de variaveis
			//   externas;
			// shared: variaveis ja declaradas que serao compartilhadas por
			//   todas as threads;
			#pragma omp parallel for default(none) shared(smoothed, size)
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
				for (long int x = row - 2; x <= row + 2; x++) {
					for (long int y = col - 2; y <= col + 2; y++) {
						
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
		
		/* Metodo __smooth:
		 * Realiza, efetivamente, a suavizacao da imagem, de maneira
		 *   paralela, usando OpenMP.
		 * Implementa o metdo de mesmo nome da classe Netpbm_Smooth.
		 * Parmetros:
		 *   first: primeira linha de interesse a ser calculada (inclusive).
		 *   last: primeira linha fora do interesse (exclusive).
		 * Retorno:
		 *   ponteiro para imagem suavizada.
		 */
		inline MPI_Smooth* __smooth(dword first, dword last) {
			
			// A posicao inicial pode ser maior que a final quando nao
			// houver area para suavizar (supoe-se que ele sera igual).
			// Ness caso, nao ha imagem para calcular.
			if (first >= last) {
				// Portanto, retorna um ponteiro nulo.
				return nullptr;
			}
			
			// Cria uma nova imagem com os mesmos parâmetros da original;
			MPI_Smooth* smoothed = new MPI_Smooth(this->width, last - first, this->max_value, this->dimension);
			
			// Calcula o número de pixels a serem calculados pelo algoritmo;
			dword size = this->height * this->width;
			
			// Calcula as posicoes de interesse;
			dword first_pos = this->map_position_2d(first, 0);
			dword last_pos = this->map_position_2d(last, 0);
			
			// Diretiva omp parallel for, já que este FOR pode ser executado
			// em pararelo, possuindo as seguintes cáusulas:
			// default(none): por padrao declara como erro o uso de variaveis
			//   externas;
			// shared: variaveis ja declaradas que serao compartilhadas por
			//   todas as threads;
			#pragma omp parallel for default(none) shared(smoothed, size, first, first_pos, last_pos)
			for (dword i = first_pos; i < last_pos; i++) {
				
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
					smoothed->data[this->map_position_3d(row, col, k) - first * this->width * this->dimension] = sums[k] / count;
				}
				
			}
			
			// Retorna a imagem resultante.
			return smoothed;
			
		}
		
	protected:
		
		/* Metodo _smooth:
		 * Atua como gerente da comunicacao necessaria pelo modelo de
		 *   passagem de mensagem (OpenMPI), dividindo as areas das imagens
		 *   para cada no, e empregando a realizacao da tarefa em cada no.
		 * Implementa o metdo de mesmo nome da classe Netpbm_Smooth.
		 * Esse metodo nao possui parametros.
		 * Retorno:
		 *   ponteiro para imagem suavizada.
		 */
		Netpbm_Smooth* _smooth() {
			
			
			int processors; // Numero de processos usados pelo OpenMPI;
			// Numero de linhas da imagem a serem processadas por cada no;
			word payload;
			
			// Cria uma imegm que aramazenara o resultado da aplicacao do estencil,
			// com os mesmos parametros da original;
			MPI_Smooth* smoothed = new MPI_Smooth(this->width, this->height, this->max_value, this->dimension);
			
			// Inicializa o ambiente OpenMPI;
			MPI_Init(nullptr, nullptr);
			
			// Identificador do processo;
			int rank;
			
			// Obtem o identificar do processo atual;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			// Obtem o numero toral de processos;
			MPI_Comm_size(MPI_COMM_WORLD, &processors);
			// Calcula a carga de cada processo;
			payload = this->height / processors;
			
			// Cria uma imagem escrava para o processo atual realizar sua
			// parte do algoritmo;
			MPI_Smooth* slave = new MPI_Smooth(this->width, payload, this->max_value, this->dimension);
			
			// Distribui os dados da imagem entre os processos;
			MPI_Scatter(this->data.data(), payload * this->width * this->dimension, MPI_UNSIGNED_CHAR, slave->data.data(), payload * this->width * this->dimension, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
			
			// Aplica o smooth para o escravo;
			MPI_Smooth* smoothed_slave = slave->__smooth();
			
			// O escravo original nao eh mas necessario. Deleta-o.
			delete slave;
			
			// Calcula o smooth no restante (que nao foi enviado via scatter)
			// da imagem;
			if (this->rank == 0) {
				
				// Aplica o smooth para essa parte "restante";
				MPI_Smooth* other_slave = this->__smooth(payload * processors, this->height);
				
				// Se algum smooth foi efetivamente feito,
				if (other_slave != nullptr) {
					
					// Calcula a primeira e a ultima posicao da imagem 
					// global que esse "resto" calculou.
					word first_pos = map_position_3d(payload * processors, 0, 0);
					word last_pos = map_position_3d(this->height, 0, 0); 
					
					// Copia esse resultado "restante" para o resultado
					// global.
					for (long int i = first_pos; i < last_pos; i++) {
						smoothed->data[i] = other_slave->data[i - first_pos];
					}
					
					// Esse "restante" nao eh mas necessario. Deleta-o.
					delete other_slave;
					
				}
				
			}
			
			// Reune os resultados dos processos;
			MPI_Gather(smoothed_slave->data.data(), payload * this->width * this->dimension, MPI_UNSIGNED_CHAR, smoothed->data.data(), payload * this->width * this->dimension, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
			
			// Os resultados individuais de cada escravo nao sao mais
			// necessarios. Deleta-os.
			delete smoothed_slave;
			
			this->rank = rank;
			
			// Finaliza o ambiente MPI;
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			
			// Retorna a imagem gerada.
			return smoothed;
			
		}
		
		/* Metodo get_control:
		 * Recupera uma funcao de controle, que sera usada para permitir/
		 *   negar operacoes de saida pelo programa.
		 * Implementa o metodo abstrato de mesmo nome na classe Netpbm_Smooth.
		 * Este metodo nao possui parametros.
		 * Retorno:
		 *   funcao a ser aplicada; nesse caso, eh uma funcao que retorna
		 *   verdadeiro apenas para o processo mestre (rank 0), concedendo
		 *   apenas a ele o direito de realizar operacoes de saida.
		 */
		virtual control_function get_control() const {
			return [](void* arg) {
				MPI_Smooth* _arg = (MPI_Smooth*) arg;
				return _arg->rank == 0;
			};
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
		MPI_Smooth(dword width, dword height, byte max_value, byte dimension)
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
		MPI_Smooth(const string& file_name) 
			: Netpbm_Smooth(file_name) {
		}
		
		/* Metodo parallelism_type:
		 * Recupera uma string de acordo com o tipo de paralelismo
		 *   empregado, para ser usada como sufixo em nomes de arquivo.
		 * Implementa o metodo abstrato de mesmo nome na classe Netpbm_Smooth.
		 * Este metodo nao possui parametros.
		 * Retorno:
		 *   string correspondente ao tipo de paralelismo;
		 *   para o smooth OpenMP + OpenMPI, essa string eh "mpi".
		 */
		string parallelism_type() const {
			return "mpi";
		}
		
	};
	
}
