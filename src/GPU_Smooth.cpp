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
 * GPU_Smooth.cpp:
 * Um dos arquivos da biblioteca para manipulacao de imagens .pgm e .ppm
 * e para aplicacao do estencil smooth.
 */

// Impede que o arquivo seja incluido mais de uma vez.
// (Equivalente a #ifndef ... #define ... #endif, porem mais poderoso).
#pragma once

#include <string> // string
#include <vector> // vector
#include <cmath> // ceil
#include <stdexcept> // runtime_error

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Netpbm_Smooth.cpp" // Netpbm_Smooth
#include "smooth.h"

// Macro para executar uma funcao CUDA:
// Executa, e encapsula o erro oocrrido em uma
// excecao.
#define cudaExec(expr) do { \
	const cudaError_t e = expr; \
	if (e != cudaError::cudaSuccess) { \
		throw std::runtime_error(cudaGetErrorString(e)); \
	} \
} while(false)

// Usando namespace padrao, para evitar qualificao completa de nomes.
using namespace std;

/* Namespace ConcurrentProgramming:
 * Todas as funcionalidades deste trabalho estarao dentro deste namespace,
 * para melhorar a organizacao do codigo.
 * Definir namespaces eh particularmente interessante quando o objetivo
 * eh a criacao de bibliotecas.
 */
namespace ConcurrentProgramming {
	
	/* Classe GPU_Smooth:
	 * Classe que aplica o estencil smooth usando CUDA / GPU.
	 * Deriva de Netpbm_Smooth e implementa seus metodos abstratos.
	 */
	class GPU_Smooth :
		public Netpbm_Smooth {
		
	protected:
		
		/* Metodo _smooth:
		 * Metodo de fachada cuja assinatura bate com o do metodo abstrato que
		 *   que ele implmenta, apenas para caracterizar (re)implmentacao
		 *   de metodo abstrato. Seu proposito eh gerenciar preparar o ambiente
		 *   no host e no device.
		 * Implementa o metodo abstrato de mesmo nome na classe Netpbm_Smooth.
		 * Esse metodo nao possui parametros.
		 * Retorno:
		 *   ponteiro para imagem suavizada.
		 */
		Netpbm_Smooth* _smooth() {
			
			// Em GPU, eh possivel tratar cada cor como se fosse um pixel
			// de imagem. Portanto, o tamanho eh o produto da altura
			// pela largura pelo numero de cores.
			dword size = this->height * this->width * this->dimension;
			size_t memory_size = size * sizeof(byte);
			
			// Usar-se-a, por razoes de eficiencia, 1024 threads por bloco.
			const word n_threads = 1024;
			
			// Assim, tem-se tamanho / 1024 blocos de processamento.
			// Essa divisão eh feita com teto, isto eh, se a divisao
			// nao for exata, sera criado um bloco com poucas threads
			// para realizar o tamanho.
			word n_blocks = ceil((double)size / n_threads);
			
			// Threads e blocos convertidos em variaveis que podem
			// ser usadas na chamada para o kernel.
			dim3 threads(n_threads);
			dim3 blocks(n_blocks);
			
			// Dados da fonte (imagem original) e destino (imagem suavizada).
			byte* source;
			byte* destination;
			
			// Aloca espaco para esses ponteiros / vetores.
			cudaExec(cudaMalloc(&source, memory_size));
			cudaExec(cudaMalloc(&destination, memory_size));
			
			// Copia da imagem para a fonte, para a fonte poder ser usada
			// em processamento CUDA.
			cudaExec(cudaMemcpy(source, this->data.data(), memory_size, cudaMemcpyHostToDevice));
			
			// Chama o metodo disparador do smooth.
			call_smooth(blocks, threads, destination, source, windows_radius, this->height, this->width, this->dimension);
			
			// A fonte nao precisa mais ser acessada. Libera-a, pois.
			cudaExec(cudaFree(source));
			
			// Instancia o objeto que corresponde ao resultado da suavizacao,
			// e copia o conteudo do vetor destino para os dados dessa imagem,
			// fazendo-a conter o resultado do calculo.
			GPU_Smooth* smoothed = new GPU_Smooth(this->width, this->height, this->max_value, this->dimension);
			cudaExec(cudaMemcpy(smoothed->data.data(), destination, memory_size, cudaMemcpyDeviceToHost));
			
			// O destino ja esta armazenado no objeto anterior.
			// Portanto, o destino ja pode ser liberado.
			cudaExec(cudaFree(destination));
			
			// Realiza operacoes finais de termino no device.
			cudaExec(cudaDeviceSynchronize());
			cudaExec(cudaDeviceReset());
			
			// Retorna a imagem suavizada.
			return smoothed;
			
		}
		
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
			return [](void* arg) { return true; };
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
		GPU_Smooth(dword width, dword height, byte max_value, byte dimension)
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
		GPU_Smooth(const string& file_name) 
			: Netpbm_Smooth(file_name) {
		}
		
		/* Metodo parallelism_type:
		 * Recupera uma string de acordo com o tipo de paralelismo
		 *   empregado, para ser usada como sufixo em nomes de arquivo.
		 * Implementa o metodo abstrato de mesmo nome na classe Netpbm_Smooth.
		 * Este metodo nao possui parametros.
		 * Retorno:
		 *   string correspondente ao tipo de paralelismo;
		 *   para o smooth em GPU, essa string eh "gpu".
		 */
		string parallelism_type() const {
			return "gpu";
		}
		
	};
	
}
