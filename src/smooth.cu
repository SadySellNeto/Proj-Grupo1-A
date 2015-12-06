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
 * smooth.cu:
 * Implementacao das funcoes / kernels de suavizacao em GPU.
 * Nao pode ser ao arquivo "GPU_Smooth.cpp", para que o compilador
 * possa diferenciar codigo C++ e codigo CUDA.
 */

#include "smooth.h" // global_smooth, call_smooth

// Para evitar inclusoes de outros codigos,
// nao sera possivel usar os alias definidos para
// os tipos. Eles terao que ser usado pelos seus
// nomes extensos

/**
 * Procedimento global_smooth:
 * Calcula e efetiva a suavizacao de uma imagem.
 * Qualificador __global__: o procedimento eh um kernel, e varias threads
 *   o executarao.
 * Parametros:
 *   destination: ponteiro / vetor para onde serao salvos os dados;
 *   source: ponteiro / vetor de onde serao lidos os dados
 *     (dados da imagem a serem suavizados);
 *   windows_radius: raio da janela de suavizacao;
 *   height: altura da imagem a ser suavizada;
 *   width: largura da imagem a ser suavizada;
 *   dimension: numero de cores da imagem a ser suavizada;
 * Procedimentos nao possuem retornos.
 */
__global__ void global_smooth(unsigned char* destination, const unsigned char* source, unsigned int windows_radius, unsigned long int height, unsigned long int width, unsigned char dimension) {
	
	// Obtem a posicao baseada na thread.
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Verifica se a posicao esta dentro dos limites do vetor.
	// Se estiver fora, encerra o procedimento sem efetuar nenhum
	// calculo ou operacao.
	if (pos >= height * width * dimension) {
		return;
	}
	
	// Numero de pontos usados na media.
	unsigned char count = 0;
	
	// Soma do valor dos pontos usados na media.
	unsigned int sum = 0;
	
	// Linha, coluna e cor da posicao atual.
	// Seriam como se fosse os indices de um arranjo 3D,
	// caso o ponteiro fosse um.
	long int row = pos / dimension / width;
	long int col = pos / dimension % width;
	long int color = pos % dimension;
	
	// Contadores para percorrer a janela nas direcoes x e y.
	long int x;
	long int y;
	
	// Lacos que percorrem a janela 5x5 cuja media sera utilizada
	// como valor do pixel;
	for (x = row - windows_radius; x <= row + windows_radius; x++) {
		for (y = col - windows_radius; y <= col + windows_radius; y++) {
			
			// Se o pixel atual da janela estiver dentro da area
			// da imagem;
			if (x >= 0 && x < height && y >= 0 && y < width) {
				
				// Eele deve ser considerado no calculo da media,
				// entao o contador eh incrementado;
				count++;
				
				// E o valor de do pixel eh somado aa soma;
				sum += source[x * width * dimension + y * dimension + color];
			}
			
		}
	}
	
	// Calcula a media e atribui no centro da janela, assim efetivando
	// a suavizacao.
	destination[pos] = sum / count;
	
}

/**
 * Procedimento call_smooth:
 * Encapsula a chamada para o procedimento global_smooth, atuando como a ponte
 *   que liga C++ e CUDA.
 * Parametros:
 *   blocks: variavel CUDA que representa o numero de blocos por grid
 *     que serao usadas no kernel;
 *   threads: variavel CUDA que representa o numero de threads por bloco
 *     que serao usadas no kernel;
 *   destination: ponteiro / vetor para onde serao salvos os dados;
 *   source: ponteiro / vetor de onde serao lidos os dados
 *     (dados da imagem a serem suavizados);
 *   windows_radius: raio da janela de suavizacao;
 *   height: altura da imagem a ser suavizada;
 *   width: largura da imagem a ser suavizada;
 *   dimension: numero de cores da imagem a ser suavizada;
 * Procedimentos nao possuem retornos.
 */
void call_smooth(dim3 blocks, dim3 threads, unsigned char* destination, const unsigned char* source, unsigned int windows_radius, unsigned long int height, unsigned long int width, unsigned char dimension) {
	// Simplesmente chama o kernel sem mais delongas.
	global_smooth<<<blocks, threads>>>(destination, source, windows_radius, height, width, dimension);
}
