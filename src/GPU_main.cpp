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
 * GPU_main.cpp:
 * Ponto de entrada para o programa paralelo usando GPU / CUDA.
 */

#include "GPU_Smooth.cpp"

// Usando, tambem, o namespace do trabalho.
using namespace ConcurrentProgramming;

int main(int argc, char* argv[]) {
	
	GPU_Smooth* img = new GPU_Smooth(argv[1]);
	img->smooth(true, true);
	return 0;
	
}
