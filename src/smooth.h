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

#pragma once

/**
 * smooth.h:
 * Cabecalho e definicoes dos kernels e suas chamadas.
 * Nao pode ser mesclado ao arquivo "smooth.cu", e tampouco podem
 * eles serem mesclados ao arquivo "GPU_Smooth.cpp", para que o compilador
 * possa diferenciar codigo C++ e codigo CUDA.
 */

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
__global__ void global_smooth(unsigned char* destination, const unsigned char* source, unsigned int windows_radius, unsigned long int height, unsigned long int width, unsigned char dimension);

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
void call_smooth(dim3 blocks, dim3 threads, unsigned char* destination, const unsigned char* source, unsigned int windows_radius, unsigned long int height, unsigned long int width, unsigned char dimension);