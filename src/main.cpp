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
 * main.cpp:
 * Arquivo principal que realiza entrada, saida e processamento,
 * e utilizacao das bibliotecas para aplicacao do filtro na imagem
 * fornecida.
 */

#include <vector> // vector
#include <string> // string
#include <iostream> // cout, cerr
#include <stdexcept> // out_of_range

#include <unistd.h>

// Usando namespace padrao, para evitar qualificao completa de nomes.
using namespace std;

/* Funcao _main:
 * Verdadeira funcao principal do codigo. A funcao "main" chamara esta funcao.
 *   que fara todo o servico do programa. Porem, diferentemente da main,
 *   os argumentos estao contidos em um vetor de strings, alem de nao incluirem
 *   o proprio nome do programa como argumento, se assemelhando a main do java
 *   e facilitando a manipulacao de tais argumentos. E como a funcao eh inline,
 *   a perda de performance eh pifia para nao dizer nula.
 * Parametros:
 *   args: argumentos da linha de comando.
 * Esta funcao nao possui retorno.
 */
inline void _main(const vector<string>& args) {
	
	// Programa a ser execuado e seus argumentos.
	const char* program;
	char* p_args[3];
	p_args[2] = NULL;
	
	// Erros podem acontecer, portanto, bloco try-catch.
	try {
		
		// Nome do arquivo sobre o qual aplicar-se-a o filtro.
		string file_name;
		
		// Se o argumento de nome de arquivo nao for fornecido:
		if (args.size() < 2u) {
			// Solicita tal informacao ao usuario.
			cout << "Enter the smooth file: ";
			cin >> file_name;
		} else {
			// Caso contrario, atribui o nome de arquivo fornecido.
			file_name = args[1];
		}
		
		// Configura o argumento de nome de arquivo.
		p_args[1] = (char*)file_name.data();
		
		// Se apenas o argumento de modo nao for especificado, assumir-se-a
		// execucao sequencial. O usuario sera notificado, no entanto.
		if (args.size() < 1u) {
			// Configura o programa a ser executado a ser o sequencial.
			program = "./.seq";
			p_args[0] = "./.seq";
		} else {
			
			// Caso o argumento seja "seq":
			if (args[0] == "seq") {
				// Configura o programa a ser executado a ser o sequencial.
				program = "./.seq";
				p_args[0] = "./.seq";
			
			// Caso o argumento seja "mpi":
			} else if (args[0] == "mpi") {
				// Configura o programa a ser executado a ser o paralelo
				// com OpenMP + OpenMPI.
				program = "./.mpi";
				p_args[0] = "./.mpi";
			
			// Caso o argumento seja "gpu":
			} else if (args[0] == "gpu") {
				// Configura o programa a ser executado a ser o paralelo
				// com CUDA / GPU.
				program = "./.gpu";
				p_args[0] = "./.gpu";
			
			// Caso nao seja nenhum desses,
			} else {
				// Lanca uma excecao de modo invalido.
				throw out_of_range("Mode not recognized. Use one of: \"seq\", \"mpi\" or \"gpu\".");
			}
			
			// Executa o programa smooth, de acordo como os parametros montados.
			// Se o programa nao pode ser iniciado:
			if (execvp(program, p_args) != 0) {
				// Sinaliza erro.
				throw runtime_error("Could not execute smooth program.");
			}
			
		}
	
	// Pega excecoes genercias do sistema:
	} catch (exception& e) {
		
		// Infroma, na saida de erro padrao, que uma excecao ocorreu.
		cerr << "Exception ocurred:" << endl;
		// Informa a mensagem associada com a excecao.
		cerr << e.what() << endl;
	
	// Pega qualquer tipo de excecao nao pega:
	} catch (...) {
		
		// Informa, na saida de erro padrao, que ocorreu uma excecao
		// desconhecida; pois se fosse conhecida, teria sido pela pelo
		// catch anterior.
		cerr << "An unknown error occurred!" << endl;
		
	}
	
}

/* Funcao main:
 * Funcao principal / ponto de entrada do programa.
 * Parametros:
 *   argc: numero de argumentos da linha de comando, contando a propria
 *     invocacao do programa.
 *   argv: valor dos argumentos da linha de comando como string de C
 *     (vetor de char).
 * Retorno:
 *   status de saida do programa, retornado para o sistema operacional.
 */
int main(int argc, char* argv[]) {
	
	// Empacota os argumentos, menos a propria invocacao do programa,
	// em um vetor de strings e chama a verdadeira funcao principal, "_main".
	_main(vector<string>(argv + 1, argv + argc));
	
	// Retorna zero (encerrou com sucesso) para o sistema operacional.
	return 0;
}
