##
# Universidade de Sao Paulo - USP
# Instituto de Ciencias Matematicas e Computacao - ICMC
# Departamento de Sistemas de Computacao - SSC
# Programacao Concorrente - SSC0143
# Professor Doutor Julio Cezar Estrella
# 
# Nomes / Nos. USP:
#   Loys Henrique Sacomano Gibertoni - 8532377
#   Sady Sell Neto - 8532418
# Data: 3 de dezembro de 2015
# 
# Copyright (C) 2015 Loys Henrique Sacomano Gibertoni, Sady Sell Neto
##

##
# Makefile:
# Utilitario para compilacao, execucao e gerenciamento
# de arquivos do trabalho.
##

# Diretorios do projeto:

SRC=src
OBJ=obj
BIN=bin
MISC=etc

# Arquivos de host do para a implementacao OpenMPI

HOSTFILE=$(MISC)/hosts.txt
NHOSTS=$(shell cat $(HOSTFILE) | wc -l)

# Flags do mpic++, para nao precisar usa-lo explicitamente.

MPI_COMPILE_FLAGS=$(shell mpic++ --showme:compile)
MPI_LINK_FLAGS=$(shell mpic++ --showme:link)

all: clean
	
	@# Compilacao das bibliotecas e geracao dos subprogramas
	
	g++ -std=c++11 -I$(SRC) $(SRC)/Simple_main.cpp -o $(BIN)/.seq -w -O3 -march=native
	g++ -std=c++11 -I$(SRC) $(MPI_COMPILE_FLAGS) $(SRC)/MPI_main.cpp -o $(BIN)/.mpi -fopenmp -w -O3 -march=native $(MPI_LINK_FLAGS)
	-nvcc -c -I$(SRC) $(SRC)/smooth.cu -o $(OBJ)/smooth.o -w -O3 -Xcompiler "-march=native"
	-nvcc -std=c++11 -I$(SRC) $(SRC)/GPU_main.cpp $(OBJ)/smooth.o -o $(BIN)/.gpu -w -O3 -Xcompiler "-march=native"
	
	@# Geracao do executavel principal
	
	g++ -std=c++11 $(SRC)/main.cpp -o $(BIN)/smooth -w -O3 -march=native
	
	@# Concecao de permissao de execucao para o script de amostragem
	chmod u+x $(MISC)/sample_times.sh

run:
	$(MISC)/sample_times.sh

clean:
	find -name "*~" | xargs rm -rf
	rm -rf $(OBJ)/* $(BIN)/*
