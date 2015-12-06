===============================================================================
Universidade de Sao Paulo - USP
Instituto de Ciencias Matematicas e Computacao - ICMC
Departamento de Sistemas de Computacao - SSC
Professor Doutor Julio Cezar Estrella

Autoria (Nome - No. USP):
Loys Henrique Sacomano Gibertoni - 8532377
Sady Sell Neto - 8532418

Copyright (C) 2015 Loys Henrique Sacomano Gibertoni, Sady Sell Neto
-------------------------------------------------------------------------------

1. Como compilar:

O .zip possui um makefile simples, mas operativo. Portanto basta usar o
utilit�rio make para compilar. Em particular usar o comando do linux:
make
compilar� o c�digo.

-------------------------------------------------------------------------------

2. Como executar:

O utilit�rio make, em particular, tamb�m conta com regras para a execucao do
executavel. Os comandos:

make run-seq
make run-mpi
make run-gpu

executar�o a vers�o sequencial, a vers�o usando OpenMP + OpenMPI e a vers�o
em CUDA, respectivamente. Tamb�m � poss�vel execut�-los manualmente. Para
isso, eles DEVEM ser executados no diret�rio que eles se encontram (bin),
via os seguintes comandos.

./smooth seq File
mpirun -np 12 --hostfile etc/hosts.txt ./smooth mpi File
./smooth gpu File

Onde, File � o arquivo de imagem a ser suavizado. Este par�metro � opcional
em todos os casos; caso ele n�o seja fornecido, o programa perguntar� ao
usu�rio qual dever ser o arquivo a ser suavizado.

-------------------------------------------------------------------------------

3. Altera��o do c�digo para altera��o do ponto de vista dos resultados:

� poss�vel alterar o c�digo para que se exiba os tempos em unidades diferentes.
(Por padr�o, o programa exibe os tempos de execu��o em segundos).

Para isso, basta alterar as linhas do arquivo:
604		double_seconds elapsed_time = end - start;

O tipo da vari�vel ("double_seconds", no caso acima) pode ser alterado para mudar a
unidade de tempo. Suas possibilidades s�o:

double_hours: horas;
double_minutes: minutos;
double_seconds: segundos (padr�o);
double_milliseconds: milissegundos;
double_microseconds: microssegundos;
double_nanoseconds: nanossegundos.
-------------------------------------------------------------------------------

4. Agradecimentos:

Agradecemos ao professor Julio Cezar Estrella pela aten��o e prestatividade,
e a todos que chegaram neste ponto do documento; fato que significa que,
provavelmente, leram ele inteiro.
-------------------------------------------------------------------------------
Copyright (C) 2015 Loys Henrique Sacomano Gibertoni, Sady Sell Neto
===============================================================================