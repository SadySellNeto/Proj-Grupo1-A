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
utilitário make para compilar. Em particular usar o comando do linux:
make
compilará o código.

-------------------------------------------------------------------------------

2. Como executar:

O utilitário make, em particular, também conta com regras para a execucao do
executavel. Os comandos:

make run-seq
make run-mpi
make run-gpu

executarão a versão sequencial, a versão usando OpenMP + OpenMPI e a versão
em CUDA, respectivamente. Também é possível executá-los manualmente. Para
isso, eles DEVEM ser executados no diretório que eles se encontram (bin),
via os seguintes comandos.

./smooth seq File
mpirun -np 12 --hostfile etc/hosts.txt ./smooth mpi File
./smooth gpu File

Onde, File é o arquivo de imagem a ser suavizado. Este parâmetro é opcional
em todos os casos; caso ele não seja fornecido, o programa perguntará ao
usuário qual dever ser o arquivo a ser suavizado.

-------------------------------------------------------------------------------

3. Alteração do código para alteração do ponto de vista dos resultados:

É possível alterar o código para que se exiba os tempos em unidades diferentes.
(Por padrão, o programa exibe os tempos de execução em segundos).

Para isso, basta alterar as linhas do arquivo:
604		double_seconds elapsed_time = end - start;

O tipo da variável ("double_seconds", no caso acima) pode ser alterado para mudar a
unidade de tempo. Suas possibilidades são:

double_hours: horas;
double_minutes: minutos;
double_seconds: segundos (padrão);
double_milliseconds: milissegundos;
double_microseconds: microssegundos;
double_nanoseconds: nanossegundos.
-------------------------------------------------------------------------------

4. Agradecimentos:

Agradecemos ao professor Julio Cezar Estrella pela atenção e prestatividade,
e a todos que chegaram neste ponto do documento; fato que significa que,
provavelmente, leram ele inteiro.
-------------------------------------------------------------------------------
Copyright (C) 2015 Loys Henrique Sacomano Gibertoni, Sady Sell Neto
===============================================================================