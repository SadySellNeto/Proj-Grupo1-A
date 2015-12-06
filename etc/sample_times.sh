ROOTDIR=$PWD
NHOSTS=$(cat $ROOTDIR/etc/hosts.txt | wc -l)

printf "%-4s" "n"
printf "%-73s" "file"
printf "%-7s" "seq"
printf "%-7s" "mpi"
printf "%-7s" "gpu"
echo ""
echo ""

cd bin/

for i in $(seq 1 10)
do
	
	printf "%-4d" $i
	f=/home/grupo01a/img/gray/cappadocia_balloon.pgm
	printf "%-73s" $f
	./smooth seq $f > $f.seq$i.txt
	printf "%-7s" "done."
	mpirun -np $NHOSTS --hostfile $ROOTDIR/etc/hosts.txt ./smooth mpi $f > $f.mpi$i.txt
	printf "%-7s" "done."
	./smooth gpu $f > $f.gpu$i.txt
	printf "%-7s" "done."
	echo ""
	
	printf "%-4d" $i
	f=/home/grupo01a/img/gray/cidade_da_horta.pgm
	printf "%-73s" $f
	./smooth seq $f > $f.seq$i.txt
	printf "%-7s" "done."
	mpirun -np $NHOSTS --hostfile $ROOTDIR/etc/hosts.txt ./smooth mpi $f > $f.mpi$i.txt
	printf "%-7s" "done."
	./smooth gpu $f > $f.gpu$i.txt
	printf "%-7s" "done."
	echo ""
	
	printf "%-4d" $i
	f=/home/grupo01a/img/gray/flower_duo.pgm
	printf "%-73s" $f
	./smooth seq $f > $f.seq$i.txt
	printf "%-7s" "done."
	mpirun -np $NHOSTS --hostfile $ROOTDIR/etc/hosts.txt ./smooth mpi $f > $f.mpi$i.txt
	printf "%-7s" "done."
	./smooth gpu $f > $f.gpu$i.txt
	printf "%-7s" "done."
	echo ""
	
	printf "%-4d" $i
	f=/home/grupo01a/img/gray/italian_valley.pgm
	printf "%-73s" $f
	./smooth seq $f > $f.seq$i.txt
	printf "%-7s" "done."
	mpirun -np $NHOSTS --hostfile $ROOTDIR/etc/hosts.txt ./smooth mpi $f > $f.mpi$i.txt
	printf "%-7s" "done."
	./smooth gpu $f > $f.gpu$i.txt
	printf "%-7s" "done."
	echo ""
	
	printf "%-4d" $i
	f=/home/grupo01a/img/rgb/cappadocia_balloon.ppm
	printf "%-73s" $f
	./smooth seq $f > $f.seq$i.txt
	printf "%-7s" "done."
	mpirun -np $NHOSTS --hostfile $ROOTDIR/etc/hosts.txt ./smooth mpi $f > $f.mpi$i.txt
	printf "%-7s" "done."
	./smooth gpu $f > $f.gpu$i.txt
	printf "%-7s" "done."
	echo ""
	
	printf "%-4d" $i
	f=/home/grupo01a/img/rgb/cidade_da_horta.ppm
	printf "%-73s" $f
	./smooth seq $f > $f.seq$i.txt
	printf "%-7s" "done."
	mpirun -np $NHOSTS --hostfile $ROOTDIR/etc/hosts.txt ./smooth mpi $f > $f.mpi$i.txt
	printf "%-7s" "done."
	./smooth gpu $f > $f.gpu$i.txt
	printf "%-7s" "done."
	echo ""
	
	printf "%-4d" $i
	f=/home/grupo01a/img/rgb/flower_duo.ppm
	printf "%-73s" $f
	./smooth seq $f > $f.seq$i.txt
	printf "%-7s" "done."
	mpirun -np $NHOSTS --hostfile $ROOTDIR/etc/hosts.txt ./smooth mpi $f > $f.mpi$i.txt
	printf "%-7s" "done."
	./smooth gpu $f > $f.gpu$i.txt
	printf "%-7s" "done."
	echo ""
	
	printf "%-4d" $i
	f=/home/grupo01a/img/rgb/italian_valley.ppm
	printf "%-73s" $f
	./smooth seq $f > $f.seq$i.txt
	printf "%-7s" "done."
	mpirun -np $NHOSTS --hostfile $ROOTDIR/etc/hosts.txt ./smooth mpi $f > $f.mpi$i.txt
	printf "%-7s" "done."
	./smooth gpu $f > $f.gpu$i.txt
	printf "%-7s" "done."
	echo ""
	
	echo ""
	
done

cd ..