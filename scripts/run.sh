#!/bin/sh
#PBS -l nodes=1:ppn=1:r662
#PBS -q mei
#PBS -l walltime=2:00:00
#PBS -N dot_pr_1_vec
#PBS -e /home/a77070/projects/aa/WorkAssignment/logs/errors.txt
#PBS -o /home/a77070/projects/aa/WorkAssignment/logs/results.txt

cd /home/a77070/projects/aa/WorkAssignment

source ~/intel/parallel_studio_xe_2019/compilers_and_libraries_2019/linux/bin/compilervars.sh intel64

#cat /proc/cpuinfo > tmp/cpuinfo.txt
#papi_avail > tmp/papi_counters.txt

#'DOT_PR_1' 'DOT_PR_1_TR' 'DOT_PR_2' 'DOT_PR_3' 'DOT_PR_3_TR' 'DOT_PR_1_BL' 'DOT_PR_2_BL' 'DOT_PR_3_BL'
algs=( 'DOT_PR_1_BL' 'DOT_PR_2_BL' 'DOT_PR_3_BL' )

#32 128 1024 2048
sizes=( 2048 )

#'l1mr' 'l2mr' 'l3mr' 'flops' 'vflops' 'time'
options=( 'time' )

for alg in ${algs[@]}
do
    if [ $alg == 'DOT_PR_1' ]; then
        export DOT_PR_1=yes
        make clean
        make
        echo ""
        echo "Index Order: i-j-k"
        for size in ${sizes[@]}
        do
            echo ""
            echo Size of matrix $size x $size
            for type in 
            do
                ./bin/main $size $type
            done
        done
        export DOT_PR_1=no
    elif [ $alg == 'DOT_PR_1_TR' ]; then
        export DOT_PR_1_TR=yes
        make clean
        make
        echo ""
        echo "Index Order: i-j-k Transposed"
        for size in ${sizes[@]}
        do
            echo ""
            echo Size of matrix $size x $size
            for type in ${options[@]}
            do
                ./bin/main $size $type
            done
        done
        export DOT_PR_1_TR=no
    elif [ $alg == 'DOT_PR_1_BL' ]; then
        export DOT_PR_1_BL=yes
        make clean
        make
        echo ""
        echo "Index Order: i-j-k Transposed with Blocking"
        for size in ${sizes[@]}
        do
            echo ""
            echo Size of matrix $size x $size
            for type in ${options[@]}
            do
                ./bin/main $size $type
            done
        done
        export DOT_PR_1_BL=no
    elif [ $alg == 'DOT_PR_2' ]; then
        export DOT_PR_2=yes
        make clean
        make
        echo ""
        echo "Index Order: i-k-j"
        for size in ${sizes[@]}
        do
            echo ""
            echo Size of matrix $size x $size
            for type in ${options[@]}
            do
                ./bin/main $size $type
            done
        done
        export DOT_PR_2=no
    elif [ $alg == 'DOT_PR_2_BL' ]; then
        export DOT_PR_2_BL=yes
        make clean
        make
        echo ""
        echo "Index Order: i-k-j with Blocking"
        for size in ${sizes[@]}
        do
            echo ""
            echo Size of matrix $size x $size
            for type in ${options[@]}
            do
                ./bin/main $size $type
            done
        done
        export DOT_PR_2_BL=no
    elif [ $alg == 'DOT_PR_3' ]; then
        export DOT_PR_3=yes
        make clean
        make
        echo ""
        echo "Index Order: j-k-i"
        for size in ${sizes[@]}
        do
            echo ""
            echo Size of matrix $size x $size
            for type in ${options[@]}
            do
                ./bin/main $size $type
            done
        done
        export DOT_PR_3=no
    elif [ $alg == 'DOT_PR_3_TR' ]; then
        export DOT_PR_3_TR=yes
        make clean
        make
        echo ""
        echo "Index Order: j-k-i Transposed"
        for size in ${sizes[@]}
        do
            echo ""
            echo Size of matrix $size x $size
            for type in ${options[@]}
            do
                ./bin/main $size $type
            done
        done
        export DOT_PR_3_TR=no
    elif [ $alg == 'DOT_PR_3_BL' ]; then
        export DOT_PR_3_BL=yes
        make clean
        make
        echo ""
        echo "Index Order: j-k-i Transposed with Blocking"
        for size in ${sizes[@]}
        do
            echo ""
            echo Size of matrix $size x $size
            for type in ${options[@]}
            do
                ./bin/main $size $type
            done
        done
        export DOT_PR_3_BL=no
    fi
done