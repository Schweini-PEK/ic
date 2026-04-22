# IC_PCG(sptrsv) 实验说明

## 项目说明
本项目是面向高性能计算场景，探索融合并行调度与精度优化的算子加速方法。
目前已经设计并展开了实验，研究不同矩阵在“低精度+重排序+预缩放”共同作用的效果。
## 数据集获取方式
当前实验中使用到的矩阵包括：
- `nos5`
- `bcsstk14`

### 命令行下载方式：
1.bcsstk14：
- wget -c http://sparse-files.engr.tamu.edu/MM/HB/bcsstk14.tar.gz
- tar -xzf bcsstk14.tar.gz
- find . -type f -name "*.mtx"
- mkdir -p data/matrices/bcsstk14
- cp ./bcsstk14/bcsstk14.mtx data/matrices/bcsstk14/bcsstk14.mtx

2.nos5：
- wget -c http://sparse-files.engr.tamu.edu/MM/HB/nos5.tar.gz
- tar -xzf nos5.tar.gz
- find . -type f -name "*.mtx"
- mkdir -p data/matrices/bcsstk14
- cp ./bcsstk14/bcsstk14.mtx data/matrices/bcsstk14/bcsstk14.mtx

### config配置内容：
1.bcsstk14：
matrix_path=./data/matrices/bcsstk14/bcsstk14.mtx
ordering=[Identity,AMD,RCM]
precision_fact=double
precision_pcg=[double,float]

2.nos5：
matrix_path=./data/matrices/nos5/nos5.mtx
ordering=[Identity,AMD,RCM]
precision_fact=double
precision_pcg=[double,float]

### 运行指令：
cmake --build build --target ict_pcg -j

./build/src/ict_pcg config/ict_pcg_options_nos5.txt

./build/src/ict_pcg config/ict_pcg_options_bcsstk14.txt

./build/src/ict_pcg config/ict_pcg_options_nos5.txt | tee data/results/sptrsv_precision/results_nos5.txt 

./build/src/ict_pcg config/ict_pcg_options_nos7.txt | tee data/results/sptrsv_precision/results_nos7.txt

./build/src/ict_pcg config/ict_pcg_options_bcsstk14.txt | tee data/results/sptrsv_precision/results_bcsstk14.txt

./build/src/ict_pcg config/ict_pcg_options_bcsstk19.txt | tee data/results/sptrsv_precision/results_bcsstk19.txt

./build/src/ict_pcg config/ict_pcg_options_bcsstk21.txt | tee data/results/sptrsv_precision/results_bcsstk21.txt

./build/src/ict_pcg config/ict_pcg_options_bcsstk18.txt | tee data/results/sptrsv_precision/results_bcsstk18.txt

./build/src/ict_pcg config/ict_pcg_options_nasa4704.txt | tee data/results/sptrsv_precision/results_nasa4704.txt

./build/src/ict_pcg config/ict_pcg_options_bcsstk24.txt | tee data/results/sptrsv_precision/results_bcsstk24.txt

