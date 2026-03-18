# ic / SpTRSV 实验说明

## 项目说明
本项目内容是 level scheduling版 SpTRSV 的实验，实验目的是观察在不同的矩阵里，fp64和fp32精度下的level scheduling sptrsv 会有什么样不同的表现。

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
