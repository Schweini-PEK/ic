# IC-PCG SpTRSV 展示系统

该目录用于搭建毕设展示系统。展示系统不重新实现算法，只负责选择实验配置、调用项目可执行程序、保存本次运行结果并进行前端可视化。

## 目录结构

```text
demo_web/
├── backend/
├── frontend/
└── runs/
```

`runs/` 用于保存展示系统产生的新实验结果，不覆盖项目原有的 `data/results/`。

## 启动方式

在WSL上启动：
```bash
cd ~/workspace/dev_lexie/demo_web/backend

uvicorn main:app --host 0.0.0.0 --port 8000
```

浏览器访问：
```text
http://100.112.194.52:8000
```

mac测试前端：
```bash
python3 demo_web/backend/dev_server.py
```

## 运行前提

展示系统依赖 7 个 SuiteSparse 测试矩阵。矩阵文件较大，不建议提交到 Git。首次在 Dell WSL 上运行时，请先在项目根目录执行：

```bash
python3 demo_web/tools/download_matrices.py
```

下载完成后应得到如下路径：

```text
data/matrices/bcsstk21/bcsstk21.mtx
data/matrices/Dubcova1/Dubcova1.mtx
data/matrices/Dubcova2/Dubcova2.mtx
data/matrices/fv1/fv1.mtx
data/matrices/kuu/kuu.mtx
data/matrices/nasa2146/nasa2146.mtx
data/matrices/Pres_Poisson/Pres_Poisson.mtx
```

展示系统会调用项目已有可执行程序：

```bash
./build/src/ict_pcg
```

如果该文件不存在，请先在项目根目录编译：

```bash
cmake -S . -B build
cmake --build build --target ict_pcg -j
```

点击页面中的“一键运行全部矩阵”后，后端会为 7 个矩阵生成临时配置文件，固定使用 `ordering=RCM`、`scaling=UnitSqrtDiag`，并根据页面选择切换 `factorized_precond_policy` 和 `precision_pcg`。运行结果按实验组合固定保存在：

```text
demo_web/runs/
├── level_double/
│   ├── bcsstk21.txt
│   ├── Dubcova1.txt
│   ├── ...
│   ├── configs/
│   └── summary.json
├── level_float/
├── super_double/
└── super_float/
```

同一组合再次运行时会覆盖该组合目录下的上一次展示结果，但不会覆盖 `data/results/` 下原有实验结果。

页面分为两部分：

1. 运行算法并生成实验结果。
2. 选择 `demo_web/runs/` 下两组实验结果，按 `A时间 / B时间` 计算 speedup，并绘制 SpTRSV 平均时间和 PCG 总时间两类对比图。
