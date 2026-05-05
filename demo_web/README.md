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

在具备 Python 环境的机器上安装依赖：

```bash
cd demo_web/backend
python3 -m pip install -r requirements.txt
```

启动后端：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

浏览器访问：

```text
http://localhost:8000
```

## 运行前提

展示系统会调用项目已有可执行程序：

```bash
./build/src/ict_pcg
```

如果该文件不存在，请先在项目根目录编译：

```bash
cmake -S . -B build
cmake --build build --target ict_pcg -j
```

点击页面中的“一键运行全部矩阵”后，后端会为 7 个矩阵生成临时配置文件，固定使用 `ordering=RCM`、`scaling=UnitSqrtDiag`，并根据页面选择切换 `factorized_precond_policy` 和 `precision_pcg`。本次运行结果会保存在：

```text
demo_web/runs/run_YYYYMMDD_HHMMSS/
├── configs/
├── raw/
├── charts/
└── summary.json
```

该目录不会覆盖 `data/results/` 下原有实验结果。
