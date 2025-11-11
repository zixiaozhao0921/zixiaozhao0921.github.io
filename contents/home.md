

[![zixiaozhao0921](https://img.shields.io/badge/zixiaozhao0921-github-blue?logo=github)](https://github.com/zixiaozhao0921)[]()

He is currently pursuing a Bachelor's Degree in Mathematics and Applied Mathematics, Shanghai Jiao Tong University, China.



#### Email
sjtu_zzx@sjtu.edu.cn

#### Education
B.S., Mathematics and Applied Mathematics, Shanghai Jiao Tong University, 2022-2026

#### Research Interests
High Performance Computing

### TODO List

- vim学习[[CSDN Blog]](https://blog.csdn.net/qq_40650558/article/details/104565133)

- Pandas学习[[CSDN Blog]](https://blog.csdn.net/ZHINV_/article/details/146568499)

- TOP TO DO: Poisson Solver using FFT on GPU

| 论文标题 | 主要内容 |
|---------|----------|
| **An Optimized FFT-Based Direct Poisson Solver on CUDA GPUs**<br>《基于CUDA GPU的优化FFT泊松求解器》 | 提出针对NVIDIA GPU的高度多线程FFT泊松求解器，通过内存访问优化（128字节合并事务）、计算与FFT维度交错策略提升性能。支持全周期边界及混合（二维周期+一维Neumann）边界条件，在Tesla/Fermi架构实现140-375 GFLOPS性能。<br>**核心价值**：GPU内存优化、FFT交错策略、边界条件实现、性能基准测试 |
| **FINITE DIFFERENCE METHODS FOR POISSON EQUATION**<br>《泊松方程的有限差分法》 | 系统阐述泊松方程的有限差分法，包括Dirichlet/Neumann边界处理（幽灵点技术）、误差分析（$\|u_I - u_h\|_\infty \leq Ch^2$）及Cell-centered网格变体。提供二阶精度公式推导及MATLAB伪代码。<br>**核心价值**：幽灵点边界处理、误差估计、Cell-centered差分、兼容性条件 |
| **3.5 Finite Differences and Fast Poisson Solvers**<br>《3.5 有限差分与快速泊松求解器》 | 从数值线性代数角度解析泊松求解：1）Kronecker积构建二维差分矩阵 2）FFT法利用可分离特征向量（$y_{kl} = \sin\frac{ik\pi}{N+1}\sin\frac{jl\pi}{N+1}$）实现$O(N^2\log N)$复杂度 3）对比消元法/循环约简/FFT性能。<br>**核心价值**：Kronecker积、可分离特征向量、FFT复杂度分析、FACR算法 |
| **The Discrete Cosine Transform**<br>《离散余弦变换》 | 解析8种DCT变体（DCT-1至DCT-8）的数学基础：1）证明DCT基向量为对称二阶差分矩阵的特征向量 2）边界条件（网格点/中点中心）决定基函数形式 3）关联FFT实现（式7）及图像处理应用（JPEG/MLT）。<br>**核心价值**：DCT正交性证明、边界中心化、图像压缩应用、重叠变换 |

- CUDA基础、CUDA Profiling
	- Nsight Systems [[b站教程]](https://www.bilibili.com/video/BV1UP411s7nE/?spm_id_from=333.337.search-card.all.click&vd_source=b2fcf1c28abf8bb0d1f1c65bb8775bd0)
	- Roofline model [[b站教程]](https://www.bilibili.com/video/BV1f34y1G741?spm_id_from=333.788.videopod.sections&vd_source=b2fcf1c28abf8bb0d1f1c65bb8775bd0)
	- .cu和.cpp的互相引用及Makefile[[b站教程]](https://www.bilibili.com/video/BV1tG411f7FA/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=b2fcf1c28abf8bb0d1f1c65bb8775bd0)
	- CUDA实战-流&事件[[b站教程]](https://www.bilibili.com/video/BV1Eh4y1A7Wi?spm_id_from=333.788.videopod.sections&vd_source=b2fcf1c28abf8bb0d1f1c65bb8775bd0)
	- 《CUDA编程基础入门系列》4.5共享内存及之后
	- 了解构建工具[[历史介绍]](https://baijiahao.baidu.com/s?id=1835925238778529521&wfr=spider&for=pc) [[CMake官方文档]](https://cmake.org/documentation/) [[CMake入门指南博客]](https://blog.csdn.net/wallwayj/article/details/147456408)
	- CUDA Tutorial 初级系列(矩阵乘优化) 中级系列(Reduce优化,GEMM优化,卷积优化)[[Github教程]](https://github.com/PaddleJitLab/CUDATutorial)
	- CUDA基础例子-加速矩阵运算[[知乎学习日记]](https://zhuanlan.zhihu.com/p/640086961)
- mfem
	- [[Tutorial]](https://mfem.org/tutorial/)
	- ...腾讯元宝AI上有历史流程记录
- Parallel Computing
	- [[Data Parallelism Algorithms]](https://dl.acm.org/doi/pdf/10.1145/7902.7903)
	- UCB CS267 Lecture 9
	- Segment scan加速SpMV原理, Parallel prefix cost on p “big” processors复杂度分析 (Lecture 8)
	- Distributed Memory Machine的Network Topology——butterfly的设计原理, MPI Programming实践(Lecture 9)
	- ```MPI_COMM_SPLIT``` (Lecture 10)
	- Why Substitute triangular solves $LX=B$ with multiply by $L^{-1}$? (Lecture 14)

#### My Computer

| 参数                  | 值                                  | 说明                     |
|-----------------------|-------------------------------------|--------------------------|
| **显卡型号**          | NVIDIA GeForce RTX 4070 Laptop GPU  | 笔记本移动端显卡         |
| **CUDA 计算能力**     | 8.9                                 | 支持最新 CUDA 特性       |
| **显存容量**         | 8 GB (8585216000 bytes)             | 中等规模计算适用         |
| **CUDA 核心数**      | 4608 (36 MPs × 128 Cores/MP)        | 并行计算单元数量         |
| **GPU 最大时钟频率** | 2175 MHz (2.17 GHz)                 | 核心运行速度             |
| **内存总线宽度**      | 128-bit                             | 显存带宽                |
| **内存时钟频率**      | 8001 MHz                            | 显存性能指标            |
| **L2 缓存大小**       | 33554432 bytes (32 MB)              | 缓存加速                


| 参数                          | 值      | 说明                                   |
|-------------------------------|---------|----------------------------------------|
| **CUDA 驱动版本**            | 12.7    | 由 NVIDIA 显卡驱动提供                |
| **CUDA Runtime 版本**        | 12.6    | 当前安装的 CUDA Toolkit 版本           |
| **WDDM 模式**               | 启用    | Windows 显示驱动模型（正常）          |
| **统一内存寻址 (UVA)**       | 支持    | 允许 CPU 和 GPU 共享虚拟地址空间      |
| **计算抢占 (Preemption)**    | 支持    | 可中断长时间运行的内核                |


| 参数                                  | 值                | 用途                                |
|---------------------------------------|-------------------|-------------------------------------|
| **最大线程块大小**                   | 1024 threads/block | 每个线程块的最大线程数             |
| **最大网格维度**                     | (2147483647, 65535, 65535) | 全局线程网格大小上限      |
| **共享内存每块**                     | 64 KB             | 线程块内共享内存容量                |
| **寄存器每块**                       | 65536             | 线程块可用的寄存器数量              |
| **多处理器最大线程数**               | 1536              | 每个 SM 的并发线程数                |
| **并发内核执行**                     | 支持（1 个复制引擎） | 可同时运行多个 CUDA 内核           |
| **纹理对齐要求**                     | 支持              | 优化纹理内存访问                    


| **参数**               | **当前值**               | **理论参考值**       | **说明**                     |
|-------------------------|--------------------------|----------------------|------------------------------|
| **GPU 型号**           | NVIDIA RTX 4070 Laptop   | -                    | 笔记本移动端显卡             |
| **PCIe 版本**          | 4.0                      | 4.0 (Max)            | 当前运行在 PCIe 4.0 x8       |
| **PCIe 链路宽度**      | x8                       | x8 (Max)             | 笔记本设计限制               |
| **Host→Device 带宽**   | 12625.8 MB/s (~12.6 GB/s)| PCIe 4.0 x8: 15.75 GB/s | CPU→GPU 数据传输速度        |
| **Device→Host 带宽**   | 12833.3 MB/s (~12.8 GB/s)| PCIe 4.0 x8: 15.75 GB/s | GPU→CPU 数据传输速度        |
| **Device→Device 带宽** | 222594.2 MB/s (~222 GB/s)| RTX 4070: 288 GB/s   | GPU 显存内部拷贝速度         |
| **显存类型**           | GDDR6                    | -                    | 带宽效率约 77% (222/288)     |



#### Study Resources

- CUDA

	- [[谭升博客]](https://face2ai.com/program-blog/)
	- [[CUDA入门课 NVIDIA官方视频]](https://www.bilibili.com/video/BV1JJ4m1P7xW/?spm_id_from=333.337.search-card.all.click&vd_source=b2fcf1c28abf8bb0d1f1c65bb8775bd0)
	- [[CUDA入门指南：从零开始掌握 GPU 并行计算 Blog]](https://blog.csdn.net/weixin_47231119/article/details/146244732)
	- [[CUDA Training Series NVIDIA官方文档]](https://www.olcf.ornl.gov/cuda-training-series/)
	- [[CUDA编译流程 官方文档]](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
	- [[CUDA编程基础入门系列 Bilibili]](https://www.bilibili.com/video/BV1sM4y1x7of?spm_id_from=333.788.videopod.episodes&vd_source=b2fcf1c28abf8bb0d1f1c65bb8775bd0&p=13)
	- [[CUDA配置]](https://zhuanlan.zhihu.com/p/1905402464343164063)
	- [[MSVC轻量级配置]](https://blog.csdn.net/m0_57309959/article/details/139815240)
	- [[CUDA学习笔记知乎专栏]](https://zhuanlan.zhihu.com/p/686594283)
	- [[CUDA入门资源汇总知乎专栏]](https://zhuanlan.zhihu.com/p/346910129)


- FFT & Poisson方程

	- [[从小白到用FFT(快速傅里叶变换)解泊松方程]](https://zhuanlan.zhihu.com/p/391398462)
	- **FFT基础、一维Poisson方程**《数值分析》(张平文 李铁军)第五章——快速Fourier变换：Fourier变换、离散Fourier变换、快速Fourier算法介绍，Dirichlet条件的一维Poisson方程DST算法
	- **二维Poisson方程**《偏微分方程数值解》(黄建国)第二章第4节——求解五点差分格式的快速DST方法
	- **Fourier矩阵角度理解FFT**《Introduction to Linear Algebra》(Fifth Edition)Chapter 9.3, The Fast Fourier Transform
	- **Kronecker Product角度理解二维Poisson方程对应系数矩阵**《Mathematical Methods For Engineers II》 (Gilbert Strang) Chaper 3.5, Finite Differences and Fast Poisson Solvers
	- **DCT的解读**《The Discrete Cosine Transform》 (Gilbert Strang)

- Parallel Computing

	- [[Applications of Parallel Computers 2022, UCB CS267-b站视频]](https://www.bilibili.com/video/BV1PS421978D/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=b2fcf1c28abf8bb0d1f1c65bb8775bd0)
	- [[Applications of Parallel Computers 2022, UCB CS267-官网]](https://sites.google.com/lbl.gov/cs267-spr2022)
	- [[FFT solver for Poisson equation]](https://youjunhu.github.io/research_notes/particle_simulation/particle_simulationsu24.html)
	- [[高性能计算的学习路线 知乎]](https://www.zhihu.com/question/33576416/answer/1243835966)
	- [[高性能计算学习笔记 Github]](https://github.com/Eddie-Wang1120/HPC-Learning-Notes)
	- [[计算数学或者计算力学的研究生或博士生C++要学到什么程度（要搞高性能计算）知乎]](https://www.zhihu.com/question/557675741)	
	- [[零基础OpenMp教程 知乎]](https://zhuanlan.zhihu.com/p/17667388663)
	- [[Roofline Model CSDN-1]](https://download.csdn.net/blog/column/9003100/141645388)
	- [[Roofline Model CSDN-2]](https://blog.csdn.net/m0_57102661/article/details/144042331)
	- [[Roofline Model 知乎]](https://zhuanlan.zhihu.com/p/663545398)
	- [[Paper Reading: Partitioned Global Address Space Languages (分区全局地址空间语言)]](https://zhuanlan.zhihu.com/p/622382722)
	- [[LAPACK]](www.netlib.org/lapack)
	- [[Templates]](www.netlib.org/templates)
	- [[Applied Numerical Linear Algebra]](gams.nist.gov)
	- [[Scalapack]](www.netlib.org/scalapack)
	- [[Etemplates]](www.cs.utk.edu/~dongarra//etemplates)



