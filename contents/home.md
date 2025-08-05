

[![zixiaozhao0921](https://img.shields.io/badge/zixiaozhao0921-github-blue?logo=github)](https://github.com/zixiaozhao0921)[]()

He is currently pursuing a Bachelor's Degree in Mathematics and Applied Mathematics, Shanghai Jiao Tong University, China.



#### Email
sjtu_zzx@sjtu.edu.cn

#### Education
B.S., Mathematics and Applied Mathematics, Shanghai Jiao Tong University, 2022-2026

#### Research Interests
High Performance Computing

### TODO List

- CUDA基础、CUDA Profiling
	- Nsight Systems性能分析工具 [[b站教程]](https://www.bilibili.com/video/BV1UP411s7nE/?spm_id_from=333.337.search-card.all.click&vd_source=b2fcf1c28abf8bb0d1f1c65bb8775bd0)
	- Roofline model [[b站教程]](https://www.bilibili.com/video/BV1f34y1G741?spm_id_from=333.788.videopod.sections&vd_source=b2fcf1c28abf8bb0d1f1c65bb8775bd0)
	- 《CUDA编程基础入门系列》4.5共享内存及之后
	- cmake学习 [[官方文档]](https://cmake.org/documentation/) 通过CMAke搭建编译链?
	- Qt生态？
	- CUDA Tutorial [[Github教程]](https://github.com/PaddleJitLab/CUDATutorial)
	- Makefile? Makelist?
- Poisson Solver Using FFT on GPU
	- 《The Discrete Cosine Transform, Gilbert Strang》
- mfem
	- [[Tutorial]](https://mfem.org/tutorial/)
	- ...腾讯元宝AI上有历史流程记录
- Parallel Computing

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

	- [[Applications of Parallel Computers 2022, UCB CS267]](https://sites.google.com/lbl.gov/cs267-spr2022)
	- [[FFT solver for Poisson equation]](https://youjunhu.github.io/research_notes/particle_simulation/particle_simulationsu24.html)
	- [[]]()

	





