此处放我的笔记

## UCB CS267 Application of Parallel Computers

### Memory Hierarchies & Matrix Multiplication

- Load/Store的速度远比Operation(+,-,*,/...)快
- C++/Fortran等的编译器(Compiler)流程：检查程序合法-转为汇编代码-优化汇编代码，优化包括
	- Unroll loops	- Merges two loops together(called fuses loops)	- Reorder loops(called interchange loops)	- Eliminate dead code	- Reorder instruction to improve register reuse and more	- Strength reduction(*2 → <<1)
- Memory access的两种costs
	- **Latency**: cost to load/store $1^{st}$ word (通常用α表示，单位是time)	- **Bandwidth**: average rate to load/store a large chunk (通常用β表示，单位是time/byte)
- 局部性（好的局部性运行速度快）
	- **spatial locality**: accessing things nearby previous accesses	- **temporal locality**: reusing an item that was previously accessed
- **Cache line length**: # of bytes loaded together in one entry (memory to cache in only **one-time** load)
- Associativity (例如direct-mapped): only 1 address (line) in a given range in cache, 每个内存地址只能映射到缓存中唯一的一个特定位置
- Pipelining: 隐藏Latency, 需要找到可并行的操作
- 向量运算的SIMD: Simple Instruction Multiple Data
- Data Dependencies Limit Parallelism(也叫Data Race)	- RAW: Read-After-Write (X = A; B = X)	- WAR: Write-After-Read (A = X; X = B)	- WAW: Write-After-Write (X = A; X = B)	- No problem / dependence for RAR: Read-After-Read- FMA: Fused Multiply Add, 单次和+, *速度相同
- 非连续内存速度慢：
	- Strided load ```...=a[i*4]```
	- Strided store ```a[i*4]=...```
	- Indexed(gather) ```...=a[b[i]]```
	- Indexed(scatter) ```a[b[i]]=...```
- A Simple Model of Memory	- Assume just 2 levels in the hierarchy (fast & slow)	- All data initially in slow memory	- m = number of memory elements moved between fast and slow memory	- $t_m$ = time per slow memory operation (only care about bandwidth, NOT about latency)	- f = number of arithmetic operations	- $t_f$ = time per arithmetic operation << tm	- **CI** (Computational Intensity) = f / m: avarage number of flops per slow memory access	- Minimum possible time = f * $t_f$ when all data in fast memory	- Actual time = f * $t_f$ + m * $t_m$ = f * $t_f$ * (1 + $t_m$/$t_f$ * 1/**CI**)	- Larger **CI** means time closer to minimum f * tf. **CI** is the KEY to algorithm efficiency.	- $t_m$/$t_f$ is called the Machine Balance, is the KEY to machine efficiency.
- Case Study - Matrix Multiplication
	- 加速手段1: Blocked(Tiled) Matrix Multiply
	- 加速手段2: Recursive Matrix Multiplication (不用知道缓存容量$M_{fast}$的大小）
	- Cache Oblivious Methods 缓存无关算法 Cache-aware 缓存相关算法（需要知道$M_{fast}$）
	- Alternate Data Layouts (例如Z-Morton order for recursion) to have better spatial locality
	- **Throrem** (Communication lower bound, Hong & Kung, 1981): 
		Any reorganization of Matrix Multiplication (using only community & associativity) has computational intensity CI = O(($M_{fast})^{1/2}$), so # of words moved between fast/slow memory = Ω($n^3 / (M_{fast})^{1/2}$)
	- Strassen's Matrix Multiply, Asymptotically faster 渐进意义更快
		MatMul: O($n^3$) → O($n^{2.81}$) [2.81 = $log_27$]	<img src="https://i.imgs.ovh/2025/08/14/tAkwH.png"  width="600" />
	
### Roofline Model
- 3个关键指标 	- 【机器的】Arithmetic performance (flops/sec)	- 【机器的】Memory bandwidth (bytes/sec)	- 【算法的】Computational (Arithmetic) Intensity, CI (flops/word or flops/byte)

		<img src="https://i.imgs.ovh/2025/08/15/KA6vg.png"  width="300" />
		
### UCB名词解释

- Threads (线程) & Process (进程)- SRAM: Static Random-Access Memory（静态随机存取存储器）包括L1, L2, L3 cache等- DRAM: Dynamic Random-Access Memory（动态随机存取存储器）包括主内存、显存等- Cashe hit & Cashe miss
	当CPU或计算单元请求的数据已经存在于缓存（Cache）中时，称为缓存命中（反之为miss）- Memory Benchmark	
	内存基准测试，是指通过标准化测试程序或工具，评估计算机内存（DRAM、Cache、HBM等）的性能指标，包括：	- 带宽（Bandwidth）：单位时间内可读写的数据量（GB/s）	- 延迟（Latency）：从发起请求到获取数据的时间（纳秒级）。	- 吞吐量（Throughput）：系统在单位时间内能完成的内存操作次数。- HBM: High Bandwidth Memory, 高宽带内存- ILP: Instruction Level Parallelism
- Pipelining
- SIMD: Single Instruction Multiple 
- FMA: Fused Multiply Add
- CI: Computational Intensity, CI = f/m: average number of flops per slow memory access
- Machine Balance: tm/tf, slow memory access time/fast arithmetic operation time- BLAS: Basic Linear Algebra Subroutines- NUMA: Non-Uniform Memory Access
- SW prefetch: Software prefetching
- POSIX: Portable Operating System Interface可移植操作系统接口- SpGEMM: Sparse General Matrix-Matrix Multiplication，稀疏通用矩阵乘法


## CUDA

- GPU性能指标：核心数、显存容量、计算峰值、显存带宽
- CUDA提供两层API接口，CUDA driver API和CUDA runtime API
- GPU状态查询和配置工具：`NVIDIA-smi`工具
- 编译CUDA文件的命令：`nvcc hello.cu -o hello`(后`./hello.cu`)
- 核函数必须是`void`类型，void前有`__global__`修饰词
- 核函数的特殊性：
	- 只能访问GPU内存
	- 不能使用变长参数
	- 不能使用静态变量
	- 不能使用函数指针
	- 核函数具有异步性
	- 不支持C++的iostream(不能`cout`)
- `cudaDeviceSynchronize()` 主机与设备同步
- 调用CUDA核函数：`someCUDAfunction<<<grid_size, block_size>>>()` 
- GPU线程模型结构：grid(网格) - block(线程块) - thread(线程)[注：每32个线程是一个线程束warp]
- Kernel核函数的内建变量: `gridDim.x, blockDim.x, blockIdx.x, threadIdx.x`
- CUDA线程模型可以组织一维至三维的grid和block
- 二维grid+二维block为例，ID计算方式为:

```
int blockId = blockIdx.x + blockIdx.y * gridDim.x;
int threadId = threadIdx.x + threadIdx.y * blockDim.x;
int id = threadId + blockId * (blockDim.x * blockDim.y);
```
- nvcc编译流程：
	- nvcc先将设备代码编译为PTX (Parallel Thread Execution) 伪汇编代码，再将PTX代码编译为二进制的cubin目标代码。
	- 在将源代码编译为PTX代码时，需要用选项`-arch=compute_XY`指定一个虚拟架构的计算能力，用以确定代码中能够使用的CUDA功能。
	- 在将PTX代码编译为cubin代码时，需要用选项`-code=sm_ZW`指定一个真实架构的计算能力，用以确定可执行文件能够使用的GPU。
	- XY分别指主版本号和次版本号。真实架构版本号要大于等于虚拟架构版本号。可单独指定虚拟架构版本号，不可单独指定真实架构版本号。
	- nvcc可指定多个GPU版本编译，使得编译出来的可执行文件可以在多GPU中执行。编译命令为`-gencode=arch=compute_XY -code=sm_XY`。生成的可执行文件称为胖二进制文件(fatbinary)。
	- 不同版本CUDA编译器在编译CUDA代码时，都有一个默认计算能力：CUDA 9.0~10.2默认计算能力3.0，CUDA 11.6默认计算能力5.2......
- 设置GPU设备 (获取GPU设备数量&设置GPU执行时使用的设备)
 
```
int iDeviceCount = 0;
cudaGetDeviceCount(&iDeviceCount);

int iDev = 0;
cudaSetDevice(iDev);
```
- CUDA内存管理
	- 内存分配 `cudaMalloc`

	```
	主机分配内存：extern void *malloc(unsigned int num_bytes);
	float *fpHost_A;
	fpHost_A = (float *)malloc(nBytes);
	设备分配内存：
	float *fpDevice_A;
	cudaMalloc((float**)&fpDevice_A, nBytes);
	```
	- 数据传递 `cudaMemcpy`

	```
	主机数据拷贝：void *memcpy(void &dest, const void *src, size_t n);
	memcpy((void *)d, (void *)s, nBytes);
	设备数据拷贝：__host__ cudaError_t cudaMemcpy (void *dst, const void *src, size_t count, cudaMemcpyKind kind)
	cudaMemcpy(Device_A, Host_A, nBytes, cudaMemcpyKind)
	cuudaMemcpyKind有四种类型和一种默认类型
		- cudaMemcpyHostToHost
		- cudaMemcpyHostToDevice
		- cudaMemcpyDeviceToHost
		- cudaMemcpyDeviceToDevice
		- cudaMemcpyDefault (默认类型只允许在支持统一虚拟寻址的系统上使用)
	```
	- 内存初始化 `cudaMemset`

	```
	主机内存初始化：void *memset(void *str, int c, size_t n);
	memset(fpHost_A, 0, nBytes);
	设备内存初始化
	cudaMemset(fpDevice_A, 0, nBytes);
	```
	- 内存释放 `cudaFree`

	```
	主机free(pHost_A);
	设备cudaFree(pDevice_A);
	```
- `__device__, __global__, __host__`global就是修饰核函数用的，核函数有可能会调用一些device函数 (在GPU上运行) ，host类型是在CPU上运行的函数类型，通常会默认。
- Debug: `cudaError_t类型, cudaGetErrorName函数(返回字符串), cudaGetErrorString函数(返回字符串)`
Debug检查函数

```
cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess)
    {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line=%d\r\n",
               error_code, 
               cudaGetErrorName(error_code), 
               cudaGetErrorString(error_code), 
               filename, 
               lineNumber);
    }
    return error_code;
}
```
- Debug检测函数示例

```
ErrorCheck(cudaMemset(fpDevice_A, 0, stBytesCount), __FILE__, __LINE__);
```
- 如果用于检测核函数，方法为

```
ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);
ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
```
- CUDA中的计时

```
cudaEvent_t start, stop;
ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
ErrorCheck(cudaEventRecord(start), __FILE__, __LINE__);
cudaEventQuery(start); //此处不可用错误检测函数

/***
运行代码
***/

ErrorCheck(cudaEventRecord(stop), __FILE__, __LINE__);
ErrorCheck(cudaEventSychronize(stop), __FILE__, __LINE__);
float elapsed_time;
ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __LINE__, __LINE__);
printf("Time = %g ms.\n", elapsed_time);

ErrorCheck(cudaEventDestroy(start), __FILE__, __LINE__);
ErrorCheck(cudaEventDestroy(stop), __FILE__, __LINE__);
```
- CUDA运行时API查询GPU信息

```
cudaDeviceProp prop; // prop是一个结构体, 包含GPU的信息
ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);
	printf("Device id:                                 %d\n",
        device_id);
    printf("Device name:                               %s\n",
        prop.name);
    printf("Compute capability:                        %d.%d\n",
        prop.major, prop.minor);
    printf("Amount of global memory:                   %g GB\n",
        prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("Amount of constant memory:                 %g KB\n",
        prop.totalConstMem  / 1024.0);
    printf("Maximum grid size:                         %d %d %d\n",
        prop.maxGridSize[0], 
        prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum block size:                        %d %d %d\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], 
        prop.maxThreadsDim[2]);
    printf("Number of SMs:                             %d\n",
        prop.multiProcessorCount);
    printf("Maximum amount of shared memory per block: %g KB\n",
        prop.sharedMemPerBlock / 1024.0);
    printf("Maximum amount of shared memory per SM:    %g KB\n",
        prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Maximum number of registers per block:     %d K\n",
        prop.regsPerBlock / 1024);
    printf("Maximum number of registers per SM:        %d K\n",
        prop.regsPerMultiprocessor / 1024);
    printf("Maximum number of threads per block:       %d\n",
        prop.maxThreadsPerBlock);
    printf("Maximum number of threads per SM:          %d\n",
        prop.maxThreadsPerMultiProcessor);
```

- CUDA Stream与Event

- 原子操作

```
atomicAdd();
atomicSub();
atomicExch();
atomicMin();
atomicMax();
atomicCAS();
```

### GPU硬件资源、CUDA内存模型

- 流多处理器 (SM)

一个GPU是由多个SM构成的，Fermi架构的SM关键资源如下：

1. CUDA核心 (CUDA core)
2. 共享内存/L1缓存 (shared memory/L1 cache)
3. 寄存器文件 (Register File)
4. 加载和储存单元 (Load/Store Units)
5. 特殊函数单元 (Special Function Unit)
6. Warps调度 (Warps Scheduler)

<img src="https://i.imgs.ovh/2025/07/04/q9B8C.png"  width="600" />


- 线程模型与物理结构

| 类别     | 层级1      | 层级2    | 层级3    |
|----------|------------|----------|----------|
| Software | Thread     | Block    | Grid     |
| Hardware | CUDA Core  | SM       | Device   |

- 线程束 (warp)

在硬件中，网格中的所有线程块需要分配到SM上进行执行。每个线程块内的所有线程会分配到同一个SM中执行，但是每个SM上可以被分配多个线程块。线程块分配到SM中后，会以32个线程为一组进行分割，每个组成为一个线程束 (wrap)。
同一个线程块中的相邻32个线程是一个线程束，因此线程块中包含的线程数量通常是32的倍数，这样有助于物理上的locality，也就有利于运行效率。

<img src="https://i.imgs.ovh/2025/07/04/qJAAq.png"  width="700" />

- CUDA内存模型
	- 寄存器 (register)
	- 共享内存 (shared memory)
	- 局部内存 (local memory)
	- 常量内存 (constant memory)
	- 纹理内存 (tesxture memory)
	- 全局内存 (global memory)

| 内存类型         | 物理位置   | 访问权限   | 可见范围             | 生命周期             |
|------------------|------------|------------|----------------------|----------------------|
| 全局内存         | 在芯片外   | 可读可写   | 所有线程和主机端     | 由主机分配与释放     |
| 常量内存         | 在芯片外   | 对线程仅可读,对Host可读可写     | 所有线程和主机端     | 由主机分配与释放     |
| 纹理和表面内存   | 在芯片外   | 一般仅可读 | 所有线程和主机端     | 由主机分配与释放     |
| 寄存器内存       | 在芯片内   | 可读可写   | 单个线程             | 所在线程             |
| 局部内存         | 在芯片外   | 可读可写   | 单个线程             | 所在线程             |
| 共享内存         | 在芯片内   | 可读可写   | 单个线程块           | 所在线程块           |

<img src="https://i.imgs.ovh/2025/07/04/qDZcm.png"  width="700" />

- 寄存器内存
	-  寄存器内存仅线程内可见，生命周期与线程一致。
	-  内建变量在寄存器中。核函数中定义的不加限定符的变量储存在寄存器中，核函数中定义的不加限定符的数组可能在寄存器中也可能在本地内存中？
	-  在核函数中定义，但储存在本地内存的情况：长度在编译时不确定的数组、可能占用大量寄存器空间的较大本地结构体和数组、任何不满足核函数寄存器限定条件的变量？
	-  不同计算能力的GPU每个SM/每个线程块/每个线程所容纳的最大寄存器数量是不同的：

<img src="https://i.imgs.ovh/2025/07/06/5UPqF.png"  width="700" />

- 本地内存
	- 每个线程最多可使用512K的本地内存
	- 对于计算能力2.0以上的设备，本地内存的数据储存在每个SM的一级缓存和设备的二级缓存中

<img src="https://i.imgs.ovh/2025/07/06/5U49d.png"  width="700" />

- 共享内存
	- 共享内存在线程块内可见，生命周期与线程块一致，可用于线程间通信
	- 使用__shared__修饰的变量存放于共享内存中，可定义动态和静态两种
	- 静态共享
		- 内存声明方式：```__shared__ float tile[size, size]```
		- 作用域：核函数中声明，则作用域局限在这个核函数中；文件核函数外声明，作用域对所有核函数有效？
		- 静态共享内存在编译时就要确定内存大小
	- 在L1缓存和共享内存	使用相同硬件资源的设备上，可通过cudaFuncSetCacheConfig运行时API制定设置首选缓存配置，L1缓存和共享内存大小固定的设备上配置无效。func必须是声明为```__global__```的函数？
	- 访问共享内存必须加入同步机制？
	线程块内同步
	```
	void __syncthreads();
	```
	- 不同计算能力的GPU每个SM/每个线程块所容纳的最大共享内存大小是不同的

<img src="https://i.imgs.ovh/2025/07/06/5UJB6.png"  width="700" />

- 全局内存
	- 使用```__device__```关键字静态声明全局内存
	- ```cudaMemcpyToSymbol```？

- 常量内存
	- 常量内存是有常量缓存的全局内存，大小仅为64K，访问速度比全局内存快
	- 常量内存中的数据对同一编译单元内所有线程可见
	- 使用```__constant__```修饰的变量存放于常量内存中，不能定义在核函数中，且是静态定义的
	- 给核函数传递数值参数时，这个变量就存放于常量内存？
	- 常量内存必须在主机端使用```cudaMemcpyToSymbol```进行初始化

- GPU缓存

- 计算资源分配

- 延迟隐藏

- 避免线程束分化


### CUDA Profiling性能分析

CUDA的最新性能分析工具——Nsight Systems (2025.3.1)
[NVIDIA-Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started)




## MFEM


## Fourier变换, 离散Fourier变换(DFT), 快速Fourier变换(FFT)

**Fourier变换**

$$ \widehat{f}(k)=\int_{-\infty}^{+\infty}f(x)e^{-ikx}dx $$

$$ f(x)=\frac{1}{2\pi}\int_{-\infty}^{+\infty}\widehat{f}(k)e^{ikx}dk $$

**性质：**

1. 
$$ \widehat{f^{\prime}}(k)=ik\widehat{f}(k) $$
2. 
$$ \widehat{f(x-a)}(k)=e^{-ika}\widehat{f}(k) $$
3. 
$$ \widehat{f*g}(k)=\widehat{f}(k)\cdot\widehat{g}(k) $$
4. (Parseval定理) 
$$ \int_{-\infty}^{+\infty}|f(x)|^{2}dx=\frac{1}{2\pi}\int_{-\infty}^{+\infty}|\widehat{f}(k)|^{2}dk $$

**离散Fourier变换**

$$ \vec{a}=(a_{0},a_{1},\cdots,a_{N-1})^{T} $$

$$ \vec{c}=(c_{0},c_{1},\cdots,c_{N-1})^{T}=\widehat{\vec{a}} $$

满足：
$$ c_k = \sum_{j=0}^{N-1} a_{j} e^{-jk\frac{2\pi i}{N}} = \sum_{j=0}^{N-1} a_{j} w^{jk}, \quad k=0,\cdots,N-1 $$

其中 $ w = e^{-\frac{2\pi i}{N}} $ 为N次基本单位根。

逆变换：

$$ a_{j} = \frac{1}{N}\sum_{k=0}^{N-1}c_k e^{jk\frac{2\pi i}{N}} = \frac{1}{N}\sum_{k=0}^{N-1}c_k w^{-jk} $$

矩阵形式：

$$ \vec{c} = F\vec{a} $$

其中 $ F $ 为Fourier矩阵：

$$
F = \begin{pmatrix}
1 & 1 & \cdots & 1 \\
1 & w & \cdots & w^{N-1} \\
\vdots & \vdots & \ddots & \vdots \\
1 & w^{N-1} & \cdots & w^{(N-1)^2}
\end{pmatrix}
$$

可逆性由Vandermonde行列式保证。

**KEY：**

$$ P(x) := a_{0} + a_{1}x + \cdots + a_{N-1}x^{N-1} $$

则
$$ c_k = \sum_{j=0}^{N-1} a_{j} w^{jk} = P(w^{k}), \quad k=0,\cdots,N-1 $$

**快速Fourier变换**

设 $ w_{N} = e^{-\frac{2\pi i}{N}} $ 为N次单位根。

求：
$
\vec{c} = \widehat{\vec{a}}
$,
$
c_k = \sum_{j=0}^{N-1} a_{j}w_{N}^{jk} = P(w_{N}^{k}) $

即求多项式 $ P(x) $ 在 $ w_{N}^{0},w_{N}^{1},\cdots,w_{N}^{N-1} $ 上的值。

**KEY:**（假设$ N=2^{m} $）

将多项式分解：
$$ P(x) = P_e(x^2) + xP_o(x^2) $$

计算：
$$
\begin{cases}
c_j = P(w_{N}^{j}) = P_e(w_{N/2}^{j}) + w_{N}^{j}P_o(w_{N/2}^{j}) \\
c_{N/2+j} = P_e(w_{N/2}^{j}) - w_{N}^{j}P_o(w_{N/2}^{j})
\end{cases}
$$

其中 $ j=0,1,\cdots,\frac{N}{2}-1 $

**注：** 此处由 $ w_{N}^{\frac{N}{2}+j} = -w_{N}^{j} $ 导出了 $ c_{N/2+j} $ 的表达式，从而将每一层的规模保持在 $ \frac{N}{2^i} * 2^i = N$（如第一层为2个 $ \frac{N}{2} $的计算量）
