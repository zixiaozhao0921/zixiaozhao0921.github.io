此处放我的笔记

## CUDA

- GPU性能指标：核心数、显存容量、计算峰值、显存带宽
- CUDA提供两层API接口，CUDA driver API和CUDA runtime API
- GPU状态查询和配置工具：`NVIDIA-smi`工具
- 编译CUDA文件的命令：`nvcc hello.cn -o hello`(后`./hello.cn`)
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
- GPU线程模型结构：grid(网格) - block(线程块) - thread(线程)
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
- `__device__, __global__, __host__`
- Debug: `cudaError_t类型, cudaGetErrorName函数(返回字符串), cudaGetErrorString函数(返回字符串)`
- Debug检查函数

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
        return error_code;
    }
    return error_code;
}
```
- Kernel函数只能为void类型，检测方法为

```
ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);
ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
```


## MFEM软件笔记


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
