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

### Parallel Programming with Threads, OpenMP

- PThreads(一种threading library)

	<img src="https://i.imgs.ovh/2025/08/16/K8Yw1.png"  width="400" />
- OpenMP

	<img src="https://i.imgs.ovh/2025/08/16/K84Ln.png"  width="600" />
	
	<img src="https://i.imgs.ovh/2025/08/16/K8K8r.png"  width="600" />

- False Sharing是多线程/多核编程中一种常见的性能问题，其核心原因是：现代CPU的缓存以缓存行（通常64字节）​​为单位加载数据。当两个线程访问​同一缓存行内的不同变量​时，即使它们逻辑上无关，也会因​缓存一致性协议（如MESI）​​引发不必要的同步竞争，从而导致并行程序性能显著下降。
- Padding, critical, for reduction(原理上也许是Padding) 是一些解决False Sharing的方法
- 并行区域(#pragma omp parallel)内的变量为private变量
- 静态调度与动态调度 Static Schedule & Dynamic Schedule

	<img src="https://i.imgs.ovh/2025/08/18/peiG4.png"  width="500" />
	
	<img src="https://i.imgs.ovh/2025/08/18/pstYX.png"  width="500" />

- Data Sharing: Changing storage attributes

	<img src="https://i.imgs.ovh/2025/08/18/p9D9c.png"  width="500" />
	
- ```#pragma omp tast```(配合```#pragma omp single```使用）

	<img src="https://i.imgs.ovh/2025/08/18/pDXJg.png"  width="200" />
	
### Sources of Parallelism and Locality in Simulation

- Basic Kinds of Simulation
	- Discrete event systems 离散空间 离散时间
	- Particle systems 离散粒子系统(离散空间) 连续时间
	- Lumped variables depending on continuous parameters 离散**物体**系统 连续时间 ODEs
	- Continuous variables depending on continuous parameters 连续空间 连续时间 PDEs

- Discrete Event Systems
	- 系统有Finite set of variables
	- state: 特定时间所有变量的取值
	-  transition function: 每个变量的更新是一个关于其他变量的转移函数
	-  系统可以是同步(sybchronous)的，也叫state machine，意味着每一个离散时间点都有明确的转移函数来更新
	-  系统也可以是异步(asynchronous)的，也叫event driven simulation，意味着每一个状态是否转移到下一个状态取决于**输入是否发生变化(如相邻的格子有新物体出现)**
	-  Domain Decomposition - 例: 矩形域分为9宫格并行计算，最小化通信成本(仅在边界处)
	
		<img src="https://i.imgs.ovh/2025/08/18/pKjoa.png"  width="400" />
	
	- Graph Partitioning 图划分问题, 可以算作一种特殊的Domain Decomposition
	
		<img src="https://i.imgs.ovh/2025/08/18/pK2W1.png"  width="400" />
	
	- Asynchronizing的两种方式: conservative & Speculative(Optimistic)
	
		<img src="https://i.imgs.ovh/2025/08/18/pKKia.png" width="370" />
		
	- 解决conservative的deadlock的方法: Are you stuck too?
	
		<img src="https://i.imgs.ovh/2025/08/18/pKx9A.png" width="375" />
		
- Particle Systems
	- a finite number of particles
	- 主要有三种力, 整个场存在的external force, 临近粒子之间的nearby force, 还有任何两两粒子之间的far-field force

		<img src="https://i.imgs.ovh/2025/08/18/pmpub.png" width="400" />
	- external force只有每个粒子单独相关, "embarrasingly parallel"
	- nearby force的计算需要考虑通信，因此需要domain decomposition

		<img src="https://i.imgs.ovh/2025/08/18/pu2ne.png" width="400" />
	- 考虑到load balance(粒子可能分布不均匀), 可能需要dynamic decompostion

		<img src="https://i.imgs.ovh/2025/08/18/puIbU.png" width="400" />
	
	- far-field force避免$O(n^2)$的方法：Particle Mesh方法——将所有粒子近似到regular mesh上计算，便可以利用FFT or Multigrid，复杂度为$O(nlogn)$或$O(n)$

		<img src="https://i.imgs.ovh/2025/08/18/pu3E6.md.png" width="400" />
		
	- far-field force的第二种方法是Tree Decomposition，同样复杂度为$O(nlogn)$或$O(n)$
	- far-field force的方法还有Barnes-Hut, Fast multipole method (FMM) by Greengard/Rohklin, Anderson s method.

- Lumped Variable Systems
	- Systems of "lumped" variables
	- Each depends on continuous parameter (usually time)
	
		<img src="https://i.imgs.ovh/2025/08/19/Ij1uC.png" width="400" />

		<img src="https://i.imgs.ovh/2025/08/19/IcHs4.png" width="400" />
		
	- 解ODE的方法1：显式方法，归结为sparse-matrix-vector mult.
	- 解ODE的方法2：隐式方法，归结为solve linear systems
	- 求解linear systems的直接方法：Gauss消元，LU分解，考虑dense和sparse两种情况
	- 求解linear systems的迭代方法：Jacobi, SOR, Conjugate Gradient, Multigrid...
	- Lumped Systems还有Eigenproblems(地震共振等），同样分为dense和sparse两种情况，归结为sparse-matrix-vector multiplication, direct methods

- Parallel Sparse Matrix-vector multiplication, 储存系数矩阵的格式-CSR, Compressed Sparse Row, 时间大部分花费在数组嵌套索引上
	
	<img src="https://i.imgs.ovh/2025/08/19/If8Fm.png" width="500" />
	
	- 将稀疏矩阵A分解为多个区域对应不同的processor，是一种Graph partitioning问题
	
	<img src="https://i.imgs.ovh/2025/08/19/Ifl5x.png" width="400" />
	
	<img src="https://i.imgs.ovh/2025/08/19/IfhHe.png" width="400" />
	
- 各大算法应用热力图

	<img src="https://i.imgs.ovh/2025/08/19/IfuSt.png" width="350" />
	
- Partial Differential Equations
	- **Elliptic problems**(steady state, global space dependence)
		
		Electrostatic, Gravitational Potential - Potential(position)
	- **Hyperbolic problems**(time dependent, local space dependence)

		Sound waves - Pressure(position, time)
	- **Parabolic problems**(time dependent, global space dependence)
		
		Heat flow - Temperature(position, time)
		
		Diffusion - Concentration(position, time)
		
	- Many problems combine features of above(Fluid flow, Elasticity)
	
	- Example: Heat Equation
		
		- 显式方法

			<img src="https://i.imgs.ovh/2025/08/20/I4oQp.png" width="400" />
		
		- 三(五)对角矩阵的load balance和minimize communication很好做

			<img src="https://i.imgs.ovh/2025/08/20/I4led.png" width="400"/>
		
		- 显式方法计算简捷, 但数值稳定性差, 时间步长有限制(足够小)

		- 隐式方法

			<img src="https://i.imgs.ovh/2025/08/20/I4tKg.png" width="400" />
			
		- 2D情况

			<img src="https://i.imgs.ovh/2025/08/20/I4pO0.png" width="400" />
			
		- 高维情况各算法时间复杂度

			<img src="https://i.imgs.ovh/2025/08/20/I4IgY.md.png" width="400" />
			
		- Application procedure: Converting mesh to a matrix, Reordering, Multigrid

			<img src="https://i.imgs.ovh/2025/08/20/IE6q0.png" width="400" />
			
			<img src="https://i.imgs.ovh/2025/08/20/IE84Y.png" width="400" />
			
			<img src="https://i.imgs.ovh/2025/08/20/IEeib.png" width="250" />
			
- N-body problem, Matmul的优化(最小化communication)

	<img src="https://i.imgs.ovh/2025/08/20/IEnIr.md.png" width="300" />
	
	<img src="https://i.imgs.ovh/2025/08/20/IElB4.png" width="270" />
	
	<img src="https://i.imgs.ovh/2025/08/20/IEw29.md.png" width="350" />
	
### Basics of GPUs

- GPU中的Branch语句串行化问题
	
	<img src="https://i.imgs.ovh/2025/08/20/ItMB9.png" width="400" />
	
- Synchronization, atomic operations

	- Threads within a block may synchronize with barriers
	- ```… Step 1 … __syncthreads(); … Step 2 …```
	- Blocks coordinate via atomic memory operations
		- e.g., increment shared pointer with atomicInc()
		- Or use cooperative thread groups

- Blocks must be independent (threads in a block can synchronize...)	
- Memory Coalescing (内存合并访问)
	- Successive 4W bytes ( W: warp size, 4: size of single word in
bytes) memory can be accessed by a warp (W consecutive
threads) in a single transaction.

	<img src="https://i.imgs.ovh/2025/08/20/IuU6n.png" width="400" />

### Data Parallel Algorithms

- Memory Operation
	- Unary Operators: A = array, B = f(A)
	- Binary Operators: A, B = array, C = A + B
	- Broadcast
	- Strided and Scatter, Gather
	- Masks
		
		<img src="https://i.imgs.ovh/2025/08/20/Iw19e.png" width="400" />
	- Reduce (Sum, Max)
	- Scan (+, Max) 前缀和 前缀最大值
	- Scan的两种变体，Inclusive scan, Exclusive scan

		<img src="https://i.imgs.ovh/2025/08/20/IXvLU.png" width="400" />
		
- Idealized Hardware and Performance Model
	- Machine in **Ideal Cost Model** for Data Parallelism
		- An **unbounded** number of processors
		- Control overhead is free
		- Communication is free
	- Cost on this abstract machine is the algorithm's **span or depth**, $T_{\infty}$.
		- Defines a lower bound on time on real machines
		- For uniary or binary operations: $O(1)$
		- For reductions and broadcasts: $O(logn)$, can be proved to be the lower bound of ideal cost if only use arbitrarily many **binary operations** machine (By binary tree)
		- For Matmul (n by n): $O(logn)$

			<img src="https://i.imgs.ovh/2025/08/20/IwaeY.md.png" width="350"/>
			
		- For scan (不止+, max, 任何支持结合律的binary operation都可以): $O(logn)$, Magic!

			<img src="https://i.imgs.ovh/2025/08/20/Iwfy1.png" width="450"/>
			
- Non Trivial Applications of Data Parallelism Using Scans

| 计算任务                                      | 时间复杂度       | 应用场景/相关人物                          |
|-----------------------------------------------|------------------|--------------------------------------------|
| Adding two n-bit integers                     | O(log n)         | -                                          |
| Inverting n-by-n triangular matrices          | O(log² n)        | -                                          |
| Inverting n-by-n dense matrices               | O(log² n)        | -                                          |
| Evaluating arbitrary expressions              | O(log n)         | -                                          |
| Evaluating recurrences                        | O(log n)         | -                                          |
| 2D parallel prefix                            | -                | 图像分割 (Catanzaro & Keutzer)             |
| Sparse-Matrix-Vector-Multiply (SpMV)          | -                | 使用分段扫描 (Segmented Scan)              |
| Parallel page layout                          | -                | 浏览器布局 (Leo Meyerovich, Ras Bodik)     |
| Solving n-by-n tridiagonal matrices           | O(log n)         | -                                          |
| Traversing linked lists                       | -                | -                                          |
| Computing minimal spanning trees              | -                | -                                          |
| Computing convex hulls of point sets          | -                | -                                          |

- Applications
	- Stream Compression

		<img src="https://i.imgs.ovh/2025/08/20/hZS5H.png" width="400" />
	- Radix Sort (基数排序) 

		<img src="https://i.imgs.ovh/2025/08/20/hZLdU.md.png" width="400"/>
		
		<img src="https://i.imgs.ovh/2025/08/20/hZU0X.png" width="400"/> 
		
	- List Ranking with Pointer Doubling

		<img src="https://i.imgs.ovh/2025/08/20/hZVzQ.png" width="400"/>
		
	- Fibonacci via Matrix Multiply Prefix

	- Adding n-bit integers in $O(logn)$ time (思想和Fibonacci一样, 递推式转为矩阵形式)

		<img src="https://i.imgs.ovh/2025/08/20/hZRS9.png" width="400"/>
		
		<img src="https://i.imgs.ovh/2025/08/20/hZzXm.png" width="400"/>
		
	- Lexical analysis (tokenizing, scanning)

	- Inverting triangular n-by-n matrices in $O(log_2 n)$ time

		<img src="https://i.imgs.ovh/2025/08/20/hZkhp.png" width="400"/>
		
	- Inverting Dense n-by-n matrices in $O(log_2 n)$ time, Completely numerically unstable
	-  Segment Scans

		<img src="https://i.imgs.ovh/2025/08/20/hZ5j6.png" width="400"/>
		
	- Parallel prefix cost on p “big” processors
		
		<img src="https://i.imgs.ovh/2025/08/20/hZ66O.png" width="400"/>
		
### Distributed Memory Machines and Programming

- Network Anaology
	- Distributed Memory Machines里面的数据交换不像Shared Memory，"Not just a bus"
	- Networks are like streets:
		- **Link** = street
		- **Switch** = intersection
		- **Distances(hops)** = number of blocks traveled
		- **Routing algorithm** = travel plan
	- Design Characteristics of a Network
		- **Topology**: crossbar, ring, 2-D, 3-D, higher-D mesh or torus, hypercube, tree, butterfyl, perfect shuffle, dragon fly
		- **Routing algorithm**: avoids deadlock
		- **Switching strategy**: Circuit switching, Packet switching
		- **Flow control**: deal with congestion, Using Stall, store data temporarily in buffers...
		- **Latency**: Vendors often report hardware latencies (wire time), Application programmers care about software latencies (user program to user program)
		- **Overhead**(开销)​​ 指消息从发送方到接收方过程中，由软件或硬件引入的额外处理延迟，是系统延迟的主要来源。其核心特征如下：
			- 主导性​: 软件/硬件开销（10–100微秒）远超硬件传输延迟（每跳10–100纳秒），尤其在频繁小消息场景中成为关键瓶颈。
			- 组成:
				- ​软件开销​: 用户程序与通信协议栈的交互（如数据序列化、内存拷贝、上下文切换）。
				- ​硬件开销: 网卡处理、中断响应等底层操作。
			- ​跨架构差异​: 不同网络设计的开销可能相差1–2个数量级，是优化延迟的核心突破口。
		- **Diameter**: the maximum (over all pairs of nodes) of the shortest path between a given pair of nodes
		- **Bisection bandwidth**: bandwidth across smallest cut that divides network into two equal halves
			
			<img src="https://i.imgs.ovh/2025/08/22/hmxo6.png" width="400" />
			
	- Different Designs of Network
		- **Linear array**: Diameter = $n-1$, Bisection bandwidth = $1$; 
		- **Torus of Ring**: Diameter = $n/2$, Bisection bandwidth = $2$;
		- **2-D mesh**: Diameter = $2*(\sqrt{n}-1)$, Bisection bandwidth = $\sqrt{n}$;
		- **2-D torus**: Diameter = $\sqrt{n}$, Bisection bandwidth = $2*\sqrt{n}$;
		- **Hypercubes**: Diameter = $d$, Bisection bandwidth = $n/2$ (Number of nodes $n=2^d$ for dim = $d$;
			
			<img src="https://i.imgs.ovh/2025/08/22/huSst.png" width="400" />
		- **Trees**: Diameter = $logn$, Bisection bandwidth = $1$, Easy layout as planar graph
		- **Fat Trees** can avoid bisection bandwidth problem (cost more money(HW) but more efficiency)

			<img src="https://i.imgs.ovh/2025/08/22/huEqn.png" width="400" />
		- **Butterfles**:
		
			<img src="https://i.imgs.ovh/2025/08/22/hudEL.md.png" width="400" />
		
		- **Dragonfly**: 近距离用电缆远距离用光缆

			<img src="https://i.imgs.ovh/2025/08/22/hx2AA.png" width="400" />
	
	- Randomized Routing: Minimal routing works well when things are load balanced, but potentially
catastrophic in adversarial traffic patterns.
		
		<img src="https://i.imgs.ovh/2025/08/22/hxJwH.md.png" width="400"/>
		
- Programming Distributed Memory Machines with Message Passing

	- Novel Features of MPI
		1. Communicators (通信器): 封装独立的通信空间 (如 ```MPI_COMM_WORLD```是默认通信器), 隔离不同库或模块的通信域，避免消息冲突
			- 支持子组划分 (```MPI_Comm_split```)，实现逻辑上的进程分组
			- 确保线程安全 (如多线程中每个线程使用独立通信器)
		2. Datatypes (数据类型)​​：定义复杂数据结构（如结构体、非连续内存块），减少数据拷贝开销，支持异构系统通信
			- 示例：使用 ```MPI_Type_create_struct``` 描述混合数据类型的消息
			- 直接发送非连续数据（如矩阵子块），避免手动打包
		3. Multiple Communication Modes (多种通信模式)​​：提供不同语义的通信模式，精确控制缓冲区管理
			- 标准模式​（```MPI_Send```）：由 MPI 管理缓冲区，可能阻塞
			- ​同步模式​（```MPI_Ssend```）：接收方就绪后才完成发送
			- ​就绪模式​（```MPI_Rsend```）：假设接收方已就绪（需用户保证安全性）
			- ​非阻塞模式​（```MPI_Isend```）：异步通信，重叠计算与通信
			- ```MPI_Bsend```: Supply own space as buffer for send
		4. ​Extensive Collective Operations (集合操作)​​：优化全局通信（如广播、规约、散射/聚集）
			- ```MPI_Bcast```：向所有进程广播数据
			- ```MPI_Reduce```：并行规约（如求和、最大值）
			- ```MPI_Alltoall```：全交换（用于矩阵转置等场景）
		5. Process Topologies (进程拓扑)​​：将逻辑进程映射到物理硬件拓扑（如网格、环）
			- 虚拟拓扑（```MPI_Cart_create``` 创建笛卡尔网格）
			- 邻居发现（```MPI_Cart_shift``` 获取相邻进程）
			- 提升数据局部性，使通信密集的进程在物理上靠近
		6. ​Profiling Interface (分析接口)​​：通过包装函数（如 ```PMPI_``` 前缀）标准化性能分析
			- 支持第三方工具（如 Vampir、Intel Trace Analyzer）
			- 低开销记录通信耗时、消息大小等指标 
		7. 还有```MPI_Request, MPI_Test, MPI_Wait, MPI_Waitall```等语句，相当于```barrier, synchronize```
	- A Simple Example

	```
#include "mpi.h"
#include <stdio.h>
int main(int argc, char *argv[])
{
    int rank, buf;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Process 0 sends and Process 1 receives */
    if (rank == 0) {
        buf = 123456;
        MPI_Send(&buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        MPI_Recv(&buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Received %d\n", buf);
    }

    MPI_Finalize();
    return 0;
}
	```

	- tag 是 ```MPI_RECV``` 和 ```MPI_SEND``` 函数中的一个关键参数，用于区分不同类型的消息。它的作用类似于邮件系统中的“邮件标签”或“主题分类”，帮助接收方精确识别和处理特定消息。例如：进程 A 发送两个消息，进程 B 可选择接收特定标签的消息。
	
	```
MPI_Send(data1, count, MPI_INT, dest, 0, MPI_COMM_WORLD);  // 标签=0（数据）
MPI_Send(data2, count, MPI_INT, dest, 1, MPI_COMM_WORLD);  // 标签=1（控制指令）
MPI_Recv(buffer, count, MPI_INT, source, 1, MPI_COMM_WORLD, &status); // 只接收标签=1的消息
	```
	
	- 利用广播机制的例子	
	
	```	
#include "mpi.h"
#include <math.h>
#include <stdio.h>
int main(int argc, char *argv[]) {
    int done = 0, n, myid, numprocs, i;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    while (!done) {
        // 进程0读取用户输入
        if (myid == 0) {
            printf("Enter the number of intervals: (0 quits) ");
            scanf("%d", &n);  // 修正OCR识别错误
        }

        // 广播区间数n到所有进程
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 终止条件
        if (n == 0) break;

        // ============== 并行计算核心 ============== //
        h = 1.0 / (double) n;       // 计算步长
        sum = 0.0;                   // 初始化局部和

        // 各进程计算分配到的区间 (数据并行)
        for (i = myid + 1; i <= n; i += numprocs) {
            x = h * ((double)i - 0.5);              // 中点坐标
            sum += 4.0 / (1.0 + x * x);             // 修正函数：f(x)=4/(1+x²)
        }
        mypi = h * sum;                             // 当前进程的局部积分结果

        // 汇总所有进程结果到进程0 (全局求和)
        MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        // ========================================== //

        // 进程0输出最终结果
        if (myid == 0) {
            printf("pi is approximately %.16f, Error is %.16f\n",
                   pi, fabs(pi - PI25DT));  // 修正格式说明符
        }
    }

    MPI_Finalize();
    return 0;
}
	```	
	
- 截至目前所有MPI语法汇总
	- ```#include "mpi.h"``` MPI库
	- ```MPI_Comm_size(MPI_COMM_WORLD, &size)``` 获取processors数量，储存在size中
	- ```MPI_Comm_rank(MPI_COMM_WORLD, &rank)``` 获取当前processor id，储存在rank中
	- ```MPI_Init(&argc, &argv)``` 初始化MPI
	- ```MPI_Finalize()``` 结束MPI
	- ```MPI_COMM_WORLD``` 默认通信器
	- ```mpirun -np 4 a.out``` 编译命令
	- ```MPI_INT, MPI_DOUBLE...``` MPI中的类型
	- ```MPI_ANY_TAG``` MPI通讯过程中会携带一个tag信息，类似于邮件的“主题”或“分类标签”，帮助接收方区分不同类型的消息。接收方可以指定具体 Tag，只接收匹配该标签的消息。```MPI_ANY_TAG```为通用类型
	- ```MPI_Send(start, count, datatype, dest, tag, comm)``` 发送数据
	- ```MPI_Recv(start, count, datatype, source, tag, comm, status)``` 接受数据, 需要同时匹配source和tag才会接受，接受的count要大于等于发送的count
	- ```MPI_ANY_SOURCE``` 可以匹配任何source processor
	- ```MPI_Status status``` 接受方可以通过```MPI_Status```类型的```status```参数得知消息来源source、标签tag、数据量count等关键信息，前两者可以使用语法```status.MPI_TAG, status.MPI_SOURCE```，数据量获取使用语法```MPI_Get_count(&status, datatype, &recvd_count)```
	- ```MPI_Bcast(start, count, datatype, tag, commm)``` 广播数据
	- ```MPI_Reduce(start, dest, count, datatype, reduce_op, root, comm)``` 规约操作
	- ```MPI_Sendrecv``` 同时发送&接受，避免send和receive操作可能导致的deadlock
	- ```MPI_Isend(start, count, datatype, dest, tag, comm, &request)``` Non-blocking operation 非阻塞操作，request参数(```MPI_Request```类型)记录数据发送是否完成
	- ```MPI_Irecv(start, count, datatype, dest, tag, comm, &request)``` Non-blocking operation 非阻塞操作，request参数记录数据接受是否完成
	- ```MPI_Wait(&request, &status)``` 该函数是一个阻塞操作，表示等待直到request对应的操作完成
	- ```MPI_Test(&request, &flag, &status)``` 该函数是一个非阻塞操作，用于检查request对应的操作是否完成，结果将保存在bool类型的flag中
	- ```MPI_Waitall(count, array_of_requests, array_of_statuses)```
	- ```MPI_Waitany(count, array_of_requests, &index, &status)```
	- ```MPI_Waitsome(count, array_of_requests, array_of indices, array_of_statuses)```
	- ```MPI_Ssend``` Synchronous mode: the send does not complete until a matching receive has begun. (Unsafe programs deadlock.)
	- ```MPI_Bsend``` Buffered mode: the user supplies a buffer to the system for its use. (User allocates enough memory to make an unsafe program safe.)
	- ```MPI_Rsend``` Ready mode: user guarantees that a matching receive has been posted.
		- Allows access to fast protocols
		- undefined behavior if matching receive not posted
	- ```MPI_Issend``` Non-blocking versions

### Advanced MPI and Collective Communication Algorithms

- Collective Data Movement
	- Broadcast
	- Scatter
	- Gather
	- Allgather
	- Alltoall
	- Reduce
	- Scan
	- `All` versions deliver results to all participating processes, not just root.
	- `V` versions allow the chunks to have variable sizes.
	- Many Routines: Allgather, Allgatherv, Allreduce, Alltoall, Alltoallv, Bcast, Gather, Gatherv, Reduce, Reduce_scatter, Scan, Scatter, Scatterv

- MPI Built-in Collective Computation Operations
	- `MPI_MAX` Maximum
	- `MPI_MIN` Minimum
	- `MPI_PROD` Product
	- `MPI_SUM` Sum
	- `MPI_LAND` Logical and
	- `MPI_LOR` Logical or
	- `MPI_LXOR` Logical exclusive or
	- `MPI_BAND` Binary and
	- `MPI_BOR` Binary or
	- `MPI_BXOR` Binary exclusive or
	- `MPI_MAXLOC` Maximum and location
	- `MPI_MINLOC` Minimum and location

- SUMMA, Scalable Universal Matrix Multiply Algorithm (Used in PBLAS, Parallel BLAS)

	- Formula
		$$
		C(I,J) = C(I,J)+\sum_kA(I,K)*B(K,J)
		$$
	
	- Naive version

		<img src="https://i.imgs.ovh/2025/08/24/G1cIh.png" width="400" />
	
	- `MPI_Ibcast` 优化版本, 并行通信与本地计算, 同时用`MPI_Comm_split`将通信器分为row communicator和column communicator, 代码见Lecture 10
	
- `Allgather`内部算法
	- Ring Algorithm: $T_{ring} = \alpha (p-1) + \beta n (p-1)/p$
	- Recursive Doubling Algorithm: $T_{rec-dbl} = \alpha lg(p) + \beta n (p-1)/p$

		<img src="https://i.imgs.ovh/2025/08/24/G1Xw4.png" width="300"/>
	- Bruck Algorithm: $T_brock = \alpha \lceil{lg(p)}\rceil + \beta n (p-1)/p$，$p$较大时比Recursive Doubling稍快，因为Recursive的复杂度实际上是$2\alpha \lfloor{lg(p)}\rfloor + \beta n (p-1)/p$

		<img src="https://i.imgs.ovh/2025/08/24/G11LA.png" width="500"/>

- Hybrid Programming with Threads

	- 三种模式: All MPI, MPI + OpenMP, MPI + Pthreads
	- MPI's Four Levels of Thread Safety
		- `MPI_THREAD_SINGLE`: 每个processor仅有一个线程，就是原始的MPI
		- `MPI_THREAD_FUNNELED`: 每个processor多线程，只有main thread进行MPI
		- `MPI_THREAD_SERIALIZED`: 每个processor多线程，同一时间只有一个thread进行MPI
		- `MPI_THREAD_MULTIPLE`: 每个processor多线程，任何thread都可以进行MPI，需要注意race problem
	- Hybriding时，初始化由`MPI_Init`变为`MPI_Init_thread（requested, provided)`
	- 注意Ordering和Blocking问题

		<img src="https://i.imgs.ovh/2025/08/24/G3zJ1.png" width="400"/>

- MPI RMA (one-sided communication)

	<img src="https://i.imgs.ovh/2025/08/24/G3gTL.png" width="500"/>
	
	- 为什么需要one-sided communication

		<img src="https://i.imgs.ovh/2025/08/24/G3qmM.png" width="400" />

	- RMA首先需要创建window
		- `MPI_WIN_CREATE` already have an allocated buffer that you would like to make remotely accessible
		- `MPI_WIN_ALLOCATE` create a buffer and directly make it remotely accessible
		- `MPI_WIN_CREATE_DYNAMIC` don't have a buffer yet, but will have one in the future, want to dynamically add/remove buffers to/from the window
		- `MPI_WIN_ALLOCATE_SHARED` want multiple processes on the same node share a buffer	
	- MPI provides ability to read, write and atomically modify date in remotely accessible memory regions
		- `MPI_PUT`
		- `MPI_GET`
		- `MPI_ACCUMULATE`
		- `MPI_GET_ACCUMULATE`
		- `MPI_COMPARE_AND_SWAP`
		- `MPI_FETCH_AND_OP`

### UPC++: Partitioned Global Address Space Languages



### Dense Linear Algebra - History and Structure, Parallel Matrix Multiplication

- Dense Linear Algebra包含的内容
	- Linear Systems $Ax=b$
	- Least Squares: choose $x$ to minimize $||Ax-b||^2$, 有各种变式 Overdetermined, underdetermined, unconstrained, constrained, weighted(ridge)
	- Eigenvalues and vectors of Symmetric Matrices, 有Standard$Ax=\lambda x$, Generalized$Ax=\lambda Bx$
	- Eigenvalues and vectors of Unsymmetric Matrices, 计算Eigenvalues, Schur form, eigenvectors, invariant subspaces, Standard, Generalized
	- Singular Values and vetors (SVD), 有Standard, Generalized
	- Different matrix structures - 28 types in LAPACK, 包括Real, complex, symmetric, Hermitian, positive definite, dense, triangular, banded
	- Level of detail
		- Simple Driver "$x=A\backslash b$"
		- Eppert Drivers with error bounds, extra-precision, other options
		- Lower level routines (apply certain orthogonal transformation, matmul)
	- Randomized Versions

- History of (Dense) Linear Algebra Software
	- do-loop, EISPACK
	- BLAS-1
		- Standard library of 15 operations (mostly) on vectors
		- Operations like AXPY
		- do $O(n^1)$ ops on $O(n^1)$ data
		- using libraries like LINPACK
	- BLAS-2
		- Standard library of 25 operations (mostly) on matrix/vector pairs
		- Operations like GEMV: $y=\alpha A x + \beta y$, GER: $A=A+\alpha x y^T$, $x=T^{-1}x$
		- do $O(n^2)$ ops on $O(n^2)$ data
		- **CI** still ~2
	- BLAS-3
		- Standard library of 9 operations (mostly) on matrix/matrix pairs
		- Operations like GEMM: $C=\alpha A B + \beta C$, $C=\alpha A A^T + \beta C$, $B=T^{-1}B$
		- do $O(n^3)$ ops on $O(n^2)$ data
		- **CI** now ~$n/2$, good!
	- LAPACK (1989 - now), Linear Algebra PACKage
	- ScaLAPACK (1995 - now), Scalable LAPACK
		- For distributed memory using MPI
		- More complex data structures, algorithm than LAPACK
	- PLASMA, DLASMA, MAGMA (now)
		- Ongoing extensions to Multicore/GPU/Heterogeneous
	- Other related projects like SLATE, Elemental, FLAME

- Matmul Communication Lower Bounds
	- Single Processor
		- Assume $n^3$ algorithm (no Strassen-like)
		- Sequential Case (M = fast memory size) **Lower bound on #words moved** = $\Omega(n^3 / M^{1/2})$
		- Attained using blocked or cache-oblivious algorithms
	- P Processors
		- M = fast memory per processor, assume load balance
		- **Lower bound on #words moved** = $\Omega((n^3 / p) / M^{1/2})$
		- 假设所有processor加到一起正好是所有内存($3n^2$)刚好储存, then $M=3n^2/p$, lower bound  = $\Omega (n^2 / p^{1/2})$
		- Attained by SUMMA, Cannon's algorithm
		- 此时需要考虑messages sent，理论最优情况下，每次sent message都是$M$大小
		- **#words\_moved per processor** = $\Omega (\#flops / M^{1/2})$
		- **#messages\_sent per processor** = $\Omega (\#flops / M^{3/2})$
	- 总结如下
		<img src="https://i.imgs.ovh/2025/08/24/mjHbd.png" width=550"/>
	- Goals for algorithms
		- Minimize #words moved
		- Minimize #messages sent
		- Minimize for multiple memory hierarchy levels
		- Fewest flops when matrix fits in fastest memory

- Parallel Data Layouts for Matrices
	 2D Row and Column Block Cyclic Layout是理论上最好用的
	
	<img src="https://i.imgs.ovh/2025/08/24/mjJ2g.md.png" width="500"/>
	
- Parallel Matrix Multiply
	- 目标: Computing $C=C+AB$
	- 使用Basic Algorithm: $2n^3$ Flops (no Strassen like)
	- 考虑的因素
		- Data Layout: 1D, 2D, Others
		- Topology of machine: Ring, Torus, 2DTorus, Others
		- Scheduling communication
	- Performance model: Message time = "latency" + #words * time per word = $\alpha + n \beta$
	- 衡量指标之一, Efficiency = serial time / (p * parallel time), perfect (linear) speedup are **efficiency = 1**
	- Communicates once at a time, 基本上就是serial computing

		<img src="https://i.imgs.ovh/2025/08/24/mjh0M.png" width = "350" />
		
	- Pairs of adjacent processors can communicate simultaneously. (Nearly) Optimal for 1D layout on Ring or Bus, even with Broadcast! 

		<img src="https://i.imgs.ovh/2025/08/24/mjGzr.png" width = "350"/>
		
		- $\text{Time of inner loop} = 2 \times \left(\alpha + \beta \times n^2/p\right) + 2 \times n \times \left(n/p\right)^2$
		- $\text{Total Time} = 2 \times n \times \left(n/p\right)^2 + (p-1) \times \text{Time of inner loop} = 2n^3/p + 2p\alpha + 2\beta n^2$
		- $\text{Parallel Efficiency} = 2*n^3 / (p*\text{Total Time}) = 1 / (1 + O(p / n))$
		- Grows to $1$ as $n/p$ increases, but still far from communication lower bound

- SUMMA, hits the lower bound on Matmul!

	<img src="https://i.imgs.ovh/2025/08/24/mjBSa.png" width="500"/>

- 如果多用一些内存，而非$M=n^2/p$，可以达到更好的效率，最好可以达到$c=P^{1/3}$ (理论上限). 但现实中没有这么多内存空间，通常是单独确定$c$，此时方法称为**2.5D SUMMA** 

	<img src="https://i.imgs.ovh/2025/08/24/mjXDt.png" width="500"/>
	
	
### Dense Linear Algebra - Parallel Gaussian Elimination and QR


- MatMul的回顾
	- Sequential communication goals
		- \#words moved = $O(n^3/M^{1/2})$
		- \#messages = $O(n^3/M^{3/2})$
	- Parallel communication goals, with minimum memory $n^2/P$
		- \#words moved = $O(n^2/P^{1/2})$
		- \#messages = $O(P^{1/2})$
	- Parallel communication goals, with c倍的minimum memory
		- \#words moved = $O(n^2/(cP)^{1/2})$
		- \#messages = $O(P^{1/2}/c^{3/2})$ for MatMul, $O((cP)^{1/2})$ for LU and QR

- Sequential: Gauss Elimination Refinement
	- Initial Version

	```
	V1:
	for i = 1 to n - 1
		for j = i + 1 to n
			m = A(j, i) / A(i, i)
			for k = i + 1 to n
				A(j, k) = A(j, k) - m * A(i, k)
	```
		
	<img src="https://i.imgs.ovh/2025/08/25/mlqXY.png" width="500" />
	
	- 不妨将m储存在A(j, i)中，将循环拆开 (目标是将循环尽可能改写为Matmul的形式)

	```
	V2:
	for i = 1 to n - 1
		for j = i + 1 to n
			A(j, i) = A(j, i) / A(i, i)
		for j = i + 1 to n
			for k = i + 1 to n
				A(j, k) = A(j, k) - A(j, i) * A(i, k)
	V3:
	for i = 1 to n - 1
		A(i + 1 : n, i) = A(i + 1 : n, i) * (1 / A(i, i)) #BLAS-1 op, scale a vector
		A(i + 1 : n, i + 1 : n) = A(i + 1 : n, i + 1 : n) - A(i + 1 : n, i) * A(i, i + 1 : n) #BLAS-2, rank-1 update
	```
	
	<img src="https://i.imgs.ovh/2025/08/25/mlJSt.png" width="350" />
	
	- We call the stricly lower triangular matrix of multipliers $M$, let $L=I+M$, call the upper triangle of the final matrix $U$. That is $A=LU$ factorization.
	- Next we consider numerical stability. It's needed to **pivot**. That is $A=PLU$ factorization.
	- But pivot的过程是非常耗时的, Big idea: **Delayed Updates**
		- Save updates to "trailing matrix" from several consecutive BLAS2 (rank-1) updates.
		- Apply many updates simultaneously in one BLAS3 (matmul) operation
		- Need to choose a **block size b** such that small enough (active submatrix consisting of b columns of A fits in cache) and large enough (make BLAS3 matmul fast)
	
	```
	V4: LAPACK version
	for ib = 1 to n - 1 step b
		end = ib + b - 1
		apply BLAS-2 version of GEPP to get A(ib : n, ib : end) = PLU
		Let LL denote the strict lower triangular part of A(ib : end, ib : end) + I
		A(ib : end, end + 1 : n) *= LL^-1
		A(end + 1 : n, end + 1 : n) -=  A(end + 1 : n, ib : end) * A(end + 1 : n, end + 1 : n)
	```
	
	<img src="https://i.imgs.ovh/2025/08/25/mtQec.png" width="300" />
	
	- Gauss Elimination 和 Matmul的效率等价性

		<img src="https://i.imgs.ovh/2025/08/25/mKcxF.png" width="300"/>
	
	- LAPACK的GEMM实际上没有Minimize Communication

		<img src="https://i.imgs.ovh/2025/08/25/mKHIC.png" width="450"/>
		
	- Alternative cache-oblivious GE formulation (Recursive)

		<img src="https://i.imgs.ovh/2025/08/25/mpkHU.png" width="450"/>	
- Parallel Gauss Elimination

	- Distributed Gaussian Elimination with a 2D Block Cyclic Layout
	- 这种方法叫做PDGESV (Parallel Distributed Gauss Elimination Solve) = ScaLAPACK Parallel LU

		<img src="https://i.imgs.ovh/2025/08/25/mp9j9.png" width="450"/>
		
		<img src="https://i.imgs.ovh/2025/08/25/mpQ76.png" width="450" />
		
	- ScaLAPACK效率
		- \#words sent = $O(n^2logP/P^{1/2}$
		- \#messages sent = $O(nlogP)$ (与$n$相关，很大)
	
	- Next Goal: For each panel of b columns spread over $P^{1/2}$ procs, identify b "good" pivot rows in one reduction
	- 这种factorization叫做 TSLU = "Tall Skinny LU"

		<img src="https://i.imgs.ovh/2025/08/25/mIftc.png" width="400" />
		
	
- Parallel QR, TSQR, 和TSLU的思路一样

	<img src="https://i.imgs.ovh/2025/08/25/mIFR6.png" width="400"/>
	
- Multicore: Expressing Parallelism as a DAG (Directed Acyclic Graph)

	<img src="https://i.imgs.ovh/2025/08/25/mI9Qe.png" width="400" />

	<img src="https://i.imgs.ovh/2025/08/25/mIDua.png" width="400"/>

### Machine Learning Part 1 (Supervised  Learning)



### Machine Learning Part 2 (Unsupervised and semi-supervised learning)



### Ray: A universal framework for distributed computing




### Structured Grids

<img src="https://i.imgs.ovh/2025/08/26/uxyKQ.png" width="400"/>

- Jacobi Method
	- Poisson Equation says: u(i, j) = (u(i-1, j) + u(i+1, j) + u(i, j-1) + u(i, j+1) + b(i, j)) / 4
	- 记u(i, j, m)为第m次迭代的解
	- u(i, j, m+1) = (u(i-1, j, m) + u(i+1, j, m) + u(i, j-1, m) + u(i, j+1, m) + b(i, j)) / 4 即为Jacobi迭代
	- Steps to converge proportional to problem size $N=n^2$
	- Therefore, serial complexity is $O(N^2)$
	- Jacobi每次取相邻点更新，所以有**有限传播速度**
		
		<img src="https://i.imgs.ovh/2025/08/26/uxcNc.png" width="400"/>
		
	- Parallize Jacobi 就是利用了有限传播速度，take k>>1 iterations for the communication cost of 1 iteration

		<img src="https://i.imgs.ovh/2025/08/26/uxrUp.png" width="400"/>
	
	- Do a little bit redundant arithmetic, but saves quite communications
	- 2D Case: 

		<img src="https://i.imgs.ovh/2025/08/26/uxvW6.md.png" width="400"/>
		
- Red-black Gauss-Seidel
	
	- converges twice as fast as Jacobi, but there are twice as many parallel steps, so the same in practice
	<img src="https://i.imgs.ovh/2025/08/26/uxTsg.png" width="400" />
	
- SOR, Successive overrelaxalation
	- Idea: The basic step in algorithm as: u(i, j, m+1) = u(i, j, m) + correction(i, j, m)
	- If correction is a good direction, then one should move further than 1: u(i, j, m+1) = u(i, j, m) + w * correction(i, j, m)
	- We can prove $w = \frac{2}{1+sin(\frac{\pi}{n+1})}$ for best convergence for Poisson
	- Number of steps to converge = parallel complexity = $O(n)$, instead of $O(n^2)$ for Jacobi
	- Serial complexity $O(n^3)=O(N^{3/2})$, instead of $O(n^4)=O(N^2)$ for Jacobi

- CG, Conjugate Gradient
	
	<img src="https://i.imgs.ovh/2025/08/26/uxqWx.png" width="500"/>
	
- Multigrid
	- Motivation
		- Recall that Jacobi, SOR, CG, or any other sparse-matrix-vector-multiply-based algorithm can only move information one grid cell at a time (Take at least n steps to move information across n x n grid)
		- Therefore, converging in O(1) steps requires moving information across grid faster than to one neighboring grid cell per step (One step can’t just do sparse-matrix-vector-multiply)
	- Algorithm
		- Replace problem on fine grid by an approximation on a coarser grid
		- Solve the coarse grid problem approximately, and use the solution as a starting guess for the fine-grid problem, which is then iteratively updated
		- Solve the coarse grid problem recursively, i.e. by using a still coarser grid approximation, etc.
	- Success depends on **coarse grid solution being a good approximation to the fine grid**!
	- Multigrid Sketch in 2D

		<img src="https://i.imgs.ovh/2025/08/26/ux8vr.md.png" width="400" />
		
	- Multigrid V-Cycle Algorithm

		<img src="https://i.imgs.ovh/2025/08/26/uxesh.md.png" width="400" />
		
		- 重要理解: Each level in a V-Cycle reduces the error in one part of the frequency domain

			<img src="https://i.imgs.ovh/2025/08/26/uxspe.png" width="400" />
			
			- Convergence Picture of Multigrid in 1D

			<img src="https://i.imgs.ovh/2025/08/26/uxnV4.png" width="400"/>
			
		- The restriction operator $R$ can do sampling or averaging

			<img src="https://i.imgs.ovh/2025/08/26/uxP4q.md.png" width="400" />
			
		
	- Weighted Jacobi: able to damp high frequency error
		- At level i, pure Jacobi says: x(j) = 1/2 * (x(j-1) + x(j+1) + b(j))
		- But Weighted Jacobi says: x(j) = 1/3 * (x(j-1) + x(j) + x(j+1) + b(j))

	- Parallel in Multigrid

		<img src="https://i.imgs.ovh/2025/08/26/ux2nA.md.png" width="400" />
		
		
|          | # Flops               | # Messages     | # Words sent            |
|----------|-----------------------|----------------|-------------------------|
| Multigrid       | $N/p + log p * log N$   | $(log N)^2$      | $(N/p)^{1/2} + log p * log N$ |
| FFT      | $N log N / p$           | $p^{1/2}$        | $N/p$                     |
| SOR      | $N^{3/2} / p$           | $N^{1/2}$        | $N/p$                     |

- SOR is slower than others on all counts
- Flops for MG depends on accuracy of MG
- MG communicates less total data (bandwidth)
- Total messages (latency) depends …

### Sparse Matrix-Vector Multiply (SpMV) and Iterative Solvers

- Recall CSR (Compressed Sparse Row) Storage: It's the main structure we use
	
	<img src="https://i.imgs.ovh/2025/08/27/xqcn0.png" width="400"/>
	
	Matrix-vector multiply kernel: y(i) = y(i) + A(i, j) * x(j)
	
	```
	for each row i
		for k = ptr[i] to ptr[i+1] do
			y[i] = y[i] + val[k] * x[ind[k]]
	```
	
	- CSS, Compressed Sparse Column
	- COO, Coordinate: row + column index per nonzero
	- DIAG, Diagonal: store main diagonal as 1D array or diagonal bands as 2D (padded)
	- Symmetric: store 1/2
	- Blocked: store each block contiguously
		- Register blocked: blocks a small and dense, avoid indexed within blocks
		- Cache blocked: blocks are large and themselves sparse

- Parallel and distributed SpMV

	- 计算Mat-Vec-Mul，直接计算，OpenMP并行计算, 使用dynamic避免稀疏矩阵分布不均匀

	```
	#pragma omp parallel num_threads(thread_num)
	{
	#pragma omp for private(j, i, tmp) schedule(dynamic)
		for(int i = 0;i < m;i++){
			for(j = ptr[i];j < ptr[i+1]; j++){
				tmp = ind[j];
				y[i] += val[j] * x[tmp];
			}
		}
	}
	```
	
	- 计算Mat-Vec-Mul，直接计算，CUDA并行

	```
	__global__ void spmv(int *ptr, int *ind, float *val, int m, float *x, float *y){
		for(int i = blockIdx.x * blockDim.x + threadIdx.x;i < m;i += blockDim.x * gridDim.x){
			float yi = 0;
			for(int j = ptr[i];j < ptr[i+1];j++){
				yi += values[j] * x[col_ind[j]];
			}
			y[i] = yi;
		}
	}
	```
	
	- 计算Mat-Vec-Mul，直接计算，结合Segmented Suffix Scan思想

		<img src="https://i.imgs.ovh/2025/09/01/w59Zb.png" width="400" />
		
	- 计算Mat-Vec-Mul，矩阵是对角格式 (columnoffset数组存储矩阵的所有对角线，0, +1, -1, ....)

	```
	for each diagonal k do
		#pragma omp parallel for
		for each row i do
			column = i + columnoffset[k]
			if(column >= 0 && column < n)
				y[i] = y[i] + val[k][column] * x[column];
	```
	
	- Ideal Sparse Structure: P diagonal Block
		- If $p_i$ holds $x_i$ and $y_i$ blocks, no vectors to communicate
		- If no non-zeros outside these blocks, no communication needed
			<img src="https://i.imgs.ovh/2025/09/01/w6DhX.png" width="400"/>
			
	- HPCG Benchmark

- Register / cache blocking and autotuning SpMV

	- Libraraies for sparse matrices
		- OSKI = Optimized Sparse Kernel Interface, pOSKI for multicore
		- BeBOP: Berkeley Benchmarking and Optimization Group
	- Automatic Register Block Size Selection
		- Selecting the $r*c$ block size
		- Step1: Off-line benchmark of "register profile"
			- Precompute Mflops(r, c) using **dense A in sparse format (blocked sparse row)** for each $r*c$
			- Once per machine / architecture
		- Step 2: Run-time "search"
			- Sample $A$ to estimate Fill(r, c) for each $r*c$ (填充零的个数占$r*c$的比例)
		- Step 3: Run-time heuristic model
			- Choose $r, c$ to minimize time ~ Fill(r, c) / Mflops(r, c)
		- Step 2中Sample的方法
			- Fraction of matrix to sample: $s \in [0,1]$, Cost ~ $O(s *nnz)$
			- Control cost by controlling $s$, control $s$ by automatically by computing statistical confidence intervals like Monitor variance (观察方差是否已经足够小)
			- Cost of tuning
				- Lower bound: convert matrix in 5 to 40 unblocked SpMVs (所以这种格式实际上适用于需要多次稀疏矩阵乘法的情况)
				- Heuristic: 1 to 11 SpMVs
	- Cache Blocking
		- For CSR, dot-product, re-use opportunity is only in the x vector
		- Matrices that are have some locality and "well ordered" e.g., near diagonal have good re-use
		- Cache blocking can help for "short wide" matrices both on serial and SMPs

	- Summary of Sequential Performance Optimizations

		<img src="https://i.imgs.ovh/2025/09/01/wsbhd.png" width="400" />
		
- CA iterative solvers, Sparse matmul (SpGEMM, SPMM, ...)

	<img src="https://i.imgs.ovh/2025/09/01/wQZkh.png" width="400"/>
	
	<img src="https://i.imgs.ovh/2025/09/01/wQjia.png" width="400"/>
	
	<img src="https://i.imgs.ovh/2025/09/01/wQrnq.png" width="400"/>
	
	<img src="https://i.imgs.ovh/2025/09/01/wQSv4.png" width="400"/>
	
	<img src="https://i.imgs.ovh/2025/09/01/wQL9A.png" width="400"/>
	

### Parallel Spectral Methods: Fast Fourier Transform (FFT) with Applications

- DFT简介
	- 一维DFT - $m$元素向量$v$的一维DFT定义为：
		$$F \cdot v$$
	其中$F$是$m \times m$矩阵，其元素定义为：		$$F[j,k] = \omega^{j \cdot k}, \quad 0 \leq j, k \leq m-1$$
	这里$\omega$是复数：
		$$\omega = e^{2\pi i / m} = \cos\left(\frac{2\pi}{m}\right) + i \cdot \sin\left(\frac{2\pi}{m}\right)$$
	$\omega$是$m$次单位根，满足$\omega^m = 1$。

	- 二维DFT - $m \times m$矩阵$V$的二维DFT定义为：
		$$F \cdot V \cdot F$$
	理解：1. 对所有列独立进行一维DFT 2. 对所有行独立进行一维DFT

	- 高维DFT的计算方法与二维类似，通过逐维应用一维DFT实现。

- Poisson Equation
	- 1D Poisson equation: solve $L_1x=b$ where (3 point stencil)

		$$
		L_1 = \left( \begin{array}{ccccc} 
2 & -1 &   &   &   \\ 
-1 &  2 & -1 &   &   \\ 
   & -1 &  2 & -1 &   \\ 
   &    & -1 &  2 & -1 \\ 
   &    &    & -1 &  2 \\ 
\end{array} \right)
		$$
	- 2D Poisson equation: solve $L_2x=b$ where (5 point stencil)

		$$
		L_2 = \begin{bmatrix}
4 & -1 & 0 & -1 & 0 & 0 & 0 & 0 & 0 \\
-1 & 4 & -1 & 0 & -1 & 0 & 0 & 0 & 0 \\
0 & -1 & 4 & 0 & 0 & -1 & 0 & 0 & 0 \\
-1 & 0 & 0 & 4 & -1 & 0 & -1 & 0 & 0 \\
0 & -1 & 0 & -1 & 4 & -1 & 0 & -1 & 0 \\
0 & 0 & -1 & 0 & -1 & 4 & 0 & 0 & -1 \\
0 & 0 & 0 & -1 & 0 & 0 & 4 & -1 & 0 \\
0 & 0 & 0 & 0 & -1 & 0 & -1 & 4 & -1 \\
0 & 0 & 0 & 0 & 0 & -1 & 0 & -1 & 4
\end{bmatrix}
		$$
		
	- 2D Poisson equation也可以看作 solve $L_1X+XL_1=B$ (5 point stencil, $X$为二维网格点构成的矩阵)

	
	- $ L_1 = F \cdot D \cdot F^T $ is eigen decomposition where
		$$ F(j,k) = \left( \frac{2}{n+1} \right)^{1/2} \cdot \sin\left( \frac{j k \pi}{n+1} \right) $$
		
		$$ D(j,j) = 2 \left( 1 - \cos\left( \frac{j \pi}{n+1} \right) \right) $$
		
	- 对 $L_1X+XL_1=B$ 等式两侧做2维DFT，得到
		$$ D(F^TXF)+(F^TXF)D = F^TBF $$
		- Step 1: 计算$B'=F^TBF$
		- Step 2: 计算$DX'+X'D=B'$, $X'(j,k) = B'(j,k) / \left(D(j, j) + D(k, k)\right)$
		- Step 3: 计算$X=FX'F^T$

	- For solving the Poisson equation and various other applications, we use variations on the FFT
		- The sin transform -- imaginary part of F
		- The cos transform -- real part of F
	- Algorithms are similar, so we will focus on F

- 一维FFT具体方法
	- FFT same as evaluating a polynomial V(x) with degree m-1 at m different points. The call tree of the D&C (Divide and Conquer) FFT algorithm is a complete binary tree of log m levels.

		<img src="https://i.imgs.ovh/2025/08/26/uRe6N.png" width="500" />
		
		<img src="https://i.imgs.ovh/2025/08/26/uRMAX.png" width="750"/>
	
	- 利用中间的Transpose步骤避免processors之间的communication

		<img src="https://i.imgs.ovh/2025/08/26/uRo86.png" width="400"/>
		
		If no communication is pipelined, Time(transposeFFT) = $2nlog(m)/p + (p-1)\alpha + m(p-1)/p^2\beta$
		If communication is pipelined, so $(p-1)\alpha$ turns out to be single $\alpha$
	
	- (Sequential Communication Complexity) FFT of size m, **\#words moved** between main memory and cache of size M (m>M):
		- Thm (Hong, Kung, 1981) \#words = $\Omega(mlogm/logM)$
		- Attained by Transpose algorithm (Sequental algorithm "simulates" parallel algorithm)
		- Attrained by recursive, "cache-oblivious" algorithm (FFTW)

	- (Parallel Communication Complexity) FFT of size m, **\#words moved** between p processors
		- Thm (Aggarwal, Chandra, Snir, 1990) \#words = $\Omega(mlogm/\left(plogm/p\right))$
		- Attained by Transpose algorithm
			- Recall assumption $log(m/p)\geq log(p)$
			- So $2\geq logm / log(m/p) \geq 1$
			- So \#words = $\Omega(m/p)$

- 高维FFT
	- FFTs on 2 or more dimensions are defined as 1D FFTs on vectors in all dimensions.
		- 2D FFT does 1D FFTs on all rows and then all columns

	- There are 3 obvious possibilities for the 2D FFT:
		1. 2D blocked layout for matrix, using parallel 1D FFTs for each row and column
		2. Block row layout for matrix, using serial 1D FFTs on rows, followed by a transpose, then more serial 1D FFTs
		3. Block row layout for matrix, using serial 1D FFTs on rows, followed by parallel 1D FFTs on columns
	- Option 2 is best, if we overlap communication and computation

	- Modified LogGP Model

		<img src="https://i.imgs.ovh/2025/08/26/u8MR6.png" width="400"/>
		
	- GASNet Communications System – Used by UPC

		<img src="https://i.imgs.ovh/2025/08/26/u84tn.png" width="400"/>
		
	- 3D FFT

		<img src="https://i.imgs.ovh/2025/08/26/u8hQe.md.png" width="400"/>
		
		<img src="https://i.imgs.ovh/2025/08/26/u8xeq.md.png" width="400"/>
	- Three different ways to break up the messages
		1. Packed Slabs (i.e., single packed "All to all" in MPI)
		2. Slabs (2D)
		3. Pencils (1D)
	- Slabs and Pencils allow overlapping communication and computation and leverage RMDA support in modern networks

- FFTW (the "Fastest Fourier Transform in the West")

	- C library for real & complex FFTs (arbitrary size/dim), also offer parallel versions for threads & MPI
	- Computational kernels (80% of code) automatically generated
	- Self-optimized for hardware = portability + performance
	- FFTW implements many FFT algorithms: A **planner** picks the best composition by measuring the speed of different combinations.
	- The resulting plan is executed with explicit recursion: enhance locality
	- The base cases of the recursion are **codelete** highly-optimized dense code automatically generated by a special-purpose "compiler"
		
		
### Graph Partitioning

- Definition of Graph Partitioning

	- Given a graph $G=(N,E,W_N,W_E)$, $N$ = nodes, $E$ = edges, $W_N,W_E$ = weights of nodes/edges

		<img src="https://i.imgs.ovh/2025/09/10/7C16qY.png" width="250"/>
		
	- 目标: Choose a partition $N = N_1 \cup N_2 \cup \cdots \cup N_p$ such that
		- The sum of the node weights in each $N_j$ is "about the same" (load balance)
		- The sum of all adge weights of edges connecting all different pairs $N_j$ and $N_k$ is minimized (minimize communication)
	- Special case: $N=N_1 \cup N_2$ = Graph Bisection

	- Application: 
		- Telephone network design
		- Load Balancing while Minimizing Communication
		- SpMV
		- VLSI (超大规模集成电路) Layout
		- Sparse Gaussian Elimination
		- Data mining and clustering
		- Physical Mapping of DNA
		- Image Segmentation

	- Choosing optimal paritioning is NP-hard!
	- We need good heuristics (启发式算法)

- First Heuristics: Repeated Graph Bisection
	
	- To partition $N$ into $2^k$ parts (bisect graph recursively $k$ times)

	- Edge Separators ($E_s$, 一些边的集合，将图分为两个差不多大小的子图), Vertex Seperators ($N_s$, 一些点的集合，同理)
		- Making an $N_s$ from an $E_s$: pick one endpoint of each edge ($|N_s|<=|E_s|$)
		- Making an $E_s$ from an $N_s$: pick all edges incident on $N_s$ ($|E_s|<=d * |N_s|$, $d$ = maximum degree)

	- **A planar graph** = can be drawn in plane without edge crossings

		- **Theorem** (Tarjan, Lipton, 1979): If graph $G$ is planar, there exist $N_s$ such that
			- $N=N_1\cup N_s \cup N_2$ is a partition
			- $|N_1| <= 2/3|N|$ and $|N_2| <= 2/3 |N|$
			- $|N_s <= (8*|N|)^{1/2}|$

			
- Partitioning without Nodal Coordinates (Coordinate-Free), **Kernighan/Lin**

	- Take a initial partition and iteratively improve it
		- Kernighan/Lin (1970), cost = $O(|N|^3)$ but easy to understand
		- Fiduccia/Mattheyses (1982), cost = $O(|E|)$, much better, but more complicated
	
	- Given G = (N, E, W_E) and a partitioning N = A U B, |A| = |B|
		- T = cost(A, B) = Σ {W(e) where e connects nodes in A and B}
		- Find subsets X of A and Y of B with |X| = |Y|
		- Consider swapping X and Y if it decreases cost:
    		- newA = (A - X) U Y and newB = (B - Y) U X
    		- newT = cost(newA, newB) < T = cost(A, B)
	
	- Need to compute newT efficiently for many possible X and Y, choose smallest (一种贪心，每次并非选择两个大小差不多的子集，而是两个点)
		- Repeat: choose x in A - X and y in B - Y s.t. swapping x,y lowers cost the most
		- Choose best sets of swapped x and y in sequence

	- Fiduccia-Mattheyses: only need to look at adjacent nodes of each node, cost = $O(|E|)$! (见Slide)

- Second Coordinate-Free Method: Spectral Bisection
	- Motivation: analogy to a vibrating string

		<img src="https://i.imgs.ovh/2025/09/10/7C11Xp.png" width="300"/>
		
	- Definition of **Laplacian matrix L(G)** of the graph $G(N,E)$ is an $|N|$ by $|N|$ symmetric matrix, defined by
		- $L(G)_{i,i} = \text{degree of node } i \text{ (number of incident edges)}$
		- $L(G)_{i,j} = -1 \quad \text{if } i \neq j \text{ and there is an edge } (i,j)$
		- $L(G)_{i,j} = 0 \quad \text{otherwise}$

	- Properties of Laplacian Matrix
		
		- $L(G)$ is symmetric (Eigenvalues are real and eigenvectors are real and orthogonal!)
		- The eigenvalue of $L(G)$ are nonnegative: $0=\lambda_1<=\lambda_2<=\cdots<=\lambda_n$
		- **# of connected components of G is equal to the number of **$\lambda_i$ **equal to 0**
		- **Definition**: $\lambda_2(L(G))$ is the **algebraic connectivity** of $G$
			- $\lambda_2$的大小衡量了connectivity
			- $\lambda_2$不等于0当且仅当$G$是连通的 (由上一条等式得出)

	- Spectral Bisection Algorithm的流程
		- Compute eigenvector $v_2$ corresponding to $\lambda_2(L(G))$
		- For each node $n$ of $G$
			- If $v_2(n) < 0$, put node $n$ in partition $N_{-}$
			- else put in partition $N_{+}$
	
	- Implementation via the **Lanczos Algorithm**
		- To optimize SpMV, we grapg partition
		- To graph partition, we find an eigenvector of a matrix associated with the graph
		- To find an eigenvector, we do SpMV
		- Have we made progress? **The first matrix-vector multiplies are slow, but use them to learn how to make the rest faster**
	- 虽然计算特征向量需要做稀疏矩阵乘法，但我们并不需要把这个乘法优化到极致之后才能开始。我们可以：
		- 接受初始性能：​​ 首先，我们使用未优化的、相对较慢的稀疏矩阵-向量乘法来运行Lanczos算法，以计算出我们所需的特征向量。
		- 打破循环：​​ 一旦我们得到了特征向量，我们就可以完成高质量的图划分。
		- ​实现最终目标：​​ 利用这个好的图划分，我们去优化稀疏矩阵-向量乘法本身。此后，所有后续的计算（包括未来可能需要进行的更多特征值计算）都能从这个优化中获益，变得非常快。
​
- Multilevel Acceleration (Multigrid的思想)
​
​	<img src="https://i.imgs.ovh/2025/09/10/7CiTit.png" width="400"/>
​	
	- 方法1: **Multilevel Kernighan-Lin**
	- **Coarsen** graph and **expand** parition using **maximal matchings**
	- **Improve** partition using **Kernighan-Lin**

	- **Definition**: A **matching** of a graph $G(N,E)$ is a subset $E_m$ of $E$ such that no two edges in $E_m$ share an endpoint
	- **Definition**: A **maximal matching** is a matching $E_m$ to which no more edges can be added and remain a matching

		<img src="https://i.imgs.ovh/2025/09/10/7CikIH.png" width="400"/>
		
		<img src="https://i.imgs.ovh/2025/09/10/7CiYd0.png" width="400"/>
		
	- 方法2: **Multilevel Spectral Bisection**
	- **Coarsen** graph and **expand** parition using **maximal independent sets**
	- **Improve** partition using **Rayleigh Quotient Iteration**

	- **Definition:** An **independent set** of a graph $G(N, E)$ is a subset $N_i$ of $N$ such that no two nodes in $N_i$ are connected by an edge.
	- **Definition:** A **maximal independent set (MIS)** of a graph $G(N, E)$ is an independent set $N_i$ to which no more nodes can be added and remain an independent set.

		<img src="https://i.imgs.ovh/2025/09/10/7CipDx.png" width="400"/>
		
	- **寻找最大独立集 (MIS)**：在当前图 $G_{i}$ 中找到一个最大独立集 (MIS) $M$。独立集意味着 $M$ 中的任意两个节点都没有边直接相连。**构建粗化图**：
		- **超节点形成**：将 $M$ 中的每个节点都作为 $G_{i+1}$ 中的一个超节点。
		- **节点吸收**：对于不在 $M$ 中的每个节点 $v$，将其与 $M$ 中与其相邻的一个节点合并 (“吸收” 或 “折叠”)。通常选择边权重最大的邻居，或者随机选择一个。
		- **边权重更新**：$G_{i+1}$ 中两个超节点之间的边权重，是 $G_{i}$ 中连接这两个超节点所代表的所有原始节点簇之间的所有边权重之和。
​
## UCB名词解释

- Threads (线程) & Process (进程)- SRAM: Static Random-Access Memory（静态随机存取存储器）包括L1, L2, L3 cache等- DRAM: Dynamic Random-Access Memory（动态随机存取存储器）包括主内存、显存等- Cashe hit & Cashe miss
	当CPU或计算单元请求的数据已经存在于缓存（Cache）中时，称为缓存命中（反之为miss）- Memory Benchmark	
	内存基准测试，是指通过标准化测试程序或工具，评估计算机内存（DRAM、Cache、HBM等）的性能指标，包括：	- 带宽（Bandwidth）：单位时间内可读写的数据量（GB/s）	- 延迟（Latency）：从发起请求到获取数据的时间（纳秒级）。	- 吞吐量（Throughput）：系统在单位时间内能完成的内存操作次数。- HBM: High Bandwidth Memory, 高宽带内存- ILP: Instruction Level Parallelism
- Pipelining
- SIMD: Single Instruction Multiple 所有处理单元在同一时刻执行相同指令（如CPU向量指令）
- SPMD: Single Program Multiple Data 处理器可能通过条件分支处理不同逻辑（如if (threadID == 0) {...}）
- FMA: Fused Multiply Add
- CI: Computational Intensity, CI = f/m: average number of flops per slow memory access
- Machine Balance: tm/tf, slow memory access time/fast arithmetic operation time- BLAS: Basic Linear Algebra Subroutines- NUMA: Non-Uniform Memory Access
- SW prefetch: Software prefetching
- POSIX: Portable Operating System Interface可移植操作系统接口
- PThreads: The POSIX threading interface, support for
	- Creating parallelism
	- Synchronizing
	- Implicit shared memory- SpGEMM: Sparse General Matrix-Matrix Multiplication，稀疏通用矩阵乘法
- PRAM (Parallel Random Access Machine，并行随机存取机器): 理论计算机科学中用于研究并行算法的一种抽象计算模型。它假设存在无限数量的处理器、共享内存以及无通信延迟的理想化并行环境，是分析并行算法时间复杂度和效率的基础工具。
- SMEM: Shared Memory
- MPI: Massage Passing Interface
- SPMD: Single Program Multiple Data (MPI的编译原理)
- Radix: 在 ​Distributed Memory Machine​​中，​Network Topology​​的 ​Radix​（基数）是指 ​每个交换节点（Switch Node）支持的直连链路数量，也称为 ​交换节点的端口数（Port Count）​。它直接决定了网络的连接能力和扩展性。 
- RDMA: Remote Data Memory Access 远程数据内存访问, 是MPI的One-sided communication的方式
- GAS: Global Address Space, 全局地址空间
- NIC: Network Interface Card, 用于one-sided communication
- GEPP: Gauss Elimination with Partial Pivoting
- OSKI, Optimized Sparse Kernel Interface 是一个为科学计算应用程序自动优化稀疏矩阵运算的库。它的核心思想是：通过让库在运行时自动“学习”矩阵的特性，来为特定的矩阵和机器架构选择并执行最高效的计算内核（Kernel），从而显著提升稀疏矩阵运算（如稀疏矩阵-向量乘法 SpMV）的性能。


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
em