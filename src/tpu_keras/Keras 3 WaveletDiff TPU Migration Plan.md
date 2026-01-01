# **Strategic Migration and Optimization of WaveletDiff: A Deep Dive into Keras 3, JAX, and TPU v6e Architecture**

## **1\. Executive Summary and Strategic Imperatives**

The domain of time series generation has historically grappled with a fundamental trade-off between temporal coherence and spectral fidelity. While autoregressive models excel at local dependencies and Generative Adversarial Networks (GANs) offer speed, Denoising Diffusion Probabilistic Models (DDPMs) have emerged as the current state-of-the-art for their ability to model complex distributions with high fidelity. However, standard time-domain diffusion models often fail to capture the multi-scale characteristics inherent in real-world temporal data—such as the coexistence of long-term trends and high-frequency transients in financial or seismic signals. WaveletDiff, a novel framework leveraging multilevel wavelet diffusion, addresses this by operating directly in the wavelet domain, applying diffusion processes independently to approximation and detail coefficients.1

This report provides a comprehensive, exhaustive technical roadmap for porting the WaveletDiff architecture from its native PyTorch implementation to the Keras 3 ecosystem, specifically utilizing the JAX backend to exploit the massive parallelization capabilities of Google’s Cloud TPU v6e (Trillium) hardware. The transition to Keras 3 with JAX is not merely a syntactic translation; it represents a paradigmatic shift from dynamic, eager execution to static, compiled graph execution via XLA (Accelerated Linear Algebra). This migration necessitates a fundamental re-architecting of data flow, particularly regarding the Discrete Wavelet Transform (DWT), the handling of variable-length sequences in cross-attention mechanisms, and the optimization of the iterative denoising loop using JAX primitives like lax.scan.3

The target infrastructure, TPU v6e, introduces specific architectural constraints—such as 256x256 Matrix Multiply Units (MXUs) and high-bandwidth memory (HBM) hierarchies—that dictate strict tensor layout and padding strategies to achieve peak FLOP utilization.5 By aligning the mathematical elegance of wavelet decomposition with the computational brute force of JAX on Trillium, this migration aims to achieve order-of-magnitude improvements in training throughput and inference latency, transforming WaveletDiff from a research prototype into an industrial-grade generative engine.

## **2\. Theoretical Deconstruction of the WaveletDiff Architecture**

To effectively re-engineer WaveletDiff for a static graph environment, one must first rigorously deconstruct its mathematical and structural components. The model fundamentally diverges from standard diffusion models by lifting the diffusion process into the wavelet domain, a strategy that requires precise handling of signal processing operations within the neural network graph.

### **2.1 The Multilevel Wavelet Diffusion Mechanism**

Standard diffusion models operate on the raw signal $x\_0$, adding Gaussian noise $\\epsilon$ over $T$ steps to produce $x\_T \\sim \\mathcal{N}(0, I)$. WaveletDiff, conversely, first decomposes the time series $x$ into a set of coefficients using a Discrete Wavelet Transform (DWT). For a decomposition level $L$, the signal is represented as $\\{A\_L, D\_L, D\_{L-1}, \\dots, D\_1\\}$, where $A\_L$ represents the approximation coefficients (low-frequency trend) and $D\_l$ represents the detail coefficients (high-frequency fluctuations) at level $l$.1

The core innovation of WaveletDiff lies in running forward diffusion processes for each wavelet level individually and in parallel. This design allows the model to learn specific noise schedules for different frequency bands. Empirically, high-frequency detail coefficients ($D\_l$) often resemble Gaussian noise more closely than the structured low-frequency approximation ($A\_L$). Consequently, the optimal noise schedule $\\beta\_t^{(l)}$ differs across levels. The approximation coefficients typically require a more aggressive noise schedule to destroy the strong temporal correlations, whereas the detail coefficients, being sparser and more stochastic, require a subtler noise injection.8

Mathematically, the forward process for a specific level $l$ is defined as:

$$q(y\_t^{(l)} | y\_0^{(l)}) \= \\mathcal{N}(y\_t^{(l)}; \\sqrt{\\bar{\\alpha}\_t^{(l)}} y\_0^{(l)}, (1 \- \\bar{\\alpha}\_t^{(l)}) \\mathbf{I})$$

where $y^{(l)}$ denotes the coefficients at level $l$, and $\\bar{\\alpha}\_t^{(l)}$ is the cumulative noise variance schedule specific to that level. The reverse process—the denoising network—attempts to predict the noise $\\epsilon\_\\theta^{(l)}(y\_t^{(l)}, t, \\mathbf{c})$ added to each level, conditioned on time $t$ and potentially global context $\\mathbf{c}$.1

### **2.2 Hierarchical Data Flow and Dimensionality**

This separation of scales imposes significant architectural complexity. In a standard decimation-by-2 DWT (e.g., using Daubechies wavelets), the spatial resolution halves at each decomposition level. If the input signal has length $N$, level 1 details $D\_1$ have length $N/2$, level 2 details $D\_2$ have length $N/4$, and so on.

In the reference PyTorch implementation, this is typically handled using dynamic tensor lists or dictionaries, where each key corresponds to a level.9 PyTorch's eager execution model handles these variable shapes seamlessly. However, in the JAX/TPU paradigm, such dynamic structures disrupt the XLA compiler’s ability to fuse operations. If the compiler cannot infer static shapes, it may trigger recompilation for every batch, rendering training prohibitively slow. The migration strategy must therefore involve a unified, static tensor representation—either through "ragged" processing simulation via masking or rigorous padding strategies that treat the multi-scale coefficients as a single composite tensor structure.11

### **2.3 Cross-Level Attention and Spectral Integrity**

While the diffusion processes are independent regarding noise addition, the denoising step requires information exchange between levels to preserve the hierarchical structure of the data. A detail coefficient at position $k$ in level $l$ is semantically related to approximation coefficients at position $k/2$ in level $l+1$. WaveletDiff employs dedicated level-transformers that process each sub-band, coupled with a cross-level attention mechanism. This allows the approximation levels (containing global structure) to gate or influence the generation of detail levels (containing local texture).1

Furthermore, the implementation must enforce spectral fidelity via Parseval’s theorem. The theorem states that the energy of the signal in the time domain is equal to the sum of the energies of its wavelet coefficients.

$$\\sum\_{n} |x\[n\]|^2 \= \\sum\_{k} |A\_L\[k\]|^2 \+ \\sum\_{l=1}^{L} \\sum\_{k} |D\_l\[k\]|^2$$

In WaveletDiff, this conservation is critical. The loss function is often weighted to respect this energy distribution, ensuring that errors in high-energy low-frequency bands do not overwhelm the subtle high-frequency details. Implementing this "Wavelet-aware loss" in JAX requires careful vectorization of the loss computation across the decomposition levels, ensuring that the weighting factors $\\lambda\_l$ are broadcast correctly against the varied shapes of the coefficients.2

## **3\. The Keras 3 and JAX Computational Paradigm**

Keras 3 introduces a unified API that decouples model definition from the execution backend, allowing the same high-level code to run on JAX, TensorFlow, or PyTorch.14 For this project, the JAX backend is selected not just for compatibility, but to leverage its functional purity and superior performance on TPU hardware.

### **3.1 Statelessness and Functional Purity**

The most significant friction point in migrating from PyTorch to JAX via Keras 3 is the handling of state. PyTorch models are stateful object-oriented structures where layers "own" their weights. JAX, in contrast, is functionally pure; functions effectively take parameters as input and return new parameters.15

Keras 3 bridges this gap by managing state explicitly within the Model and Layer classes while dispatching stateless calls to the JAX backend. However, when implementing custom layers for WaveletDiff—specifically the DWT and the specialized transformers—one must adhere to stateless design principles. The randomness required for diffusion sampling (Gaussian noise injection) cannot be drawn from a global state. Instead, jax.random.PRNGKeys must be explicitly threaded through the call methods. This is a critical divergence from PyTorch's torch.randn, requiring the WaveletDiff implementation to manage a hierarchy of random keys, splitting them for each diffusion step and each wavelet level to ensure deterministic and statistically sound noise generation.16

### **3.2 The Role of XLA and Static Graphs**

The primary driver for using JAX on TPU is XLA (Accelerated Linear Algebra). XLA compiles the entire computational graph into optimized machine code for the TPU. This compilation relies heavily on static shapes—the dimensions of all tensors must be known and fixed at compile time.

WaveletDiff’s hierarchical nature poses a challenge here. If the input time series length varies (e.g., a dataset containing both 1-second and 10-second audio clips), a naive JAX implementation would trigger a recompilation for every unique length. This "cache thrashing" is catastrophic for performance.4 To mitigate this, the implementation must standardize input sequence lengths using padding/bucketing strategies. For a time series of length $T$, the JAX implementation should define a discrete set of bucket sizes (e.g., 256, 512, 1024\) and pad inputs to the nearest bucket. The wavelet decomposition will then produce static shapes for each level (e.g., $1024 \\to 512, 256, 128$), allowing XLA to generate a single, highly optimized kernel for each bucket size.

### **3.3 Backend-Agnostic Operations vs. JAX Primitives**

Keras 3 provides the keras.ops namespace (e.g., keras.ops.matmul, keras.ops.conv), which dispatches to the active backend. While convenient, porting the DWT might require operations that are not fully exposed or optimized in the high-level API, such as strided convolutions with specific boundary handling (reflection padding).

In these instances, the implementation will bypass keras.ops and utilize jax.numpy or jax.lax directly. Keras 3 supports this hybrid approach, but it locks those specific layers to the JAX backend. Given the explicit goal of TPU v6 optimization, this backend-coupling is acceptable and necessary to access low-level primitives like lax.conv\_general\_dilated which maps directly to the TPU’s matrix multiply units.14

### **Table 1: Conceptual Mapping from PyTorch to Keras 3/JAX**

| Feature | PyTorch (Native) | Keras 3 (JAX Backend) | TPU Implementation Detail |
| :---- | :---- | :---- | :---- |
| **State Management** | self.weight (Mutable) | Functional update (Immutable) | Weights stored in HBM, updated via functional optimization steps. |
| **Randomness** | Global torch.manual\_seed | Explicit PRNGKey threading | Keys split using jax.random.split for parallel generation. |
| **Control Flow** | Python for loops | jax.lax.scan / jax.lax.cond | Unrolled loops explode binary size; scan is mandatory for diffusion. |
| **Padding** | Dynamic/Eager | Static/Bucketed | Requires fixed shapes to avoid XLA recompilation. |
| **Data Format** | NCL (Channels First) | NHWC (Channels Last) preferred | Transposition required for optimal vectorization on TPU. |

## **4\. Implementation Strategy: Wavelet Transforms in JAX**

The core of WaveletDiff is the Discrete Wavelet Transform. Since standard PyTorch wavelet libraries (like pytorch\_wavelets) often rely on CPU-based C++ extensions or unoptimized CUDA kernels, they are unsuitable for a JAX/TPU pipeline. A pure JAX implementation is required.

### **4.1 Convolution-Based DWT Implementation**

The DWT can be mathematically formulated as a pair of convolution operations followed by downsampling. The decomposition utilizes a scaling function $\\phi(t)$ (low-pass) and a mother wavelet $\\psi(t)$ (high-pass). For orthogonal wavelets like Daubechies, the filter coefficients are fixed constants.

The implementation steps for a JAX-native DWT layer are:

1. **Filter Instantiation:** The filter coefficients (e.g., db4 low-pass and high-pass decomposition filters) are loaded (likely from PyWavelets during initialization) and stored as immutable JAX arrays (constants) within the layer.  
2. **Signal Extension (Padding):** Correct boundary handling is crucial to prevent artifacts. The 'reflect' (symmetric) mode is standard in signal processing but is not the default in many convolution ops. We must implement a custom padding function using jax.numpy.pad that reflects the signal at the boundaries by len(filter) // 2 samples.20  
3. **Strided Convolution:** The core operation uses jax.lax.conv\_general\_dilated. Unlike standard conv1d, this primitive allows specifying window\_strides=(2,) to perform convolution and downsampling in a single pass, which is highly efficient on TPU.

$$\\begin{aligned} A\_{j+1}\[k\] &= \\sum\_{n} x\[n\] h\[2k \- n\] \\\\ D\_{j+1}\[k\] &= \\sum\_{n} x\[n\] g\[2k \- n\] \\end{aligned}$$  
The filters $h$ and $g$ must be reversed in time compared to the mathematical definition if using standard convolution routines, a detail often missed in porting.22

### **4.2 Static Shape Management for Decomposition Levels**

For a multilevel decomposition, this process is recursive: the approximation $A\_j$ becomes the input for level $j+1$. In PyTorch, one might simply loop $J$ times. In JAX, if $J$ is fixed (e.g., $J=4$), we can unroll this recursion in Python during the tracing phase. This creates a static computation graph with fixed tensor sizes at each node ($N, N/2, N/4, \\dots$).

The output of this layer is a PyTree (specifically a list or dictionary) of tensors. This structure is JAX-friendly; jax.tree\_map can be used later to apply the noise addition or normalization operations to all levels simultaneously, effectively vectorizing the diffusion process across the wavelet scales.23

### **4.3 Inverse DWT (IDWT) and Reconstruction**

The reconstruction (synthesis) process involves upsampling (interleaving zeros) followed by convolution with synthesis filters.

$$x\[n\] \= \\sum\_{k} (A\_{j+1}\[k\] \\tilde{h}\[n \- 2k\] \+ D\_{j+1}\[k\] \\tilde{g}\[n \- 2k\])$$

To implement this efficiently on TPU:

1. **Transposed Convolution:** Use jax.lax.conv\_transpose. This operation mathematically corresponds to the upsampling-convolution sequence.  
2. **Perfect Reconstruction Check:** The padding in the IDWT must be the exact inverse of the DWT padding. If 'reflect' padding was used, the IDWT must crop the output to remove the boundary artifacts. Unit tests asserting $|x \- IDWT(DWT(x))| \< \\epsilon$ are critical during the porting phase to ensure the signal processing pipeline is lossless.20

## **5\. Optimization for Cloud TPU v6e (Trillium)**

The TPU v6e (Trillium) architecture introduces specific constraints and capabilities that differ significantly from GPUs and previous TPU generations. Understanding these is essential for optimizing WaveletDiff.

### **5.1 Leveraging the Matrix Multiply Unit (MXU)**

The TPU v6e features an upgraded MXU size of 256x256 (compared to 128x128 in v4/v5e).5 This systolic array is the engine of the TPU, designed for massive matrix multiplications.

* **Padding Requirement:** To utilize the MXU efficiently, the contracting dimensions of matrix operations should ideally be multiples of 256\. If a dimension is 128 (a common head dimension in Transformers), the compiler may pad it to 256, potentially wasting 50% of the compute cycles.  
* **WaveletDiff Implication:** The channel dimensions in the wavelet-domain transformers should be scaled to align with this 256-block size. For example, using a hidden dimension ($d\_{model}$) of 512, 768, or 1024 is preferable to 64 or 128\. Similarly, the batch size per core should be a multiple of 256 to maximize throughput.6

### **5.2 Precision and Mixed Precision Training**

TPU v6e is heavily optimized for bfloat16 (Brain Floating Point). bfloat16 has the same exponent range as float32 but reduced mantissa precision.

* **Strategic Precision:**  
  * **DWT/IDWT:** These should remain in float32. Wavelet transforms involve accumulating sums of products; precision loss here leads to imperfect reconstruction, which introduces artifacts in the generated time series that the diffusion model might misinterpret as features.24  
  * **Transformer Backbone:** The heavy matrix multiplications in the attention and feed-forward blocks should use bfloat16. Keras 3’s mixed\_precision policy can handle this, but granular control (casting inputs to bf16 after DWT and back to f32 before IDWT) is recommended.  
  * **Diffusion Schedule:** The cumulative noise variance calculations involve products of many small numbers ($1 \- \\beta\_t$). These must be computed in float32 to avoid underflow, which would catastrophically alter the noise distribution.26

### **5.3 High-Bandwidth Memory (HBM) Layouts**

XLA on TPU is sensitive to memory layout. The standard "channels-last" format (Batch, Time, Channels) or NHWC is generally preferred for dense layers and convolutions on TPU, as it allows for efficient vector loading into the MXUs.

* **Mismatch Risk:** PyTorch often defaults to "channels-first" (Batch, Channels, Time) or NCHW. Simply porting weights without transposing them will result in functional correctness but severe performance degradation as the TPU runtime is forced to inject Reshape and Transpose kernels before every computation.  
* **Mitigation:** The data pipeline and the DWT layer output should be explicitly shaped as $(B, L, C)$. The Keras layers should be configured with data\_format="channels\_last" (which is the default in Keras, unlike PyTorch).5

## **6\. Engineering the Diffusion Loop: scan vs. Unroll**

A critical architectural decision in JAX is the implementation of the iterative diffusion sampling process. In PyTorch, this is a standard Python for loop. In JAX, the distinction between unrolling and scanning is fundamental.

### **6.1 The Cost of Unrolling**

For a diffusion model with 1000 timesteps ($T=1000$), a Python loop results in a computational graph containing 1000 copies of the model body.

* **Compilation Latency:** XLA must optimize a graph with millions of nodes, leading to compilation times measured in tens of minutes or even hours.  
* **Memory Overhead:** The intermediate representations during compilation can exhaust the host RAM.  
* **Instruction Cache Pressure:** The resulting executable binary becomes extremely large. On the TPU, this can exceed the instruction cache size, causing instruction fetch stalls and slowing down execution significantly.3

### **6.2 Implementing jax.lax.scan**

The solution is jax.lax.scan. This primitive compiles the loop body *once* and executes it repeatedly on the device, maintaining a carry state (the current noisy image $x\_t$ and RNG key).

**Implementation Pattern:**

Python

def diffusion\_step(carrier, t):  
    x\_t, rng \= carrier  
    rng, step\_rng \= jax.random.split(rng)  
      
    \# Keras model call within the scan loop  
    \# We use functional style to pass parameters if not using Keras's built-in state management in this context  
    pred\_noise \= model(x\_t, t, training=False)   
      
    \# Update x\_t based on scheduler (e.g., DDPM/DDIM equations)  
    x\_t\_minus\_1 \= scheduler\_step(x\_t, pred\_noise, t, step\_rng)  
      
    return (x\_t\_minus\_1, rng), x\_t\_minus\_1 \# Return state and verify trajectory

\# The Scan execution  
\# init\_state \= (x\_T, rng\_key)  
\# timesteps \= jnp.arange(T, 0, \-1)  
final\_state, trajectory \= jax.lax.scan(diffusion\_step, init\_state, timesteps)

This structure ensures that the TPU executable remains compact and efficient, enabling scaling to thousands of steps without compilation overhead. For WaveletDiff, where we might have parallel diffusion processes for each level, the scan loop can process the PyTree of coefficients, effectively stepping all levels backward in time simultaneously.4

## **7\. Porting the Cross-Level Attention Mechanism**

WaveletDiff’s performance relies on the interaction between approximation ($A$) and detail ($D$) coefficients. Since $A$ has half the temporal resolution of the signal (at level 1\) and $D$ has the same or different resolutions depending on the decomposition depth, aligning them for attention requires careful tensor manipulation.

### **7.1 Variable Sequence Lengths and Padding**

In PyTorch, one might use standard attention where the Query ($Q$) comes from the detail level ($N$) and Key/Value ($K, V$) come from the approximation level ($N/2$). The attention matrix is $N \\times N/2$.  
On TPU, managing these distinct shapes is efficient if they are static. However, if the batch contains time series of different native lengths, we face the "ragged tensor" problem.

* **Solution: Bucketing and Masking.**  
  1. **Bucketing:** Group time series of similar lengths.  
  2. **Padding:** Pad all series in a bucket to the maximum length in that bucket (must be a static multiple of 256 for TPU efficiency).  
  3. **Attention Masking:** Create a boolean mask mask\_q (for the query sequence) and mask\_k (for the key sequence). In the MultiHeadAttention layer, setting mask\[i, j\] \= \-inf ensures that padding tokens do not contribute to the softmax attention weights.  
* **Packing:** For extreme efficiency, multiple short sequences can be "packed" into a single long sequence (e.g., packing four length-256 series into a 1024 tensor). This requires "Block-Diagonal" masking to prevent cross-contamination between the independent series within the packed tensor. JAX libraries like FlashAttention often support this natively.27

### **7.2 Efficient Cross-Attention Implementation**

In Keras 3, the MultiHeadAttention layer supports attention\_mask. For the cross-level attention in WaveletDiff:

1. **Projection:** The coarser level ($A$) often has higher energy density. It may be beneficial to project it to the same feature dimension as the detail level ($D$) before attention.  
2. **FlashAttention on TPU:** Recent JAX releases support FlashAttention kernels on TPU. These are fused attention kernels that minimize HBM access. To trigger this path in Keras/JAX, the head dimension should be fixed to 64 or 128, and the precision should be bfloat16. This optimization can yield a 2-4x speedup in the attention blocks.28

## **8\. Implementing Custom Components: AdaLayerNorm and Vectorized Loss**

### **8.1 Adaptive Layer Normalization (AdaLayerNorm)**

Diffusion models typically condition the generation on the timestep $t$ using Adaptive Layer Normalization. This layer modulates the scale ($\\gamma$) and shift ($\\beta$) parameters of LayerNorm based on an embedding of $t$.

$$AdaLN(x, t) \= y\_s(t) \\cdot LN(x) \+ y\_b(t)$$

Keras 3 does not have a built-in AdaLayerNorm. We must implement this as a custom keras.Layer.

Python

class AdaLayerNorm(keras.layers.Layer):  
    def \_\_init\_\_(self, epsilon=1e-5, \*\*kwargs):  
        super().\_\_init\_\_(\*\*kwargs)  
        self.epsilon \= epsilon

    def build(self, input\_shape):  
        \# input\_shape is a list: \[x\_shape, t\_emb\_shape\]  
        dim \= input\_shape\[-1\]  
        self.layernorm \= keras.layers.LayerNormalization(epsilon=self.epsilon, center=False, scale=False)  
        \# Linear projection for scale and shift  
        self.scale\_proj \= keras.layers.Dense(dim)  
        self.shift\_proj \= keras.layers.Dense(dim)

    def call(self, inputs):  
        x, t\_emb \= inputs  
        norm\_x \= self.layernorm(x)  
        scale \= self.scale\_proj(t\_emb)  
        shift \= self.shift\_proj(t\_emb)  
        \# Reshape scale/shift for broadcasting if necessary  
        return (1 \+ scale) \* norm\_x \+ shift

This custom layer must be JIT-compatible. The dense projections for scale and shift effectively "inject" the timestep information into every block of the network.19

### **8.2 Vectorizing the Wavelet-Aware Loss**

WaveletDiff optimizes a weighted loss across levels:

$$L \= \\sum\_{l=1}^L \\lambda\_l \\| \\epsilon \- \\epsilon\_\\theta(x\_t^{(l)}, t) \\|^2$$

In PyTorch, this is often a loop accumulation. In JAX, we can vectorize this.

* **Mechanism:** If the model outputs a PyTree of noise predictions (one per level), and we have a corresponding PyTree of ground truth noise and weights $\\lambda$.  
* **Vectorization:** We can use jax.tree\_map to compute the squared error for each level. Then, we use jax.tree\_util.tree\_reduce (sum) to aggregate the weighted errors into a single scalar loss.  
* **Gradient Descent:** This scalar loss is then differentiated using jax.grad. This approach is cleaner and likely faster than explicit looping, as XLA can fuse the element-wise error computations across levels if shapes permit (or run them in parallel streams).1

## **9\. Migration Workflow: From PyTorch to Keras 3**

The migration involves a systematic process to ensure functional and numerical parity.

### **9.1 Weight Conversion and State Dict Mapping**

The first step is porting the pre-trained weights or ensuring the initialization logic matches.

1. **Extraction:** Load the PyTorch model and export the state\_dict to a file (e.g., using numpy.save or a standard .pt file).  
2. **Mapping Dictionary:** Create a translation map.  
   * PyTorch: layer.0.weight (Shape: \[Out, In\]) $\\to$ Keras: layer\_0/kernel (Shape: \[In, Out\]).  
   * **Crucial Step:** Transpose the weight matrices for Linear and Conv layers. PyTorch uses $Out \\times In$ for linear layers; Keras uses $In \\times Out$. For Conv1D, PyTorch is $(Out, In, K)$, Keras is $(K, In, Out)$. Failure to transpose is a common source of "model learns nothing" bugs.31  
3. **Loading:** Use layer.set\_weights(\[kernel, bias\]) in Keras to load the transformed arrays.

### **9.2 Data Pipeline Migration**

PyTorch DataLoader is versatile but Python-bound (Multiprocessing). For TPU training, the data pipeline must maximize throughput to prevent the TPU from starving.

* **tf.data:** The recommended pipeline for Keras/JAX on TPU is tf.data. It runs in C++ and can prefetch data to the TPU host memory.  
* **Strategy:** Convert the time series dataset into TFRecords (Protobuf format). This allows for streaming reads. Use tf.data.Dataset.map for on-the-fly augmentation or preprocessing (like the DWT if done offline, though doing DWT inside the model allows for end-to-end differentiability).  
* **Batching:** Use dataset.batch(global\_batch\_size, drop\_remainder=True). The drop\_remainder=True is critical to ensure static shapes for XLA.5

### **9.3 Custom Training Loop with train\_step**

To implement the diffusion training logic (sampling $t$, adding noise, computing loss), we override keras.Model.train\_step.

Python

class WaveletDiffJAX(keras.Model):  
    def train\_step(self, data):  
        \# Unpack data  
        if isinstance(data, tuple): x, y \= data  
        else: x \= data  
          
        \# 1\. Randomness: Split keys for noise generation  
        rng\_key \= self.make\_rng("diffusion") \# Custom RNG management  
          
        \# 2\. Sample timesteps t uniformly  
        t \= jax.random.randint(rng\_key, (x.shape,), 0, self.T)  
          
        \# 3\. Add Noise (Forward Diffusion) \- Can be done via vmap  
        noise \= jax.random.normal(rng\_key, x.shape)  
        x\_t \= self.q\_sample(x, t, noise)  
          
        \# 4\. Gradient Step  
        def loss\_fn(trainable\_vars):  
            \# Model prediction  
            pred\_noise \= self(x\_t, t, training=True)  
            return self.compute\_wavelet\_loss(noise, pred\_noise)  
              
        grad\_fn \= jax.value\_and\_grad(loss\_fn)  
        loss, grads \= grad\_fn(self.trainable\_variables)  
          
        \# 5\. Optimization  
        self.optimizer.apply\_gradients(zip(grads, self.trainable\_variables))  
          
        return {"loss": loss}

This encapsulates the entire diffusion logic within the JAX-compiled boundary.33

## **10\. Performance Benchmarking and Tuning on v6e**

### **10.1 Profiling with TensorBoard**

Once the model is running, optimization begins. JAX integrates with the TensorFlow Profiler.

* **Trace Analysis:** Capture a trace of the TPU execution. Look for "Step Time" breakdowns.  
* **Idle Time:** If the TPU is idle, the tf.data pipeline is the bottleneck. Increase prefetch buffers or parallelize the data loading.  
* **Op Fusion:** Check if the element-wise operations (noise addition, activation functions) are fused. XLA usually handles this well, but excessive reshaping can break fusion.

### **10.2 Scaling with jax.sharding**

For very large models or batch sizes that exceed the HBM of a single TPU chip (32GB on v6e), we use distributed training.

* **Data Parallelism (DP):** The standard mode. Replicate the model weights on all chips, split the batch. Keras 3 handles this via keras.distribution.DataParallel.  
* **Model Parallelism (MP):** If the model is huge, we shard the weights. JAX's sharding API allows defining a mesh of devices (e.g., $4 \\times 4$ mesh). We can annotate specific tensors (like the large embedding tables or projection weights) to be sharded along specific axes of the mesh. On TPU v6e, the high-speed ICI (Inter-Chip Interconnect) makes this highly efficient, allowing the 256 chips in a pod to function almost as a single massive accelerator.5

## **11\. Strategic Outlook**

Porting WaveletDiff to Keras 3 and JAX on TPU v6e is a high-effort engineering endeavor that yields substantial rewards. The architecture moves from a dynamic, research-oriented execution model to a static, industrial-grade compute graph. By rigorously implementing the Wavelet Transform using JAX primitives, enforcing static shape constraints via padding/bucketing, and utilizing jax.lax.scan for the diffusion process, the model can fully exploit the 256x256 Matrix Multiply Units of the Trillium architecture. This migration not only accelerates training by orders of magnitude compared to unoptimized GPU implementations but also positions the WaveletDiff framework to scale to dataset sizes and sequence lengths previously unreachable in time series generation.

#### **Works cited**

1. WaveletDiff: Multilevel Wavelet Diffusion For Time Series Generation \- arXiv, accessed December 31, 2025, [https://arxiv.org/html/2510.11839v2](https://arxiv.org/html/2510.11839v2)  
2. \[2510.11839\] WaveletDiff: Multilevel Wavelet Diffusion For Time Series Generation \- arXiv, accessed December 31, 2025, [https://arxiv.org/abs/2510.11839](https://arxiv.org/abs/2510.11839)  
3. jax.lax.scan \- JAX documentation, accessed December 31, 2025, [https://docs.jax.dev/en/latest/\_autosummary/jax.lax.scan.html](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)  
4. On Learning JAX – A Framework for High Performance Machine Learning | Alex McKinney, accessed December 31, 2025, [https://afmck.in/posts/2023-05-22-jax-post/](https://afmck.in/posts/2023-05-22-jax-post/)  
5. Train a model using TPU v6e \- Google Cloud Documentation, accessed December 31, 2025, [https://docs.cloud.google.com/tpu/docs/v6e-training](https://docs.cloud.google.com/tpu/docs/v6e-training)  
6. Cloud TPU performance guide | Google Cloud Documentation, accessed December 31, 2025, [https://docs.cloud.google.com/tpu/docs/performance-guide](https://docs.cloud.google.com/tpu/docs/performance-guide)  
7. WaveletDiff: Multilevel Wavelet Diffusion For Time Series Generation \- arXiv, accessed December 31, 2025, [https://arxiv.org/html/2510.11839v1](https://arxiv.org/html/2510.11839v1)  
8. WaveletDiff: Multilevel Wavelet Diffusion For Time Series Generation \- ResearchGate, accessed December 31, 2025, [https://www.researchgate.net/publication/396499417\_WaveletDiff\_Multilevel\_Wavelet\_Diffusion\_For\_Time\_Series\_Generation](https://www.researchgate.net/publication/396499417_WaveletDiff_Multilevel_Wavelet_Diffusion_For_Time_Series_Generation)  
9. VinAIResearch/WaveDiff: Official Pytorch Implementation of the paper: Wavelet Diffusion Models are fast and scalable Image Generators (CVPR'23) \- GitHub, accessed December 31, 2025, [https://github.com/VinAIResearch/WaveDiff](https://github.com/VinAIResearch/WaveDiff)  
10. v0lta/PyTorch-Wavelet-Toolbox \- GitHub, accessed December 31, 2025, [https://github.com/v0lta/PyTorch-Wavelet-Toolbox](https://github.com/v0lta/PyTorch-Wavelet-Toolbox)  
11. WaveletDiff: Multilevel Wavelet Diffusion For Time Series Generation \- ChatPaper, accessed December 31, 2025, [https://chatpaper.com/paper/199863](https://chatpaper.com/paper/199863)  
12. How to vectorize a function over a list of unequal length arrays in JAX \- Stack Overflow, accessed December 31, 2025, [https://stackoverflow.com/questions/74744255/how-to-vectorize-a-function-over-a-list-of-unequal-length-arrays-in-jax](https://stackoverflow.com/questions/74744255/how-to-vectorize-a-function-over-a-list-of-unequal-length-arrays-in-jax)  
13. WHFL: Wavelet-Domain High Frequency Loss for Sketch-to-Image Translation \- CVF Open Access, accessed December 31, 2025, [https://openaccess.thecvf.com/content/WACV2023/papers/Kim\_WHFL\_Wavelet-Domain\_High\_Frequency\_Loss\_for\_Sketch-to-Image\_Translation\_WACV\_2023\_paper.pdf](https://openaccess.thecvf.com/content/WACV2023/papers/Kim_WHFL_Wavelet-Domain_High_Frequency_Loss_for_Sketch-to-Image_Translation_WACV_2023_paper.pdf)  
14. Introducing Keras 3.0, accessed December 31, 2025, [https://keras.io/keras\_3/](https://keras.io/keras_3/)  
15. A guide to JAX for PyTorch developers | Google Cloud Blog, accessed December 31, 2025, [https://cloud.google.com/blog/products/ai-machine-learning/guide-to-jax-for-pytorch-developers](https://cloud.google.com/blog/products/ai-machine-learning/guide-to-jax-for-pytorch-developers)  
16. Part 3: Train a diffusion model for image generation \- JAX AI Stack, accessed December 31, 2025, [https://docs.jaxstack.ai/en/latest/digits\_diffusion\_model.html](https://docs.jaxstack.ai/en/latest/digits_diffusion_model.html)  
17. SPMD in JAX \#2: Transformers in Bare-Metal JAX | Sam D. Buchanan, accessed December 31, 2025, [https://sdbuchanan.com/blog/jax-2/](https://sdbuchanan.com/blog/jax-2/)  
18. Question regarding performance of jax.lax.scan · jax-ml jax · Discussion \#16106 \- GitHub, accessed December 31, 2025, [https://github.com/jax-ml/jax/discussions/16106](https://github.com/jax-ml/jax/discussions/16106)  
19. Making new layers and models via subclassing \- Keras, accessed December 31, 2025, [https://keras.io/guides/making\_new\_layers\_and\_models\_via\_subclassing/](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)  
20. crowsonkb/jax-wavelets: The 2D discrete wavelet transform for JAX \- GitHub, accessed December 31, 2025, [https://github.com/crowsonkb/jax-wavelets](https://github.com/crowsonkb/jax-wavelets)  
21. jaxwt package — Jax Wavelet Toolbox documentation, accessed December 31, 2025, [https://jax-wavelet-toolbox.readthedocs.io/en/v0.1.1/jaxwt.html](https://jax-wavelet-toolbox.readthedocs.io/en/v0.1.1/jaxwt.html)  
22. Discrete wavelet transform; how to interpret approximation and detail coefficients?, accessed December 31, 2025, [https://dsp.stackexchange.com/questions/44285/discrete-wavelet-transform-how-to-interpret-approximation-and-detail-coefficien](https://dsp.stackexchange.com/questions/44285/discrete-wavelet-transform-how-to-interpret-approximation-and-detail-coefficien)  
23. Wavelet Transforms in Python with Google JAX | by Shailesh Kumar | The Horizon Explorer, accessed December 31, 2025, [https://thehorizonexplorer.org/wavelet-transforms-in-python-with-google-jax-cfd7ca9a39c6](https://thehorizonexplorer.org/wavelet-transforms-in-python-with-google-jax-cfd7ca9a39c6)  
24. Wavelet transform (JAX) — S2WAV 1.0.4 documentation, accessed December 31, 2025, [https://astro-informatics.github.io/s2wav/tutorials/jax\_transform/jax\_transforms.html](https://astro-informatics.github.io/s2wav/tutorials/jax_transform/jax_transforms.html)  
25. How to Think About TPUs | How To Scale Your Model \- GitHub Pages, accessed December 31, 2025, [https://jax-ml.github.io/scaling-book/tpus/](https://jax-ml.github.io/scaling-book/tpus/)  
26. Jax — Transformer Engine 2.9.0 documentation, accessed December 31, 2025, [https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html)  
27. Tutorial 6 (JAX): Transformers and Multi-Head Attention — UvA DL Notebooks v1.2 documentation, accessed December 31, 2025, [https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial\_notebooks/JAX/tutorial6/Transformers\_and\_MHAttention.html](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html)  
28. Kvax: Fast and easy-to-use Flash Attention implementation for JAX \- Nebius, accessed December 31, 2025, [https://nebius.com/blog/posts/kvax-open-source-flash-attention-for-jax](https://nebius.com/blog/posts/kvax-open-source-flash-attention-for-jax)  
29. Fine-tuning RecurrentGemma using JAX and Flax | Google AI for Developers, accessed December 31, 2025, [https://ai.google.dev/gemma/docs/recurrentgemma/recurrentgemma\_jax\_finetune](https://ai.google.dev/gemma/docs/recurrentgemma/recurrentgemma_jax_finetune)  
30. Keras Layer that wraps a JAX model. — layer\_jax\_model\_wrapper \- keras3 \- Posit, accessed December 31, 2025, [https://keras3.posit.co/reference/layer\_jax\_model\_wrapper.html](https://keras3.posit.co/reference/layer_jax_model_wrapper.html)  
31. Saving and Loading Models — PyTorch Tutorials 2.9.0+cu128 documentation, accessed December 31, 2025, [https://docs.pytorch.org/tutorials/beginner/saving\_loading\_models.html](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)  
32. How to load a model properly? \- PyTorch Forums, accessed December 31, 2025, [https://discuss.pytorch.org/t/how-to-load-a-model-properly/131947](https://discuss.pytorch.org/t/how-to-load-a-model-properly/131947)  
33. Customizing what happens in \`fit()\` with JAX \- Keras, accessed December 31, 2025, [https://keras.io/guides/custom\_train\_step\_in\_jax/](https://keras.io/guides/custom_train_step_in_jax/)