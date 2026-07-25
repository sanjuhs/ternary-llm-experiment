The biggest issue with the Bonsai thing is that not everything is quantized, right? How would you quantize everything? In a transformer, if you look at it, a lot of parts are already quantized, right? That is, the attention can be quantized, the weights can be quantized, and the other parts, which include the KV cache and, if I'm not wrong, activations, can be quantized, correct? How would that work

So in the bonsai, I think they've only quantized a few things, not everything.
Worked for 1m 17s

You are mostly correct, but Bonsai has quantized more of the model than “only a few layers” suggests.

It quantizes all 100 matrix-heavy linear layers inside its 25 MMDiT blocks—including Q/K/V projections, attention output projections and MLP linears. Those layers represent more than 95% of the transformer’s parameters. What it does not quantize aggressively is much of the dynamic computation around those weights: activations, attention scores, normalization, residual streams, modulation layers, the VAE and portions of the text-conditioning pipeline.

What Bonsai actually looks like

Approximately:

Prompt
  │
  ▼
Qwen text encoder
  └── weights: 4-bit
      activations: probably FP16/BF16 or higher precision
  │
  ▼
MMDiT transformer
  ├── Q/K/V weight matrices: ternary
  ├── attention-output weights: ternary
  ├── MLP weight matrices: ternary
  ├── activations: higher precision
  ├── attention scores/softmax: higher precision
  ├── modulation + embeddings: FP16
  ├── output norm: FP16
  └── final projection: FP16
  │
  ▼
Latent denoising state
  └── FP16/BF16
  │
  ▼
VAE decoder
  └── FP16

The supporting FP16 tensors are under 5% of the transformer’s parameters, but that does not mean under 5% of runtime memory or computation. Activations scale with image resolution and token count, so they can become the dominant runtime cost even though they are not stored in the checkpoint.

The important W versus A distinction

A label such as:

W1.58A16

means:

Weights: ternary, approximately 1.58-bit.
Activations: FP16.

Bonsai is broadly in this category, although exact precision varies between operations.

Something such as:

W1.58A4

means:

Ternary weights.
4-bit activations entering the matrix multiplications.

And:

W1.58A2

means:

Ternary weights.
Activations averaging around 2 bits.

That last configuration is much closer to the genuinely low-bit model you are imagining.

How activation quantization works

Suppose a layer receives a floating-point activation tensor x.

You calculate a scale s, usually per token or per group:

s=
2
b−1
−1
max(∣x∣)
	​


Then convert the values into integers:

x
q
	​

=clip(round(
s
x
	​

),q
min
	​

,q
max
	​

)

The matrix multiplication becomes conceptually:

y≈s
x
	​

s
w
	​

(x
q
	​

W
q
	​

)

Where:

W
q
	​

∈{−1,0,+1}
x
q
	​

 might be INT2 or INT4
accumulation generally happens in INT16, INT32 or FP32
the result may be rescaled into FP16 or quantized again before the next layer

A realistic low-bit transformer is therefore not permanently storing every intermediate as 2-bit. It repeatedly does:

quantize → low-bit matrix multiply → accumulate → rescale → quantize again

The quantization and rescaling steps need to be fused into kernels; otherwise their overhead can eliminate much of the speed advantage.

Attention contains several separately quantizable things

When people say “attention is quantized,” they may mean very different things.

1. Attention projection weights
Q = XWq
K = XWk
V = XWv

Bonsai quantizes W
q
	​

, W
k
	​

 and W
v
	​

 to ternary values.

But X, Q, K and V may still remain FP16.

2. Q, K and V activations

You can quantize the input and outputs:

X: A4
Q: A4
K: A4
V: A4

Then the projection becomes a W1.58A4 operation.

This is considerably harder than weight-only quantization because activation distributions change with:

Prompt
Image content
Resolution
Transformer depth
Denoising timestep

DiT activations are particularly troublesome because their distributions shift throughout the denoising trajectory. Post-GELU activations have also been found to vary substantially between timesteps.

3. Attention scores

The attention logits are:

S=
d
	​

QK
⊤
	​


These can be calculated using low-bit Q and K, but the accumulator normally needs higher precision.

You could use:

Q/K: INT4
QKᵀ accumulation: INT32
attention logits: INT8

RobuQ, for example, keeps attention-score operations at 8-bit because these activation–activation operations are highly sensitive but represent only a relatively small fraction of overall FLOPs.

4. Softmax

Softmax is awkward to quantize:

P=softmax(S)

It involves exponentials, normalization and values spanning very different magnitudes. It can be approximated with fixed-point arithmetic, lookup tables or integer polynomial approximations, but retaining FP16 or INT8 here usually costs little compared with the large QKV and MLP matrix multiplications.

5. Attention-value multiplication
O=PV

Both P and V could be INT8 or INT4, but quantizing the attention probabilities too aggressively can remove weak yet important attention connections.

Therefore, a sensible attention configuration is something like:

Projection weights:       ternary
Projection inputs:        INT4 or mixed INT2–INT4
Q/K/V:                    INT4
QKᵀ accumulator:          INT32
Attention scores:         INT8
Softmax:                  FP16 or INT8 approximation
Attention × V:            INT8
Output projection input:  INT4

This still means most expensive operations are low-bit without forcing every numerically sensitive operation into ternary arithmetic.

There is usually no LLM-style KV cache here

This is an important difference.

An autoregressive LLM generates one token after another. Earlier tokens do not change, so their keys and values can be cached:

token 1 → token 2 → token 3 → token 4
          reuse previous K/V

An ordinary diffusion transformer instead processes the entire latent image grid at every denoising step:

noisy latent at t=4
        ↓
new latent at t=3
        ↓
new latent at t=2
        ↓
new latent at t=1

Because the image-token activations change between every step, their K and V tensors also change. Therefore, there is no exact persistent KV cache comparable to an LLM’s cache.

FLUX-style MMDiTs also exchange information bidirectionally between text and image streams, further reducing the amount that can simply be cached unchanged.

There are approximate diffusion-caching methods that reuse block outputs or attention results across nearby denoising steps, but that is different from ordinary KV-cache quantization. With Bonsai using only four denoising steps, caching would also offer less benefit than in a 30–50-step diffusion pipeline.

Why activations are much harder than weights

Weights are static. You can examine them, select scales and optimize the model around their quantized representation.

Activations are generated dynamically and contain:

Outlier channels
Different distributions per token
Different distributions per transformer block
Different distributions per timestep
Strongly asymmetric post-activation values
Residual additions that accumulate quantization error

For instance, suppose an activation vector is:

[-0.08, 0.12, 0.04, 0.16, 11.4]

The last outlier determines the INT2 scale. The other four values can collapse to zero.

One solution is an orthogonal rotation such as a Hadamard transform:

before: energy concentrated in one channel
after:  energy distributed across many channels

That makes the activation distribution easier to quantize. RobuQ uses Hadamard transformations and per-token quantization to push DiT activations toward approximately normal distributions before converting them to low precision.

What a genuinely end-to-end quantized Bonsai could use

A practical target would be:

Component	Sensible precision
DiT QKV weights	Ternary
DiT attention-output weights	Ternary
DiT MLP weights	Ternary
DiT activations	Mixed INT2–INT4
Attention scores	INT8
Softmax	INT8 or FP16
AdaLN/modulation	INT4–INT8
Residual accumulation	INT8, FP16 or INT16
Input embeddings	INT8 or FP16
Final output layer	INT8 or FP16
Text-encoder weights	INT4
Text-encoder activations	INT8 or INT4
VAE weights	INT8 or INT4
VAE activations	INT8
Denoising latent state	FP16 or carefully calibrated INT8
Scheduler arithmetic	FP16/FP32

This would be an end-to-end mixed-precision low-bit pipeline, even though a small percentage of operations remain FP16.

Trying to make literally every operation W1.58A1.58 would likely be counterproductive. Softmax probabilities, normalization statistics, residual sums, timestep modulation and the evolving latent state do not naturally fit a three-value representation.

How I would build it
Phase 1: Quantize the DiT activations

Start from Bonsai’s ternary checkpoint and add fake quantization around:

attn.qkv
attn.proj
mlp.fc1
mlp.fc2

Begin at W1.58A8, then move to W1.58A4.

Use:

Per-token activation scales
Timestep-aware calibration
Hadamard rotations
Quantization-aware training
Distillation against the FP16 FLUX.2 Klein teacher

Weight-only post-training quantization probably will not be enough at A2–A4.

Phase 2: Use mixed activation precision

Do not force all layers to 2 bits.

Something like:

early MLP blocks:         A2
middle MLP blocks:        A2–A3
late MLP blocks:          A3–A4
QKV projections:          A3–A4
attention projections:    A4
attention scores:         A8
AdaLN:                    A4

RobuQ found that attention paths and later blocks generally require more activation precision, with attention output projections particularly sensitive under tight bit budgets.

Phase 3: Quantize the rest of the pipeline separately

The text encoder and VAE have very different distributions from the DiT, so they should have separate calibration and fine-tuning.

For example:

Text encoder: W4A8 → W4A4
VAE encoder:  W8A8
VAE decoder:  W8A8 → W4A8

The VAE is only around 0.17 GB in Bonsai’s current package, so reducing it from FP16 to INT8 saves little storage. The more valuable reason to quantize it would be faster decoding and lower activation memory at high resolution.

Phase 4: Write real low-bit kernels

This is essential.

Current hardware does not generally provide native ternary matrix multiplication. Bonsai packs ternary values into 2-bit slots, while RobuQ’s published deployment packs values and executes through an INT4 backend rather than a truly native ternary unit.

For genuine acceleration you would need fused kernels that perform:

packed ternary weight load
+ packed INT2/INT4 activation load
+ low-bit matrix multiplication
+ scale application
+ bias
+ activation function
+ requantization

Without fusion, you may get a much smaller model but not proportionate speed or energy savings.

The bottom line

Bonsai is:

Nearly complete ternary weight quantization of the DiT, but not complete computation quantization.

It quantizes the parts responsible for most parameter storage, but activations and sensitive operations remain substantially higher precision.

What you are describing is closer to:

Ternary-weight, ultra-low-bit-activation diffusion—W1.58A2/A4—with an INT8 VAE and text encoder.

RobuQ is currently the clearest research demonstration of that direction, but even RobuQ retains embeddings and the final layer in full precision, attention-score operations at 8-bit, AdaLN at 4-bit and a floating-point low-rank compensation branch. In practice, “fully quantized” means almost all expensive FLOPs are low-bit, not that absolutely every scalar is represented using 1.58 bits.

A weekly watch for newly released W1.58A2 image checkpoints and kernels would be useful here; say the word and I’ll set it up.

Worked for 2m 43s

Yes—a transformer can function with ternary weights and ternary activations. The surprising part is that representational capacity is probably not the main obstacle.

A 768-dimensional residual vector whose coordinates are only {−1,0,+1} can still represent:

3
768
≈10
366

different states per token. That is an absurdly large state space.

The real obstacles are:

Accumulation
Optimization
Maintaining information through depth
Producing useful attention and output probabilities
The four meanings of “everything is 1.58-bit”

These are often mixed together:

Level	Meaning	Feasible?
1	Every learned weight is ternary	Yes, demonstrated
2	Weights and stored activations are ternary	Yes, demonstrated on some transformers
3	Almost every forward operation uses ternary operands	Plausible and hardware-friendly
4	Weights, activations, accumulators, gradients, optimizer states, loss and updates are all ternary	Not with conventional transformer training

BitNet trains ternary weights from scratch, but uses 8-bit activations and retains high-precision latent weights, gradients and optimizer states.

Separately, Meta researchers demonstrated generative transformers with both ternary weights and ternary activations. Their ternary BART models produced meaningful summarization and translation results. However, they started from full-precision pretrained models, used knowledge distillation and relied on real-valued counterparts and straight-through gradients during training.

So:

Ternary forward computation works. Fully ternary conventional backpropagation is the unsolved part.

Why even a ternary network needs wider accumulators

Consider one attention dot product in GPT-2.

If the head dimension is 64 and:

q
i
	​

,k
i
	​

∈{−1,0,+1}

then every multiplication is also ternary:

q
i
	​

k
i
	​

∈{−1,0,+1}

But the sum is:

s=
i=1
∑
64
	​

q
i
	​

k
i
	​


Therefore:

s∈[−64,64]

That is 129 possible values, requiring approximately 8 bits, not 1.58 bits.

Likewise, an MLP dot product over 3,072 values can range across thousands of integer levels and needs roughly 13 bits for exact accumulation.

This does not defeat the idea. It gives you the most sensible architecture:

ternary operand
× ternary operand
→ small integer accumulator
→ threshold/rescale
→ ternary output

The expensive multipliers disappear. The wider accumulator is small and temporary.

A custom ternary chip could consequently have:

2-bit-packed ternary operands
Add/subtract/skip units instead of multipliers
INT8–INT16 local accumulators
A fused threshold that immediately returns the result to ternary
Almost no FP16 arithmetic
Very low activation-memory bandwidth

That could still be dramatically cheaper than FP16 even though the accumulator itself is not ternary.

What happens if the accumulator is also ternary?

Suppose you force the running sum back into {−1,0,+1} after every addition:

a
i+1
	​

=Q(a
i
	​

+q
i
	​

k
i
	​

)

Then magnitude information disappears almost immediately.

For example:

+1 +1 +1 +1 -1

and:

+1 -1 +1 -1 +1

could collapse to similar internal states, despite representing very different evidence.

The network effectively becomes a system of:

Majority gates
Threshold gates
Boolean/ternary logic
Finite-state transitions

Such a network can still compute extremely complicated functions. Digital computers themselves are built from binary gates. But it would no longer be a normal numerical transformer. It would be closer to a learnable ternary circuit.

It might need much greater width or depth to compensate.

Attention could be redesigned rather than approximated

Conventional attention performs:

softmax(
d
	​

QK
⊤
	​

)V

Softmax is fundamentally continuous. If every attention probability must be one of only three values, ordinary softmax no longer makes much sense.

But a fully ternary model does not have to use ordinary softmax.

Ternary routing attention

Compute integer similarity counts and classify every relationship as:

A
ij
	​

∈{−1,0,+1}

where:

+1: attend
0: ignore
−1: inhibit

The output becomes:

o
i
	​

=Q(
j
∑
	​

A
ij
	​

v
j
	​

)

This is no longer probabilistic attention. It is a learned routing circuit.

Top-k attention

Use ternary Q and K, integer similarity accumulation and select only the best few keys:

ternary Q/K
→ integer match score
→ top-k comparator
→ ternary or binary mask
→ sum selected values

No exponential or division is required.

Associative-memory attention

Another possibility is to interpret each ternary vector as a code word. Attention becomes a nearest-code lookup based on matched positive values, matched negative values and mismatches.

That would resemble content-addressable memory more than softmax attention and may be extremely efficient in custom silicon.

Fully ternary generative transformers have already shown that low-bit attention can produce non-trivial language outputs, but their authors also found attention and autoregressive error accumulation particularly sensitive to extreme quantization.

Residual streams are probably the hardest forward-pass problem

Suppose:

x∈{−1,0,+1}

and a transformer branch produces:

f(x)∈{−1,0,+1}

Then the residual addition produces:

x+f(x)∈{−2,−1,0,+1,+2}

You must ternarize it again:

x
new
	​

=Q(x+f(x))

Across 12 or 24 layers, repeated ternarization can erase small but meaningful updates.

For example:

residual = +1
branch   = small negative contribution

If both are ternary, the branch may either:

Do nothing
Completely cancel the residual
Reverse it

There is no subtle correction.

A radically different residual mechanism

Instead of storing one ternary value per feature, maintain several ternary lanes:

feature = [lane₁, lane₂, lane₃, lane₄]

The effective feature is the population vote:

x
effective
	​

=
r=1
∑
R
	​

x
r
	​


Each physical lane remains ternary, but the collection represents more magnitude levels.

For four lanes, a feature can represent:

−4,−3,…,+3,+4

This is similar to obtaining greater precision through spatial redundancy, rather than greater precision per number.

It costs additional width, but the hardware remains entirely ternary.

You might call this a population-coded ternary transformer.

Why standard backpropagation cannot be literally ternary

The ternary quantizer:

Q(w)∈{−1,0,+1}

is piecewise constant. Its real derivative is:

∂w
∂Q
	​

=0

almost everywhere, and it is undefined at the transition thresholds.

So ordinary gradient descent sees almost no usable gradient.

BitNet solves this using the straight-through estimator: the forward path uses the quantized value, but backward propagation pretends the quantizer has a convenient derivative. It also maintains high-precision hidden weights to accumulate small updates.

Conceptually:

FP32 latent weight:  0.12 → 0.14 → 0.18 → 0.23 → 0.29
ternary weight:         0     0     0     0    +1

Without the latent value, every small update would be discarded.

A ternary weight has only two possible adjacent transitions:

-1 ↔ 0 ↔ +1

A single update is therefore enormous relative to ordinary gradient descent.

Could the gradients themselves be ternary?

Potentially:

g
q
	​

=sign(g)

or:

g
q
	​

∈{−1,0,+1}

This tells the weight only:

Move upward
Stay
Move downward

Low-bit gradient methods have existed for years; early quantized-network research successfully reduced gradients to around six bits, while binary-network forward passes used binary weights and activations.

But a ternary gradient loses two important things:

How confident the update is
How multiple small gradients should accumulate

Normally Adam stores:

m
t
	​

=running gradient mean
v
t
	​

=running squared-gradient mean

Neither can be represented meaningfully with only three values.

A truly ternary training algorithm would look different

Instead of Adam and conventional backpropagation, imagine every parameter as a physical three-state device:

-1 state
 0 state
+1 state

Training sends positive or negative update pulses.

positive pulse → slightly raises probability of upward transition
negative pulse → slightly raises probability of downward transition

After enough accumulated evidence:

-1 → 0
0 → +1

Precision is represented through time and stochastic frequency, not through a high-precision numeric value.

For example, a desired update of +0.01 could mean:

Send a positive transition pulse with probability 1% on every training occurrence.

The weight always remains ternary, but the expected update over many examples is continuous.

This is closely related to:

Stochastic computing
Neuromorphic learning
Probabilistic synapses
Markov-chain optimization
Discrete coordinate descent

There are already experimental methods that train quantized networks without conventional high-precision gradients by using discrete stochastic search. However, these have not established GPT-2-scale, strict all-ternary training; discrete optimization also becomes fundamentally difficult as the model grows.

A strict ternary GPT-2 experiment

I would divide the experiment into three increasingly radical stages.

Stage A — Does the ternary representation have enough capacity?

Use GPT-2 Small or, preferably initially, a 10–50M parameter GPT on TinyStories.

Quantize:

token embeddings              ternary
positional embeddings         ternary
Q/K/V weights                 ternary
Q/K/V activations             ternary
attention output              ternary
MLP weights                   ternary
MLP activations               ternary
residual stream               ternary
normalization parameters      ternary
output projection weights     ternary

Allow:

INT8/INT16 accumulators
FP16 loss
FP16/FP32 latent training weights
FP32 optimizer state

If this trains, it proves that the forward representation is sufficient.

Based on existing fully ternary BART results, I expect this stage could produce coherent language, although likely with noticeably worse perplexity than normal GPT-2.

Stage B — Remove latent full-precision weights

Store only:

ternary model weights
INT8 update counters

Each weight gets a small signed counter:

counter reaches +threshold → weight moves upward
counter reaches -threshold → weight moves downward

For example:

weight:   -1 → 0 → +1
counter:  INT8

The counter acts as the missing update accumulator.

This is no longer literally 1.58-bit training state, but it would still be vastly cheaper than Adam, which normally stores several high-precision values per parameter.

Stage C — Remove numeric counters

Replace counters with stochastic transitions:

P(w→w+1)=f(g)
P(w→w−1)=f(−g)

Now every model parameter remains ternary, and update precision exists only statistically over time.

This is the closest thing to truly ternary learning.

I would expect it to train:

Small classifiers
Tiny language models
Algorithmic tasks
Character-level models

Whether it scales to GPT-2-quality language modelling is genuinely open.

The output layer creates another unavoidable issue

GPT-2 selects among approximately 50,000 vocabulary tokens.

If every final logit is strictly restricted to:

{−1,0,+1}

then 50,000 tokens receive only three possible scores. Thousands of tokens would tie.

That is not enough for nuanced language generation.

But you can retain ternary operands and generate integer logits:

ℓ
v
	​

=
i
∑
	​

h
i
	​

W
vi
	​


where h
i
	​

 and W
vi
	​

 are ternary.

The output score may be an 11–13-bit integer. No multiplication is required; it is only addition, subtraction and counting.

Thus the practical rule should be:

All persistent vectors and operands are ternary, but reductions produce temporary integers.

That is still a fundamentally ternary computer.

Would a GPT-2 trained this way be equally capable?

My expectation:

Ternary weights only

Yes. At sufficient scale, this is already strongly supported by BitNet. BitNet reports matching comparable half-precision transformers from around the multi-billion-parameter range, while preserving high-precision training state.

Ternary weights and activations

Yes, it can learn and generate language. Existing ternary generative transformer experiments establish that it is not intrinsically impossible. But quality degradation and optimization difficulty increase substantially.

Ternary residuals, attention probabilities and embeddings

Probably possible with architecture redesign, increased width, multi-lane coding, learned scales or discrete routing. A vanilla GPT-2 replacement would probably perform poorly.

Ternary gradients and optimizer states

Possibly on small networks using stochastic or discrete optimization. Standard Adam/backpropagation would not work in its usual form.

Every operation, including accumulators and logits, ternary

A network could still compute, but it becomes a threshold circuit rather than a numerical transformer. It would likely require dramatically different architecture and much greater redundancy.

The deepest answer

A neural network does not fundamentally require floating-point numbers.

It requires:

A sufficiently expressive state space
A way to combine information
A way to preserve information through depth
A learning rule capable of searching the state space

Ternary networks clearly have the first two.

The difficult question is whether we can invent a learning rule and architecture for the last two that does not secretly rely on high-precision shadow variables.

So I would phrase the research hypothesis like this:

Can a language model operate as a large, trainable ternary dynamical system, where precision emerges from population coding, accumulation and stochastic time rather than high-bit scalar representations?

That is more radical—and potentially more consequential—than merely quantizing GPT-2.
