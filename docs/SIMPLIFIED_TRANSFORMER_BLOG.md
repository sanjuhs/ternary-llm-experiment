# Can an entire Transformer think in two bits?

The short answer is: **most of its large tensors can probably be stored and
moved in two bits, but every intermediate arithmetic result cannot remain two
bits.** That distinction matters.

Imagine adding 256 tiny numbers. Even if each number is only `-1`, `0`, or
`+1`, their sum can range from `-256` to `+256`. A two-bit box cannot hold that
sum. Real low-bit chips therefore multiply low-bit values, collect the answer
in a wider accumulator, and compress it again before the next major boundary.

That still gives us the important benefits: much smaller models, less memory
traffic, and matrix multiplications whose inputs are extremely cheap.

## First: ternary is not quite the same as binary

A ternary number has three possible values:

```text
-1, 0, +1
```

Three values contain about 1.58 bits of information. Ordinary computers pack
each ternary value into a two-bit slot, leaving one of the four possible bit
patterns unused. This is why papers often call ternary models “1.58-bit
models,” while their practical files use two bits per weight.

## What is inside a Transformer?

A small language model repeatedly performs the following recipe:

1. Look up a vector for every token.
2. Use attention to decide which earlier tokens matter.
3. Mix and transform that information in a feed-forward network.
4. Add the old information back through a residual connection.
5. Repeat this through several blocks.
6. Turn the final vector into next-token probabilities.

Here is how far each part can be pushed.

| Transformer part | Low-bit target | Difficulty |
|---|---|---|
| Token embeddings | ternary weights | relatively straightforward |
| Q, K, V, and attention-output weight matrices | ternary weights | straightforward with quantization-aware training |
| Feed-forward weight matrices | ternary weights | straightforward with quantization-aware training |
| Q, K, and V activations | ternary or four-bit | hard, but demonstrated in partial systems |
| Attention scores | four 2-bit codes | promising |
| Attention probabilities | four 2-bit codes or binary routes | difficult because probabilities are strongly skewed |
| Residual stream between blocks | ternary or four-bit | one of the hardest parts because errors accumulate |
| Normalization input/output | ternary boundary with wider statistics | possible, but the mean and inverse square root need more precision |
| Dot-product accumulator | INT32 or wider | must be wider; two bits would overflow |
| Softmax row sum | wider integer | must be wider for the same reason |
| Training gradients and optimizer state | float | normally kept high precision during training |

So a defensible “fully quantized Transformer” means that the large stored
tensors and the inputs crossing expensive compute boundaries are low-bit. It
does **not** mean that a 256-term sum is somehow stored in two bits.

### Does an INT32 sum turn the next layer into FP32?

No. Think of the accumulator as a temporary bucket, not the network's storage
format.

For one attention head in our model, a Q·K dot product adds 32 ternary products.
The exact answer fits in about seven signed bits. A feed-forward dot product can
add 1,024 products and needs about twelve. We use INT32 because existing hardware
likes it and it is comfortably safe. A custom chip could use narrower buckets.

After the sum, the chip applies a small scale, rounds the answer, and writes the
next tensor back as ternary or four-bit. The next matrix multiplication therefore
still receives low-bit operands. Nothing requires an FP32 activation to be saved
between the two operations.

Softmax needs a wider row sum and RMSNorm needs a wider sum of squares for the
same reason. Those few temporary numbers do not erase the storage, bandwidth, and
add/subtract/skip benefits of ternary weight and activation matrices.

## What our current numbers mean

Our TinyStories model has about 5.8 million parameters. Starting from its
trained checkpoint:

- ternary weights with approximately 16-bit activations gave validation loss
  **2.0251** and perplexity **7.58** on the original 10-batch comparison;
- Hadamard-projected four-bit activations gave loss **2.1712**;
- COAT-projected four-bit activations gave loss **2.1690**;
- after 1,000 steps of quantization-aware fine-tuning, those A4 losses improved
  to **2.1156** and **2.1126**, respectively;
- direct ternary-activation training was unstable, beginning with gradient
  norms around one trillion and finishing far behind the A4 models.

The important correction is that **2.0251 is not the float baseline**. It is
the ternary-weight/A16 reference on a short, fixed validation sample. On the
larger 100-batch validation, the separately measured float checkpoint scored
1.6989 and its ternary-weight interpretation scored 2.0312. Results should only
be compared when they use the same checkpoint, validation batches, and seed.

## Why attention is special

Attention first computes similarities between queries and keys. It then uses
softmax to turn those scores into probabilities. Most probabilities are tiny,
while a few can be large. A uniform two-bit ruler tends to erase all of the
small values or distort the few important ones.

We are testing three solutions:

1. **Quantize before softmax.** Subtract the row maximum and replace every
   remaining score with one of four codes. A four-entry exponential lookup
   table can then replace a general exponential unit.
2. **Quantize after softmax.** Replace each probability with one of four
   non-negative codes and renormalize the row.
3. **Binary routing.** Keep only selected attention links and give them equal
   weight. This is the cheapest, but also the most destructive.

The first local matched screen was encouraging. We then repeated the comparison
over 100 validation batches on an RTX 4090 and fine-tuned every selected arm for
the same 500-step budget:

| Attention representation after fine-tuning | Loss | Perplexity |
|---|---:|---:|
| Float attention control | 2.1132 | 8.275 |
| Four-level softmax input | 2.1489 | 8.576 |
| **Four-level probability** | **2.1383** | **8.485** |
| Four-level input and probability together | 2.1569 | 8.644 |
| Binary routing, best tested threshold | 2.1889 | 8.926 |

The four-level probability model is only 0.0251 loss above its matched control.
That is very close, but it is not literally the same loss. The stricter model,
where both sides of softmax cross a four-code boundary, is 0.0437 higher. Binary
routing is viable but still trails the four-code variants.

The experiment also showed why “just lower the threshold” is incomplete. A
binary threshold of 0.125 beat both 0.25 and 0.5, but lowering it again to
0.0625 made quality worse because too many tokens received equal weight.

## Why changing training can help

Our failed direct ternary run jumped from a reasonably smooth signal to only
three activation values in one step. That is like teaching someone to paint
and suddenly giving them only black, white, and grey.

A stronger training recipe changes the model gradually:

1. Start from the trained ternary-weight model.
2. Move activations through progressively smaller odd alphabets, such as
   19, 15, 11, 7, and finally 3 values.
3. Match the scale when moving between stages so the next stage does not see a
   sudden magnitude shock.
4. Keep a high-precision teacher beside the quantized student.
5. Teach the student to match the teacher’s output probabilities, attention
   maps, and block outputs—not only the next correct token.
6. Spend at least half of the adaptation budget at the final ternary stage.
7. Introduce low-bit attention only after the residual stream is stable.

This is closely related to the smooth multi-stage recipe in
[BWTA](https://arxiv.org/abs/2604.03957). BWTA is valuable evidence for
ternary activations and binary attention, but its reported language-model
experiment replaces only 30% of the least-sensitive layers. It does not yet
prove that every layer of an LLM can be made ternary without a quality cost.

There is also a small but important hardware detail. BWTA gives a whole layer
one learned scale for Q, K, the attention map, and V. It can therefore perform
the big matrix multiplications on packed binary/ternary codes and multiply by
the scale afterward. Our present experiment gives each V token its own scale,
which usually reconstructs values more accurately but makes Route·V harder to
implement as one pure binary-by-ternary kernel. A future shared-scale V arm will
measure that accuracy-versus-hardware trade-off directly.

## What would count as success?

One attractive sample is not enough. We will call a method competitive only
when:

- its full-validation loss is close to a matched A4 or A16 control;
- the result repeats across at least three random seeds;
- generated stories remain coherent under fixed prompts;
- attention does not collapse into one route or uniform noise;
- gradient norms remain finite and controlled;
- the exported checkpoint really packs its large tensors into the claimed
  low-bit format;
- a low-bit kernel demonstrates a speed or memory benefit, not only fake
  quantization inside floating-point PyTorch.

The hypothesis is plausible. The honest state of the evidence is that ternary
weights are already strong, four-bit activations are close, and two-bit
attention looks experimentally reachable. Uniformly ternary activations across
every layer remain the open part—and the training transition, more than the
threshold alone, is likely to decide whether it works.

## The latest experiment

We have separated Q/K/V precision from residual precision. This matters because
the earlier COAT A4 model quietly inherited four-bit Q, K, and V. The new pilot
forces all three to ternary while keeping the residual stream at four bits long
enough to isolate the attention problem.

The combined student tested:

- Q-ViT-style learned reshaping to make ternary Q and K codes more informative;
- ternary Q·K with a wider integer sum;
- an EXAQ/I-LLM-style four-code softmax input and four-entry integer lookup table;
- either a two-bit attention route or BWTA-style binary route;
- ternary V;
- a learned binary per-head gate that can zero an update cleanly;
- teacher matching on output logits, attention maps, and Q-Q/K-K relationships.

The first post-training switch was intentionally bad: forced ternary Q/K/V
raised loss from the 2.14 range to 3.61, and untrained Q-ViT rectification raised
it further. Training recovered most of that damage:

| Model after equal-budget training | Loss | Plain-language result |
|---|---:|---|
| Matched A4-QKV control | 2.1391 | Best quality in this pilot |
| Ternary Q/K/V | **2.2898** | Best ternary-attention quality |
| Ternary Q/K/V + integer softmax route | **2.2999** | Almost the same as plain ternary Q/K/V |
| Q-ViT rectification + hard gate | 2.5619 | Worse; every gate stayed open |

In simple terms, forcing the three attention operands to ternary now works well
enough to generate recognizable stories. Replacing the score-to-probability path
with two-bit codes and a four-entry integer lookup adds only about 0.01 loss.
However, the strict model is still about 0.16 loss behind the matched control,
so it is promising rather than equivalent.

## Did making the rest ternary work?

We have now run that experiment. Instead of throwing away precision all at once,
we gave the model fewer and fewer activation choices:

```text
19 → 15 → 11 → 9 → 7 → 5 → 3
```

At the final stage, the values saved between Transformer blocks really have only
three choices: negative, zero, or positive. Q, K, and V are ternary too. We used
ReLU so the feed-forward step does not need a complicated smooth lookup, and
attention connections are kept or dropped with a binary route.

The model runs and generates text. That is an important engineering milestone.
But the text is fragmented, and its best validation loss is **5.0989**. The
four-bit activation model scored **2.1126**, where lower is better. So this is
not yet a useful fully ternary language model.

The interesting clue is where quality falls:

| Choices per residual value | Validation loss |
|---:|---:|
| 19 | 2.3975 |
| 11 | 2.7650 |
| 7 | 3.3627 |
| 5 | 4.2366 |
| 3 | **5.0989** after extra training |

The big break happens below seven choices. Think of the residual stream as the
model's notebook. Three symbols are enough to record a direction, but apparently
not enough for this ordinary Transformer to preserve all the details it needs
across six layers.

The next experiment gave each number two or three tiny ternary “planes.” Each
plane still contains only minus, zero, or plus, so a future chip can keep using
cheap ternary operations. Separately scaled planes give the notebook many more
possible combinations. They cost more than one ternary code per number, but
they cleanly test whether residual capacity—not the basic ternary arithmetic—is
the missing ingredient.

The exact result inventory and arithmetic contract are in the
[experiment ledger](EXPERIMENT_LEDGER_AND_ROADMAP.md). The raw story outputs are
in the [attention generation appendix](GATED_ATTENTION_GENERATION_SAMPLES.md)
and the
[fully ternary generation appendix](FULLY_TERNARY_GENERATION_SAMPLES.md).

## There are really three versions of “fully ternary”

This is easiest to understand by imagining that every number carries a tiny
stack of cards.

- With **one ternary card**, the number can be minus, zero, or plus. This is the
  smallest representation, but our ordinary Transformer loses too much detail.
- With **two binary cards**, the model gets four combinations using exactly two
  bits. Every large multiplication is still only add or subtract. This is our
  strict two-bit-storage experiment.
- With **two or three ternary cards**, the model gets many more combinations and
  preserves more detail. A ternary chip can still perform every large matrix
  operation, but the stack now occupies four or six physical bits per number.

So there are three separate questions:

| Question | What would count as success? |
|---|---|
| Can the model be stored in about two bits? | Two binary cards or one packed ternary code at every large boundary |
| Can a ternary chip run the big calculations? | Every matrix operand contains only minus, zero, or plus, even if it uses several planes |
| Can quality match the normal model? | The loss stays within a declared margin on exactly the same validation text |

Recent papers support different boxes. BitNet v2 and TWLA show that four-bit
activations can work very well with ternary weights. R2Q shows how two binary
refinement planes can encode a two-bit weight. ExTernD gets close to normal-model
quality by expanding each matrix into several ternary factors, but spends more
storage and additions. An older but especially relevant result, TBT, generated
summaries and translations with ternary weights **and** ternary activations.
Residual-free Transformers try to redesign the model so the values are easier
to compress in the first place.

The newest results also explain why “two-bit” in a paper title needs careful
reading:

| Paper | What actually runs at inference | What we can borrow |
|---|---|---|
| TBT | Ternary BART/mBART weights and activations on generation tasks | Use `{-scale, 0, +scale}` for signed values, but `{0, scale, 2×scale}` for attention probabilities and ReLU outputs |
| TWLA | Ternary weights; layers choose 2, 4, 6, or 8 activation bits under an average four-bit budget | Rotate values and give sensitive layers more room |
| BWLA | Binary weights; usually six-bit activations; a small higher-precision correction | Shape values into a quantizer-friendly distribution |
| TurboAttention | Q/K/V calculations at eight bits; KV memory mixes two- and four-bit heads | Integer attention, small lookup tables, and head sensitivity |
| IntAttention | The attention pipeline stays integer, but uses eight-bit operands | Integer lookup-table softmax and integer normalization |
| BinaryAttention | Q and K keep only their signs; the rest of the Transformer is not claimed fully ternary | Bitwise Q·K plus training that preserves sign-based similarity |
| ELiTeFormer | Ternary linear projections with hybrid linear attention on an FPGA; the cache/state is compressed but not claimed ternary | A ternary hardware datapath and a possible alternative attention architecture |
| FTerViT | Ternary weights and normalization parameters; eight-bit activations; vision rather than language | A possible ternary normalization design |

None of these papers has already built our exact machine. That is why our
experiment matters: we require the big operands to be ternary, forbid a hidden
floating correction path, and measure story loss on the same text every time.

TBT gives us one very practical correction to the mental model. A ternary
activation does not always need to mean “minus, zero, plus.” Attention
probabilities and ReLU outputs cannot be negative, so spending a code on minus
would waste one third of the alphabet. TBT instead uses “zero, small, large”
for those values. Our integer attention codes already follow that idea, while
our signed residual planes use “minus, zero, plus.” TBT's generation quality
still remained below its full-precision models, and its task scores are not
comparable to TinyStories loss, so it is evidence that the route is real—not
evidence that loss parity has already been solved.

BinaryAttention offers another useful card trick. It throws away Q and K
magnitudes and keeps only plus or minus, then teaches those signs to preserve
the teacher's similarity pattern. That makes Q·K especially cheap, and binary
is a valid subset of what a ternary chip can process. But the paper tested
vision and diffusion models, used eight-bit values and eight-bit routing for
the second attention calculation, and allowed an optional richer bias. We
keep the Q/K trick but reject that higher-precision escape hatch: our fallback
still uses ternary V and two-bit routing. It remains an experiment, not proof
that the complete machine is solved.

One training idea looks especially useful. Ordinary Transformers sometimes
create a few enormous internal numbers. Compressing them is like drawing both a
mountain and a pebble with only four shades: the pebble disappears. A method
called Softmax-1, paired with an optimizer that spreads updates across
directions, kept the normal model's quality while making those internal values
far less extreme. For our integer attention, the natural version is an empty
attention slot: the head may choose “send no message” instead of inventing an
extreme score. We will test that only after the current matched clip and scale
experiments, so it has a clean control.

The first attention repair is already measurable. Giving the four-entry
attention codebook a smaller range made every code usable and reduced the
matched validation loss from 2.2512 to **2.2269**. A longer follow-up first
worsened to 2.2553, recovered to 2.2401, and finished at 2.2490 on the shorter
check. Reading the entire validation collection gave it **2.2354**, our best
complete strict result so far. It also continued using all four attention
symbols.

We kept both checkpoints and retested them on the same pages before choosing
the next experiment's source. The short run scored 2.2269 and the longer run's
best checkpoint scored 2.2408, so the short run advances. The longer endpoint
is still saved because it won the separate full-book measurement. This is why
we distinguish a quick screening score from the final exhaustive score—and why
more training time does not automatically earn a win.

There is one more ordinary Transformer component to simplify. GELU is a curved
activation function that normally needs a floating approximation. ReLU simply
asks whether a number is positive and otherwise replaces it with zero. Our
small experiment slightly favored ReLU, so the large experiment will give GELU
and ReLU exactly the same extra training budget. If ReLU keeps the loss, it
removes another awkward operation from the future ternary chip.

That gives us two sensible products rather than one vague promise:

1. a **strict two-bit model**, where storage wins but quality may be lower; and
2. a **ternary-compute model**, where all large operations suit a ternary chip
   but several code planes may be used to preserve quality.

We will continue to report both, including the cost of scales and wider
temporary sums. Calling the second model “1.58-bit” would be misleading even
though its actual multiplications are ternary.

## What happened when we tried the cards?

We ran the comparison. Every model kept ternary weights, ternary Q/K/V, and the
low-bit attention route. We changed only the number of cards used to carry the
information between Transformer blocks.

| Residual representation | Physical code bits per number | Loss |
|---|---:|---:|
| Two binary cards | **2** | 4.3323 |
| Two ternary cards | 4 | 3.2882 |
| Three ternary cards | 6 | **2.6613** |
| Earlier four-bit activation control | 4 | 2.1126 |

Lower loss is better. The exact-two-bit model works, but it still forgets too
much. Giving each number an explicit zero choice makes a large improvement, and
adding a third ternary card improves it again. The three-card model is far
better than our old single-card result of 5.0989, but it still does not match
the four-bit control.

We also compared a learned COAT rotation with a fixed Hadamard rotation. After
equal training, the fixed transform was a little better in every row. That is
good news for hardware: Hadamard mixing is just a known pattern of additions
and subtractions, so the model does not need a dense floating-point rotation.

We then let the best versions train longer and read the entire validation book,
not just a sample:

| Finished version | Physical bits per number | Full loss |
|---|---:|---:|
| Two binary cards | **2** | 4.2517 |
| Three binary cards | 3 | 3.7509 |
| Three ternary cards | 6 | 2.5985 |
| Five-bit average, extra cards in the noisiest layers | 5 | 2.8590 |
| Three ternary cards plus simple ReLU | 6 | **2.5183** |

Two surprises are useful. First, measuring which layers lose the most
information is better than assuming the last layers deserve all the extra
cards. Second, replacing the smooth GELU function with the much simpler ReLU
made the model better. ReLU is basically “keep positive values, replace
negative values with zero,” which is far easier to implement in small integer
hardware.

The lesson is simple. Rotation helps organize the notebook, but the number of
symbols available in the notebook matters more. The strict two-bit version has
now plateaued, so merely training it longer is unlikely to solve the quality
gap. The promising route is to redesign the network for its ternary cards,
spend extra cards only where measurements justify them, and keep every cost
honest.

The new 27.4-million-parameter normal model has now finished. Its full
validation loss is **1.3442**, and its common-text score is about **0.495 bits
per byte**. The released TinyStories-33M model scores about **0.554 bits per
byte**, where lower is better. In plain language: our normal-sized teacher is
good enough. If its ternary children fall behind, we can no longer blame an
undertrained teacher.

The first ternary child has also finished. Its full loss is **1.5355**, and its
common-text score is about **0.566 bits per byte**—only around 2.2% behind the
released TinyStories-33M model, though still behind its own stronger parent.
Its deployable ternary-weight file is about **7.5 MB**. That is encouraging for
the weights; the next tests ask how much quality is lost when the messages
moving between layers are reduced as well.

The first message-compression test is now finished. Keeping ternary weights but
using four-bit messages produces **1.8001 loss**, compared with **1.5355** when
those messages remain ordinary floating-point values. So four bits are usable,
but they are not free: the model loses a noticeable amount of story-prediction
quality.

The stricter 27.4-million-parameter model has now finished too. It makes Q, K,
and V ternary, routes attention with four tiny integer choices, and carries
each between-block message on three ternary cards. Its best saved checkpoint
scores **2.2410 loss**; continuing the same setup to 30,000 steps makes it
worse, at **2.3039**. The normal model remains at **1.3442**, and ternary
weights with ordinary messages remain at **1.5355**. So the honest answer is:
we have a working ternary-style inference graph that writes recognizable
stories, but it does not yet preserve the normal model's quality.

The failure is informative. As training continued, about 88% of possible
attention routes became zero and the variety of attention choices kept
shrinking. Think of eight people in a meeting where most message channels have
gone silent. Training longer cannot fix a language model if its communication
system is collapsing.

The next experiments therefore change that communication system one piece at
a time. We will try a smaller attention range so all four route cards are
actually used, give Q/K/V one stable scale per head, add a legitimate
“send no message” choice, compare GELU with simple ReLU, try binary Q/K with
ternary V, and finally replace floating RMS normalization with its exact
integer reference. Each candidate receives the same training and validation
budget as its control. A readable sample alone cannot win; the full validation
loss and the ternary/integer arithmetic audit must also pass.
