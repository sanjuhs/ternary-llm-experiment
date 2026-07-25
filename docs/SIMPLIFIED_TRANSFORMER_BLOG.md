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
