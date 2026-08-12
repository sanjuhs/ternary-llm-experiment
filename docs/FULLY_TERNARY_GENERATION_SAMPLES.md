# Fully ternary residual generation samples

These are unedited outputs from the strictest residual-curriculum checkpoints.
Every persistent residual boundary uses exactly the three codes `{-1, 0, +1}`.
Q, K, and V are ternary, the feed-forward nonlinearity is ReLU, attention uses
binary routes, and the score path uses a two-bit code plus a four-entry integer
lookup. Wider accumulators and fixed-point scale state remain necessary.

These samples demonstrate that the inference graph executes and emits partially
story-like text. They do **not** demonstrate useful language quality or parity
with the A4/A16 controls.

## Strict ternary-operand endpoint

- Architecture: factorizable per-head QKV scales, ternary Q/K/V, two-bit
  integer attention, three ternary residual planes, ReLU, integer RMSNorm
- Exhaustive validation loss: **2.250638**
- Perplexity: **9.4938**
- Packed ternary-operand contract: **passed**
- Remaining non-fused boundaries: fixed-point requantization arithmetic and
  final token-sampling Softmax

```text
Once upon a time, there was a little girl named Sarah. Sarah had a best sister and mum and gave her lots of toys. One day, Sarah and her mum went for a ride.

"Sarah, it looks so fun!" said Sarah with a big smile.

"Sure, just sit here," her mum said, "That's a very nice seat. I'll get you one."

Sarah was so happy and she thanked her mum, and then they went off off together to find a special one. Sarah was excited to have her new toy back and her mum was so happy.

Lily found a tiny red door. She wanted to move it. She walked up to the door and opened it with a smile on her face. She tried to pull the door open the door, but it was too heavy. The door was locked. Lily was scared and scared.

Later the next day, Lily found a very beautiful purple car inside the garden. She wanted to open the door, but she could not. But then she got a persistent tutor to climb up. As she gaked the car up and the car started to move!

Lily was so happy that she finally managed to lift

Tom wanted to help his friend, Lily. They had a big card that they could use. They used some tape to make the card stick the card stick.
"Look, Mom, Mom, this card for the letter!" Tom said. He put the card with the card and put the card stick it in the box. He gave it to Lily and said, "Let's open the box!"

They put the card in the box and took the box. They found the card and the card stick and started to cut the card. They thought the card was very pretty and made a picture of the card.
```

These outputs are unedited. They are markedly more coherent than the early
one-plane checkpoints below, but repetition and broken narrative transitions
remain visible.

## Best strict checkpoint

- Training: 1,500 additional teacher-free cross-entropy steps at the final
  three-code stage
- Validation loss: **5.098914**
- Perplexity: **163.8439**
- Residual code use: 26.46% zero and 73.54% saturated nonzero

```text
Once upon a time. They saw!" her things. 
 She were so shiny. They did but Timmy gave the boy. He was very happy and have the girl took a a fun, Lily, She opened the garden. And The little man was an huge. She wanted the day the park and the big mom says. He was time, it and made it was a big day and angry.

Onceory on to like the best it got each mom, they was very girl." He said said. She knew. She thought the ground. The water and get a big boy of the time.
Lily found a tiny red door. They't not bite. She had cry a big and Lily. They are be sad, the slide's. She tried it. But the flowers and Lily's girl and they was her, Anna saw the kitchen. They see in the new big bear, the welcome with the bunny. The mom was beautiful friend, he was so a special place.



Tim was very kind ror. It to see the best new fish.
Ben had the new garden."




 "

But the ground. The boy and living.
Tom wanted to help his friend to show, Lily. He thanked and that they said and a time, but but "
The man's friends.
```

## Strict binary-route checkpoint before the extra CE-only phase

- Training: 750 steps after the ReLU transition
- Validation loss: **5.142406**
- Perplexity: **171.1270**

```text
Once upon a time then was a very her girl. 
eter."
Lily found a tiny red door. They saw a time a mess and the girl, Lily is a house. As, she did.
"
When he started to my little ground. He saw her, She told the pretty for my door.
Tom in the new big bear, the favorite home and the big mom felt very little friend, he was so the voice a day said they said when.


But Jack and have it, he was just.
"






 "OnceTom.

One day to a tall me for not eat the boy. He was happy
Tom wanted to help his friend. They saw a big toy that and She were his wind is a big time a girl looked was the little girl.
```

The complete checkpoints, resolved configurations, metrics, and diagnostics are
mirrored in the
[Hugging Face artifact bundle](https://huggingface.co/sanjuhs/ternary-llm-experiment/tree/main/fully-ternary-pilot).
