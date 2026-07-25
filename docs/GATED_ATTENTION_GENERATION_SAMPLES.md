# Gated ternary-attention generation samples

These samples came from the two strongest 750-step RunPod pilot checkpoints.
They are deliberately unedited. The models are tiny and only lightly adapted,
so the samples test whether generation remains recognizable rather than whether
the prose is production quality.

| Checkpoint | Validation loss | Perplexity | Inference attention path |
|---|---:|---:|---|
| Ternary Q/K/V | 2.289840 | 9.8734 | ternary Q/K/V, existing 2-bit probability route |
| Strict LUT, distilled | 2.299876 | 9.9729 | ternary Q/K/V, 2-bit score, four-entry Q15 exponential LUT, low-bit route |

## Ternary Q/K/V

### Prompt: `Once upon a time`

> Once upon a time, there was a little girl named Sarah. She loved to have a
> puzzle. Every day she would jump out the box and play with it. One day, Sarah
> would get it for her birthday. She was so excited to see what was inside.
>
> Sarah was so excited to have found the puzzle! She put it on and started to
> dance. She felt so proud that she could fly!
>
> After she finished her success with her new puzzle, then ran back home. She
> was still

### Prompt: `Lily found a tiny red door`

> Lily found a tiny red door. She wanted to test it. She was scared of the
> monster. She wanted to get away. But no one came. She started to worry.
>
> But Lily tried to open the letter. She did not know what was in the mail. She
> had a hat and a key. She stuck her hand to the door. She pushed it with a
> open.
>
> But then she heard a voice. She saw the door. It was her mom. She saw the
> box. She saw Lily

### Prompt: `Tom wanted to help his friend`

> Tom wanted to help his friend Sally. She said yes and gave him a big hug.
> They decided to hug and make her walk. After a long time, they decided to go
> home.
>
> On the day, they went to the park, Tom was in the park. There was a beautiful
> park with lots of slides and flowers. Tim loved the park and the park with
> his friends.
>
> Tim was happy that his friend and they played on the swings. But when he saw
> a big tree. It was very tall

## Strict integer-LUT route with distillation

### Prompt: `Once upon a time`

> Once upon a time, there was a little girl named Lily. She loved to eat candy,
> but one day she did not like the candy she had any candy. She didn't know
> what to do was her favorite.
>
> Lily's mom said, "Mommy, I forgot the candy before you to eat it, sweetie?"
> Her mom said, "It's a good story. It's very good for you, Lily. I don't like
> candy."
>
> Lily was so happy and said, "Thank you

### Prompt: `Lily found a tiny red door`

> Lily found a tiny red door. She saw a big hole in her hand. The wind blew and
> did not know what to do. Lily tried to get it back.
>
> She ran to her mom and dad. They saw the big box. They were scared. Mom and
> dad came to the house with the key. The storm was very strong, and it could
> not fly.
>
> They reached the box and saw that the door was on the ground. They saw a big
> box with a lock on it. They decided to

### Prompt: `Tom wanted to help his friend`

> Tom wanted to help his friend, so he asked her to help him. Mom said, "You
> need to be gentle, but be careful. You can't have any more. You must be
> careful and you will fallen off my head."
>
> Tom did not like to do it. He did it again. The end.

## Reading the samples honestly

Both checkpoints preserve the basic TinyStories form: characters, objects,
simple actions, and a story-like continuation. Both also show repetition,
grammar errors, identity drift, and occasional contradictions. The strict
integer-LUT checkpoint remains generative, but these six samples do not prove
parity. Full validation, multiple seeds, and human or automated generation
evaluation are still required.
