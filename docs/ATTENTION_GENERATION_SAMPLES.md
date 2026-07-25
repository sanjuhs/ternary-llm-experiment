# Matched attention-generation samples

These samples compare the equally fine-tuned float-attention control, the
best-quality two-bit-probability arm, and the strict arm that quantizes both
softmax input and probability. All use the same tokenizer, sampling settings,
seed, and prompt within each group. The text is reproduced without editing.

The underlying model has only 5.8 million parameters, so grammar and repetition
errors also appear in the control. These samples are a qualitative sanity check;
the 100-batch validation loss is the primary quality measure.

## Prompt: “Once upon a time, a little robot”

### Float-attention control

> Once upon a time, a little robot was driving down a street. Suddenly, it broke
> a traffordinary place. The robot was sad and it wanted to be fixed. The toy
> was very happy and it wanted to be fixed.
>
> The little robot saw the broken traffic, so fast and slow. It had a great
> journey, but, it was problem. The robot was sad and couldn't find the way back

### Two-bit probability

> Once upon a time, a little robot named Red. Red lived in a blue sky. Red loved
> to play with his friends every day. One day, Red was into the living room with
> his friends, a little bird named Blue.
>
> Tim was very happy. He learned that it is important because it is important to
> always be nice to others. And Red always had new friends who liked to be
> friendly. And they all lived happily

### Two-bit score and probability

> Once upon a time, a little robot named Timmy was playing with his toys and
> running around. Suddenly, he realized he couldn't find them anywhere. He asked
> his mommy where she was going and asked, "During your birthday, can I find
> them?" His mommy smiled and said, "Yes, but not worry, my little robot is in
> the mine. It might not be bad". Timmy was very happy and replied, "No

## Prompt: “Mia found a tiny dragon”

### Float-attention control

> Mia found a tiny dragon in the forest. She was very happy and started to clap
> her hands. But the dragon was very angry and flew far away. Mia was scared and
> ran away.
>
> Mia shouted, "Where did that? And we are going to see a monster!"
>
> But then, Mia heard a

### Two-bit probability

> Mia found a tiny dragon. She wanted to play with it. She ranger very fast.
>
> "Look, Mia, a giant!" Mia shouted. "This dragon is the biggest dragon!"
>
> Mia saw Tim's dragon and was curious. She wanted to know the dragon. She was
> mean, but Tim did

### Two-bit score and probability

> Mia found a tiny dragon in the forest. She had a big stick that she thought
> would be the dragon's and play with it. Mia was very happy and showed it to
> her friends. They all wanted to see the dragon and play together. They all
> wanted to see who could play with the dragon and have fun.

## Prompt: “The moon was afraid of the dark”

### Float-attention control

> The moon was afraid of the dark day and he didn't want to go away.
>
> The moon flew away, and when he got inside, he saw an underneath the sky. He
> looked down and saw the big chest of an old, white ball.
>
> He had never seen an old, old, white

### Two-bit probability

> The moon was afraid of the dark, so he flew away. As he flew, the bird heard a
> strange sound. It was a big, scary monster in the sky!
>
> The sky was risking and shaking. Tim was scared, but he wanted to go back and
> explore the world. He ran to his Mom and

### Two-bit score and probability

> The moon was afraid of the dark and it was dark. Suddenly, the lights started
> to tremble, and the night was very dark. The moon was filled with bright
> lights and they were all gone.
>
> The moon and the moon became best friends. They played together and had lots
> of fun. It was a fun day
