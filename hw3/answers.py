r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=64,
        seq_len=150,
        h_dim=256,
        n_layers=3,
        dropout=0.5,
        learn_rate=1e-3,
        lr_sched_factor=0.1,
        lr_sched_patience=1,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I. SCENE I."
    temperature = 0.75
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
Question: Why do we split the corpus into sequences instead of training on the whole text?

Answer:

"""

part1_q2 = r"""
**Your answer:**
Question: How is it possible that the generated text clearly shows memory longer than the sequence length?

Answer:

"""

part1_q3 = r"""
**Your answer:**
Question: Why are we not shuffling the order of batches when training?

Answer:

"""

part1_q4 = r"""
**Your answer:**

Questions:
(1) Why do we lower the temperature for sampling (compared to the default of  1.0 )?
(2) What happens when the temperature is very high and why?
(3) What happens when the temperature is very low and why?

Answers:
**(1)**

**(2)** When we raise the temperature too high (for example to $T=10,000$), the text we generate becomes absolute gibberish. This is because we are essentially evening out the probabilities for every character, giving them all a much mroe similar weight. Thus, the model has lost its ability to generate reasonable sentence structure and spelling, and the choice of characters is essentially random. Here the analogy to temperature in the thermodynamic sense becomes more clear: high temperature causes high entropy, meaning that there is disorder and chaos in the system. Our generated text here is super disordered and chaotic, and very little real information can be obtained from it.

**(3)** When we lower the temperature drastically (for example to $T=0.0001$), the model generates a highly repetitive text that seems to just keep printing the same line over and over, as if the model is unable to generate anything else but that line. What we have done here is essentially boosted the weights of the most likely characters, while diminishing the weights of unlikely characters. This caused the model to generate the sentence "And the streets of the streets of the state" over and over again.

We notice something interesting about this sentence. It contains 3 (the, and, of) out of the top 5 most common words in the corpus (the, and, of, to, I), and is only made up of 11 different letters (n, d, h, r, f, i, o, s, a, t, e), of which 9 are the top 9 most common letters in the corpus (i, o, h, s, a, t, e, r, n). Clearly, our model preferred to use (almost) only the most common words and characters in the corpus, and so this gives us another insight into how the low temperature affected the model.

Harking back to the thermodynamics analogy, the low temperature here caused low entropy, meaning there is too much order and structure and nothing really changes or becomes unpredictable. The same line was printed over and over again, showing too much structure and order (like a crystal), with little unpredictability or chaos.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 128
    hypers['h_dim'] = 128
    hypers['z_dim'] = 32
    hypers['x_sigma2'] = 0.0001
    hypers['learn_rate'] = 0.0002
    hypers['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======

    hypers['batch_size'] = 16
    hypers['z_dim'] = 12
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.2
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['lr'] = 0.0015
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0015
    # ========================
    return hypers


part3_q1 = r"""
In the case of GANs, we want to keep the gradients during training for the generator and discriminator separately because they are trained separately.
When training the generator we maintain gradients of the generator and not the discriminator and vice versa.
"""

part3_q2 = r"""
1.	**NO**. The entire model loss depends on both the generator loss and the discriminator loss and thus the
loss might still be too high even if the generator loss went under some threshold. For example if the discriminator
loss is extremely high, it could cause the generator loss to be low even though the images that it generates are bad.

2.	It could mean that generator’s learning speed is higher than the discriminator’s and that the discriminator is not learning fast enough, to keep up with the generator’s learning pace.

"""

part3_q3 = r"""
The main difference between the images generated from the two models seems to be the colors and the features learned by
each model.
It seems that the VAE model results have more details in the face area, while the GAN results have more details in the body area and the colors are different.

"""


# ==============
