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

The general idea is similar to the reason why mini-batch gradient descent is often more optimal than batch gradient descent: this method is more computationally efficient and allows the model to "see" more data, causing it to learn more effectively. Beyond that, this method allows us to conduct parallel computations on "batches" (sequences), which can help reduce computation time further and boost the learning process. This grants us the ability to use less deep networks for our problem, and so we can avoid vanishing or exploding gradients more efficiently.
"""

part1_q2 = r"""
**Your answer:**

The generated text seems to have a memory longer than the sequence length due to the fact that our model uses hidden states to memorize longer text than the sequence length, as well as the fact that our model is "seeing" the data sequentially and thus relies on information from previous "batches". Thus, it is able to produce longer sentences.
"""

part1_q3 = r"""
**Your answer:**

Our model here is a recurrent neural network (RNN), meaning that it learns from the training data in a sequential manner and so every training step where the model is learning from a "batch" of data will be affected by the batches that precede it, and even moreso it will be affected by their order. If we were to shuffle the order of the batches when training our RNN, we would not allow it to properly learn the "flow" of human language, since the model can no longer view the training text in the correct order. It is crucial in fact that we don't shuffle the order of the batches during training, so that our model will be able to learn correctly and efficiently.
"""

part1_q4 = r"""
**Your answer:**

Questions:
(1) Why do we lower the temperature for sampling (compared to the default of  1.0 )?
(2) What happens when the temperature is very high and why?
(3) What happens when the temperature is very low and why?

Answers:
**(1)**

Yuval: Compared to the default of 1.0, we lower the temperature for sampling in order to allow to model to choose a “better” word (from a score/value perspective) – by lowering the temperature we increase the difference between different probabilities for different words, therefore allowing a better choice of the model.

Tomer: Lower temperature results a less uniform distribution of next likely letters. Therefore the model is likely  to pick letters which will create coherent words and grammatically correct sentences which closely resemble data it previously encountered during its training phase.

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
    hypers['x_sigma2'] = 0.0002
    hypers['learn_rate'] = 0.0004
    hypers['betas'] = (0.5, 0.599)
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
            betas=(0.0, 0.0)
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            betas=(0.0, 0.0)
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 128
    hypers['z_dim'] = 3
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.2
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['lr'] = 0.0002
    hypers['discriminator_optimizer']['betas'] = (0.5, 0.999)
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

In the case of GANs, we want to keep the gradients during training for the generator and discriminator separately because they are trained separately. When training the generator we maintain gradients of the generator and not the discriminator and vice versa.
"""

part3_q2 = r"""
**Your answer:**

**(1)**	*NO*. The entire model loss depends on both the generator loss and the discriminator loss and thus the loss might still be too high even if the generator loss went under some threshold. For example if the discriminator loss is extremely high, it could cause the generator loss to be low even though the images that it generates are bad.

**(2)**	It could mean that generator’s learning speed is higher than the discriminator’s and that the discriminator is not learning fast enough, to keep up with the generator’s learning pace.
"""

part3_q3 = r"""
**Your answer:**

The main difference between the images generated from the two models seems to be the colors and the features learned by each model. It seems that the VAE model results have more details in the face area, while the GAN results have more details in the body area and the colors of the background. Because the architectures are different it seems that the features learned are also different. it could be the case that the features learned to fool the discriminator are not necessarily the sharpnes of the face, thus causing the images to appear blurry in that area. It also seems that the VAE model did a better job in learning features that allow us (as humans) to recognize the image as George Bush.
"""

# ==============
