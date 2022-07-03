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

**(1)** Lowering the temperature for sampling causes our model to generate text based on a less uniform distribution, i.e. the more likley words/characters are boosted while the less likely words/characters are diminished. This encourages the model to choose more common ("better") sequences of characters and words, which are thus more likely to follow the real flow of the English language as we know it. We don't want to make the temperature too low (we'll explain why in part (3)), and so we'll choose a value less than 1.0 but not too far from it. We chose $T=0.75$ for our model, and it seemed to produce the closest thing to an "intelligible" text that we could obtain with our current model.

**(2)** When we raise the temperature too high (for example to $T=10,000$), the text we generate becomes absolute gibberish. This is because we are essentially evening out the probabilities for every character, giving them all a much mroe similar weight. Thus, the model has lost its ability to generate reasonable sentence structure and spelling, and the choice of characters is essentially random. Here the analogy to temperature in the thermodynamic sense becomes more clear: high temperature causes high entropy, meaning that there is disorder and chaos in the system. Our generated text here is super disordered and chaotic, and very little real information can be obtained from it.

**(3)** When we lower the temperature drastically (for example to $T=0.0001$), the model generates a highly repetitive text that seems to just keep printing the same line over and over, as if the model is unable to generate anything else but that line. What we have done here is essentially boosted the weights of the most likely characters, while diminishing the weights of unlikely characters. This caused the model to generate the sentence *"And the streets of the streets of the state"* over and over again.

We notice something interesting about this sentence. It contains 3 (*the, and, of*) out of the top 5 most common words in the corpus (*the, and, of, to, I*), and is only made up of 11 different letters (*n, d, h, r, f, i, o, s, a, t, e*), of which 9 are the top 9 most common letters in the corpus (*i, o, h, s, a, t, e, r, n*). Clearly, our model preferred to use (almost) only the most common words and characters in the corpus, and so this gives us another insight into how the low temperature affected the model.

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

**Question:** What does the $\sigma^2$ hyperparameter (`x_sigma2` in the code) do? Explain the effect of low and high values.

**Answer:**

Yuval 1: The x_sigma2 hyperparameter is basically a regularization parameter, that helps us control the ratio between the two terms in the loss function – these are the data loss and the KLdiv loss. Low and high values of x_sigma2 will have an effect on the loss – low values would give the dominance to the data loss, while high values would give it to the KLdiv loss. In vivo, we notice that values that are too low produce images that are similar to the origin (the input of the model), and values that are too high produce blurry images, or even lead to the model not learning. 

Yuval 2: It is the variance of our likelihood, and also serves as a weight for the data reconstruction loss, (thus controlling the "randomness" of the model). Low values of it will lead to generate photos that is closer to our data set, while high will lead to generate photos further away from the learned data set.
"""

part2_q2 = r"""
**Your answer:**

**Question:**
(1) Explain the purpose of both parts of the VAE loss term - reconstruction loss and KL divergence loss.
(2) How is the latent-space distribution affected by the KL loss term?
(3) What's the benefit of this effect?

**Answers:**

**(1)**

**(2)**

**(3)**

Yuval 1:

1. The parts of the VAE loss term are the **reconstruction loss** and the **KLdiv loss**.
The former, the reconstruction loss, is practically the loss we know from previous assignments and courses – it is the loss that implies how accurate is the model over the training samples, and we may refer to it as the “data fit” loss term that shows us if we are actually progressing during the learning process. The latter, the KLdiv loss, measurs the distance between the posterior distribution that was model and the original one – to help us make sure that they are close to one another (and allow us to sample latent variables that are actually within the original distribution, and will produce ones that are “meaningful” for the model to decode).

2. As mentioned earlier, the latent-space distribution is affected by the KLdiv loss term in such way that it helps us get the posterior distribution close to the original, and binds this distribution to the learning process itself.

3. The benefit of this effect, as stated already, is that using this KLdiv loss term allows us to sample “meaningful” latent-variables from the latent-space distribution, hence decoding these variables might actually produce data that is similar to the original.

Yuval 2:

1. 
- Reconstruction loss - A loss that reflects how well the model succeeds to reconstruct an image.
- The Kullback-Liebler Divergence loss serves as a regularization factor. It encourages the output distribution to be closer to the standard normal distribution.

2. The KL loss regularizes the mapping to the latent space, where it encourages similarity to the standard normal distribution.

3. It enforces the distribution to be more "simple", and hopefully make close points in  the latent space to output similar content.
"""

part2_q3 = r"""
**Your answer:**

**Question:** In the formulation of the VAE loss, why do we start by maximizing the evidence distribution, $p(\bb{X})$?

**Answer:**

Yuval 1: In the formulation of the VAE loss, we start by maximizing the evidence distribution, p(X), in order to allow the model to generate “fresh new” images, instead of regenerating images that are identical to those in the input (images that were previously seen). The evidence distribution, p(X), basically represents our model’s ability to generate new images that are similar to the previously seen images (during the learning process of the model). Maximizing it would help us get such new images, since we would have a probability as high as possible of generating such images.

Yuval 2: It allows our model to generate samples closer to the original samples.
"""

part2_q4 = r"""
**Your answer:**

**Question:** In the VAE encoder, why do we model the **log** of the latent-space variance corresponding to an input, $\bb{\sigma}^2_{\bb{\alpha}}$, instead of directly modelling this variance?

**Answer:** 

Yuval 1: In the VAE encoder, we model the **log** of the latent-space variance corresponding to an input instead of directly modelling the variance. That is because we are looking to minimize numerical errors and instabilities, and this allows us to have more control over this value (while scaling the sigma to a log scale helps us to that). Another reason, which is of lower importance but still exists, is that we might pass the *sigma* through some activation function (e.g. ReLU) that requires it to be positive, and with the log modelling we make sure the *sigma* itself is.

Yuval 2: We model it with log because of stability concerns - log maps [0,1] -> [-$\infty$,0]
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
