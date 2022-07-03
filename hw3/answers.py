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

The $\sigma^2$ hyperparameter is a parameter known as the *likelihood variance*, and it is the variance of the parametric likelihood distribution $p_{\bb{\beta}}(\bb{X}|\bb{Z}=\bb{z})=\mathcal{N}(\Psi_{\bb{\beta}}(\bb{z}),\sigma^{2}\bb{I})$. It is from this distribution that the decoder of the VAE is supposed to sample for a new instance $x$ given some latent $z$, however in our implemenetation we did not conduct our sampling for the decoder in this way: instead of sampling from $\mathcal{N}(\Psi_{\bb{\beta}}(\bb{z}),\sigma^{2}\bb{I})$, we just took the mean $\Psi(z)$. Where $\sigma^2$ actually came into play for us was during the loss calculation step. The loss of our VAE was calculated according to the following formula:
$$
\ell(\vec{\alpha},\vec{\beta};\bb{x}) =
\frac{1}{\sigma^{2} d_{x}} \left\| \bb{x} - \Psi_{\bb{\beta}}\left(  \bb{\mu}_{\bb{\alpha}}(\bb{x})  +
\bb{\Sigma}^{\frac{1}{2}}_{\bb{\alpha}}(\bb{x}) \bb{u}   \right) \right\|_{2}^{2} +
\mathrm{tr}\,\bb{\Sigma}_{\bb{\alpha}}(\bb{x}) +  \|\bb{\mu}_{\bb{\alpha}}(\bb{x})\|^{2}_{2} - d_{z} - \log\det \bb{\Sigma}_{\bb{\alpha}}(\bb{x})
$$
We see that $\sigma^2$ appears in the denominator of the first term (dubbed the *data-reconstruction loss*), and does not appear in the rest of the terms (dubbed the *KL-divergence loss*). Thus, $\sigma^2$ serves as a *regularization parameter* for the data-fitting term: decreasing its value will cause the data-reconstruction loss to become dominant, while increasing it will cause the KL-divergence loss to become dominant.

Setting the $\sigma^2$ to be too low (i.e. $\sigma^2=10^{-6}$) prevents the model from being able to formulate recognizable human faces, instead producing blurry (yet colorful) images that lack much structure. Setting the $\sigma^2$ too high (i.e. $\sigma^2=10^{6}$) on the other hand allowed the model to produce images that were easier to recognize as human faces (though still blurry and lacking some clear features such as eyebrows or hair), however there was little to no variation between any of the images the model generated. It's as if the model found one instance that nicely approximated all of the images in the training set and just focused on that, rather than trying to express the same variaty as the original data (i.e. different poses, colors).

For our model, we chose $\sigma^{2}=2\cdot10^{-4}$ since this value allowed the model to produce images with a good variety of colors and poses, while still maintaining the identifiable structure of human faces.
"""

part2_q2 = r"""
**Your answer:**

**(1)** The *data-reconstruction loss* is represented by the following term: $\frac{1}{\sigma^{2} d_{x}} \|\bb{x} - \Psi_{\bb{\beta}}(\bb{\mu}_{\bb{\alpha}}(\bb{x}) + \bb{\Sigma}^{\frac{1}{2}}_{\bb{\alpha}}(\bb{x})\bb{u})\|_{2}^{2}$. This is a classic likelihood ($p(\bb{X}|\bb{Z})$) loss term for measuring the fit of our model by directly comparing the training set $x$ to the data generated by our VAE, $\tilde{x}=\Psi_{\bb{\beta}}(\bb{\mu}_{\bb{\alpha}}(\bb{x}) + \bb{\Sigma}^{\frac{1}{2}}_{\bb{\alpha}}(\bb{x})\bb{u})$. The norm term $\|\bb{x} - \tilde{x}\|_{2}^{2}$ is the *mean squared error* (MSE) of the generated data $\tilde{x}$, while the $\sigma^2$ term acts as a regularization parameter for this MSE (we elaborated upon this in part (1)). Thus, the data-reconstruction loss tells us how accurately our VAE model is able to generate new data that is similar to the training set.

The *Kullback-Leibler (KL) divergence loss* is represented by the following term: $\mathrm{tr}\,\bb{\Sigma}_{\bb{\alpha}}(\bb{x}) +  \|\bb{\mu}_{\bb{\alpha}}(\bb{x})\|^{2}_{2}-d_{z}-\log\det\bb{\Sigma}_{\bb{\alpha}}(\bb{x})$. It measures the divergence of one probability distribution from another, and we implement it in the VAE loss in order to compare the approximate posterior probability distribution $q_{\bb{\alpha}}(\bb{Z}|\bb{X}) = \mathcal{N}(\bb{\mu}_{\bb{\alpha}}(\bb{x}),\mathrm{diag}\{\bb{\sigma}^{2}_{\bb{\alpha}}(\bb{x})\})$ to the real posterior distribution $p(\bb{Z}|\bb{X})$. It directly assesses the performance of our encoder, which is tasked with approximating $q_{\bb{\alpha}}(\bb{Z}|\bb{X})$, and we can think of it as telling us how much information we "lost" by using the approximation $q_{\bb{\alpha}}(\bb{Z}|\bb{X})$ instead of the real distribution $p(\bb{Z}|\bb{X})$ in our model.

**(2)** The latent-space distribution $p(\bb{Z})$ is essentially what we are approximating in the encoder, and the KL-divergence loss gives us an indication of how good this approximation is. Since we want to increase our *evidence lower bound* (ELBO) and the KL-divergence is always positive, we will seek to minimize the KL-divergence loss term (i.e. get it as close to zero as possible) in order to ensure that our model is correctly modelling the posterior distribution as well as the evidence distribution. 

**(3)** By minimizing the KL-divergence loss, we are ensuring that our approximation of the posterior distribution $p(\bb{Z}|\bb{X})$ is accurate while also increasing the lower bound on the evidence distribution $p(\bb{X})$ - as will be explained in question (3), we want to maximize the evidence distribution, and so increasing its lower bound is ideal.
"""

part2_q3 = r"""
**Your answer:**

We want to maximize the evidence distribution $p(\bb{X})$ in order to allow our model to learn from the training set more closely and produce new instances that are close to the original instance space. We are improving our model's ability to generate new images that look like the original ones. This process was visible during the training step, where the images generated by the VAE went from almost solid blocks of color in the beginning that couldn't possibly be recognized as containing the structure of a human face, to images that could be more easily identified as containing human faces within a relatively small number of epochs (usually between 30 to 40).
"""

part2_q4 = r"""
**Your answer:**

Modelling the logarithm of the latent-space variance $\bb{\sigma}^{2}_{\bb{\alpha}}$, rather than directly modelling the variance itself, was a consideration we made mainly to mitigate the effects of numerical errors and instabilities during our computations, as well as to make the training process easier.

The latent-space variance is always a positive number, and is usually pretty small (often dwelling in the regime $(0,1]$). If we modelled the variance directly, we'd have to ensure that the output of our network is always positive using an activation such as ReLU. This increases the complexity of our model, and in the case of ReLU can muck up our gradient calculations since we are in a region near zero where the gradient might not be well defined (and thus introduces numerical instabilities). Also, since the variance is usually small, there can be numerical instablities caused by floating point arithmetic with very small numbers.

Modelling the logarithm of the variance instead helps us treat these problems, since we are mapping from a region of small, positive numbers to a large region containing both positive and negative values ($-\infty < log(\sigma^{2}) < \infty$). Thus, we obtain more numerical stability and ease of training for our model, while minimizing numerical errors.
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
