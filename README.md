# extCVAE

Department of Statistics, University of Missouri

> Emulating complex climate models via integrating variational autoencoder and spatial extremes

### Max-infinitely divisible processes

Our model is based on the max-infinitely divisible process proposed by Bopp et al. (2021) and allows for both short-range asymptotic independence and dependence along with long-range asymptotic independence, which can be specified as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.image?X_t(\textbf{s})=\epsilon_t(\textbf{s})Y_t(\textbf{s})" />
</p>

where $X_t(s)$ is a spatio-temporal output from a simulator (e.g., high-resolution climate model),  $\epsilon_t(s)$ is a white noise process with independent $1/\alpha$-Fréchet marginal distribution, and $Y_t(s)$ is described by a low-rank representation:

<p align="center">
<img src="https://latex.codecogs.com/svg.image?Y_t(\textbf{s})=\left(\sum_{k=1}^K&space;\omega_k(\textbf{s},&space;r_k)^{1/\alpha}Z_{kt}\right)^\alpha" />
</p>

Here, we use compactly-supported Wendland basis functions $\omega_k(s, r_k)$, $k=1,\ldots,K$, which are centered at $K$ pre-specified knots. The latent variables are lighter-tailed, exponentially tilted, positive-stable variables:

<p align="center">
<img src="https://latex.codecogs.com/svg.image?https://latex.codecogs.com/svg.image?Z_{kt}\sim&space;H(\alpha,\alpha,\theta_k),\;&space;k=1,\ldots,&space;K,&space;" />
</p>

in which $\alpha\in (0,1)$ is the concentration parameter, and larger values of the tail index $\theta_k\geq 0$ induce lighter-tailed distributions of $Z_{kt}$. 

### Conditional variational autoencoder
![plot1](www/Extremes_CVAE.png)
