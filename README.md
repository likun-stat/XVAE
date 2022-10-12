# extCVAE

Emulating complex climate models via integrating variational autoencoder and spatial extremes

>Department of Statistics, University of Missouri


### Max-infinitely divisible processes

Our model is based on the max-infinitely divisible process proposed by Bopp et al. (2021)[[1]](#1) and allows for both short-range asymptotic independence and dependence along with long-range asymptotic independence, which can be specified as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.image?X_t(\textbf{s})=\epsilon_t(\textbf{s})Y_t(\textbf{s}),&space;" />
</p>

where $X_t(s)$ is a spatio-temporal output from a simulator (e.g., high-resolution climate model),  $\epsilon_t(s)$ is a white noise process with independent $1/\alpha$-Fréchet marginal distribution, and $Y_t(s)$ is described by a low-rank representation:

<p align="center">
<img src="https://latex.codecogs.com/svg.image?Y_t(\textbf{s})=\left(\sum_{k=1}^K&space;\omega_k(\textbf{s},&space;r_k)^{1/\alpha}Z_{kt}\right)^\alpha.&space;" />
</p>

Here, we use compactly-supported Wendland basis functions $\omega_k(s, r_k)$, $k=1,\ldots,K$, which are centered at $K$ pre-specified knots. The latent variables are lighter-tailed, exponentially tilted, positive-stable variables:

<p align="center">
<img src="https://latex.codecogs.com/svg.image?Z_{kt}\sim&space;H(\alpha,\alpha,\theta_k),\;&space;k=1,\ldots,&space;K,&space;" />
</p>

in which $\alpha\in (0,1)$ is the concentration parameter, and larger values of the tail index $\theta_k\geq 0$ induce lighter-tailed distributions of $Z_{kt}$. 

### Conditional variational autoencoder
![plot1](www/Extremes_CVAE.png)


## References
<a id="1">[1]</a> 
Gregory P Bopp, Benjamin A Shaby, and Raphaël Huser. A hierarchical max-infinitely
divisible spatial model for extreme precipitation. _Journal of the American Statistical
Association_, 116(533):93–106, 2021.
