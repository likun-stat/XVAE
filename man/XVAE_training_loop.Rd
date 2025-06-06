% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/XVAE_utils.R
\name{XVAE_training_loop}
\alias{XVAE_training_loop}
\title{XVAE Training Loop}
\usage{
XVAE_training_loop(alpha, nEpoch, learning_rate, alpha_v)
}
\description{
This function performs training for XVAE.
It implements forward passes, evidence lower bound (ELBO) computation, and
parameter updates using backpropagation.
}
\details{
The \code{XVAE_training_loop} function performs the following key steps for training:
\itemize{
\item \strong{Learning Rate and Momentum Adjustment}: Dynamically adjusts the learning rate and momentum
at regular intervals based on the logarithmic scale of the epoch count.
\item \strong{Forward Pass}:
\itemize{
\item Random noise is generated for the reparameterization trick.
\item Encoders are applied to generate latent space representations for primary and auxiliary spaces.
\item Decoding is performed to reconstruct the input data.
}
\item \strong{ELBO Computation}:
\itemize{
\item Reconstruction loss: Measures the error in reconstructing input data.
\item Latent space regularization: Enforces structure in the learned latent representations.
\item Gaussian prior penalty: Regularizes the latent variables using a Gaussian prior.
}
\item \strong{Backpropagation}:
\itemize{
\item Gradients of the ELBO loss with respect to all parameters are computed.
}
\item \strong{Parameter Updates}:
\itemize{
\item Parameters are updated using momentum-based gradient descent.
\item Gradients are reset after each update.
}
}
}
\note{
The function modifies several global variables using the \verb{<<-} operator. These include parameters
(e.g., weights and biases) and intermediate variables used during the training process.
}
\examples{
# Initialize required global variables before running the function
XVAE_training_loop()

}
\seealso{
\href{https://torch.mlverse.org}{torch} for tensor operations,
\code{\link[mvtnorm:Mvnorm]{mvtnorm::rmvnorm()}} for generating multivariate normal noise.
}
