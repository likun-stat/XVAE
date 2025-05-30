% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/XVAE_utils.R
\name{single_rejection_sampler}
\alias{single_rejection_sampler}
\title{Single Rejection Sampler}
\usage{
single_rejection_sampler(theta = theta, alpha = 1/2)
}
\arguments{
\item{theta}{A numeric parameter influencing the sampling process.}

\item{alpha}{A numeric value, typically between 0 and 1}
}
\description{
A simpler rejection sampling method for generating samples from distributions controlled by \code{alpha} and \code{theta}.
It uses inverse gamma and stable distribution sampling depending on the value of \code{alpha}.
}
