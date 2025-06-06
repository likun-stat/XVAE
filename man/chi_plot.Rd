% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/XVAE_utils.R
\name{chi_plot}
\alias{chi_plot}
\title{Chi-Plot for Spatial Dependence}
\usage{
chi_plot(
  X,
  stations,
  emulation,
  distance,
  tol = 0.001,
  u_vec = c(seq(0, 0.98, 0.01), seq(0.9801, 0.9999, 1e-04)),
  L = 25,
  ylab = expression(chi[u]),
  uniform = FALSE,
  legend = TRUE,
  show.axis.y = TRUE,
  duplicated = TRUE,
  alpha = 1/2
)
}
\arguments{
\item{X}{A matrix of simulated values at spatial locations.}

\item{stations}{A data frame with spatial coordinates (\code{x}, \code{y}) of the stations.}

\item{emulation}{A matrix of emulated values at spatial locations.}

\item{distance}{Distance for selecting pairs of spatial points for comparison.}

\item{tol}{Tolerance for distance selection (default: \code{0.001}).}

\item{u_vec}{A vector of quantile thresholds for chi calculations (default: \code{c(seq(0,0.98,0.01),seq(0.9801,0.9999,0.0001))}).}

\item{L}{Number of levels for marginal transformation (default: \code{25}).}

\item{ylab}{Label for the y-axis (default: \code{expression(chi[u])}).}

\item{uniform}{Logical; whether to use uniform marginals (default: \code{FALSE}).}

\item{legend}{Logical; whether to display a legend (default: \code{TRUE}).}

\item{show.axis.y}{Logical; whether to display the y-axis labels (default: \code{TRUE}).}

\item{duplicated}{Logical; whether to retain duplicate pairs of points (default: \code{TRUE}).}
}
\value{
A ggplot object representing the chi-plot.
}
\description{
This function generates a chi-plot for evaluating spatial dependence in simulated and emulated data at specified distances.
}
\examples{
# Example data
X <- matrix(rnorm(1000), ncol=10)
emulation <- matrix(rnorm(1000), ncol=10)
stations <- data.frame(x = runif(100), y = runif(100))
chi_plot(X, stations, emulation, distance=0.5)
}
