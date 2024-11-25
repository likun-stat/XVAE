#' Example Data for XVAE Modeling and Emulation
#'
#' This dataset contains two components: `X`, a matrix of simulated data using the Model 
#' III setting in Zhang et al., and `stations`, a vector of station coodinates. The dataset
#'  is useful for testing and demonstrating XVAE and other relavant emulators.
#'
#' @format A list containing two named elements:
#' \describe{
#'   \item{\code{X}}{A matrix of simulated data points, used as input for XVAE.}
#'   \item{\code{stations}}{A character vector of station coordinates corresponding to the rows in \code{X}.}
#' }
#'
#' @details 
#' This dataset is intended for use in testing and demonstrating the XVAE 
#' approaches. The matrix \code{X} contains simulated data, and \code{stations} provides 
#' the coordinates of stations from which the data were simulated. 
#' The dataset can be used for basic exploration, analysis, or to benchmark XVAE.
#'
#' @examples
#' # Load the example dataset
#' data(example_X)
#'
#' # Explore the simulated data matrix
#' str(X)
#'
#' # Examine the station names
#' head(stations)
"example_X"


#' Simulated and Generated Copulas
#'
#' This dataset contains simulated and generated data using various methods in their uniform
#' scale. The data includes grids for simulated data, XVAE-generated data, 
#' and GAN-generated data.
#'
#' @format A list containing three named elements, each represented as a matrix:
#' \describe{
#'   \item{\code{U_sim_grid}}{A matrix containing simulated data points based on copula models.}
#'   \item{\code{U_xvae_grid}}{A matrix containing data generated using an XVAE (Variational Autoencoder) approach.}
#'   \item{\code{U_gan_grid}}{A matrix containing data generated using a GAN (Generative Adversarial Network) approach.}
#' }
#'
#' @details 
#' This dataset can be used to compare the performance and characteristics of 
#' different data generation methods in the context of copula modeling. 
#' Each grid provides a set of points useful for analysis, visualization, or 
#' methodological evaluation.
#'
#' @examples
#' # Load the data
#' data(copulas)
#'
#' # Explore the simulated data
#' str(U_sim_grid)
#'
#' # Compare data from different methods
#' summary(U_xvae_grid)
#' summary(U_gan_grid)
"copulas"

