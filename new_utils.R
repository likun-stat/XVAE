# Function to visualize knots and the coverage of Wendland basis functions
# Inputs:
#   knots: A dataframe or matrix with knot coordinates (columns 'x' and 'y')
#   stations: A dataframe or matrix with station coordinates (columns 'x' and 'y')
#   r: The radius of influence for the Wendland basis functions
#   W: A matrix of weights from the Wendland basis functions, where rows correspond to stations 
#      and columns correspond to knots
#   select: A vector of indices specifying knots whose coverage will be visualized
# Output:
#   A ggplot object visualizing:
#   - Knots as red '+' markers
#   - Coverage regions as circles
#   - Stations influenced by selected knots in different colors (blue, green, yellow)

visualize_knots <- function(knots, stations, r, W, select = c(1, 12, 10)) {
  # Step 1: Generate circular paths for each knot's coverage
  # Create the first circle based on the first knot
  dat <- cbind(circleFun(unlist(knots[1, ]), diameter = r * 2, npoints = 100), group = 1)
  colnames(knots) <- c('x', 'y')
  
  # Add circles for the remaining knots
  for (iter in 2:nrow(knots)) {
    dat <- rbind(dat, cbind(circleFun(unlist(knots[iter, ]), diameter = r * 2, npoints = 100), group = iter))
  }
  
  # Step 2: Load ggplot2 library for visualization
  library(ggplot2)
  
  # Step 3: Create the plot
  fig <- ggplot(knots) +
    # Plot knots as red '+' symbols
    geom_point(aes(x = x, y = y), shape = '+', size = 6, color = 'red') +
    # Plot circular coverage regions
    geom_path(data = dat, aes(x = x, y = y, group = group)) +
    # Plot stations influenced by the first selected knot in blue
    geom_point(
      data = stations[which(W[, select[1]] > 0.001), ],
      aes(x = x, y = y),
      colour = scales::alpha("blue", 0.1)
    ) +
    # Plot stations influenced by the second selected knot in green
    geom_point(
      data = stations[which(W[, select[2]] > 0.001), ],
      aes(x = x, y = y),
      colour = scales::alpha("green", 0.1)
    ) +
    # Plot stations influenced by the third selected knot in yellow
    geom_point(
      data = stations[which(W[, select[3]] > 0.001), ],
      aes(x = x, y = y),
      colour = scales::alpha("yellow", 0.1)
    ) +
    # Plot stations with missing data (NA weights) in black
    geom_point(
      data = stations[which(apply(W, 1, function(x) any(is.na(x)))), ],
      aes(x = x, y = y),
      colour = "black"
    )
  
  # Step 4: Return the ggplot object
  fig
}
