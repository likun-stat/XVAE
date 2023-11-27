# # Edit Renviron
# usethis::edit_r_environ()
# PYTORCH_ENABLE_MPS_FALLBACK=1
library(torch)

convnet <- nn_module(
  "convnet",
  
  initialize = function() {
    # nn_conv2d(in_channels, out_channels, kernel_size)
    self$conv1 <- nn_conv2d(1, 20, 3, padding='same')
    self$conv2 <- nn_conv2d(20, 40, 3, padding='same')
    # self$conv3 <- nn_conv2d(32, 64, 3)
    
    self$output <- nn_linear(1000, 400)
    
  },
  
  forward = function(x) {
    x %>% 
      self$conv1() %>% 
      nnf_relu() %>%
      nnf_max_pool2d(2) %>%
      nnf_dropout2d(p = 0.05) %>% 
      self$conv2() %>% 
      nnf_relu() %>%
      nnf_max_pool2d(2) %>%
      nnf_dropout2d(p = 0.05) %>% 
      # self$conv3() %>% 
      # nnf_relu() %>%
      # nnf_max_pool2d(2) %>%
      torch_flatten(start_dim = 2) %>%
      self$output()
  }
)

model <- convnet()

img <- torch_randn(1, 1, 20, 20)
model(img)





##### Read in training and valid data
# setwd("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/GEV-GP_VAE/")
knots <- expand.grid(x=seq(1,20, by=1), y=seq(1,20, by=1))

input_dataset <- dataset(
  name = "input_dataset()",
  initialize = function(dat_training_torch, thetas_training_torch) {
    dat_training_torch <- na.omit(dat_training_torch)
    thetas_training_torch <- na.omit(thetas_training_torch)
    self$x <- dat_training_torch
    self$y <- thetas_training_torch
  },
  .getitem = function(i) {
    list(x = self$x[i,,, ], y = self$y[i,])
  },
  .length = function() {
    dim(self$x)[1]
  }
)




dat_train <- read.csv("./train_MEIs_PS_vars.csv")
dat_training_torch <- torch_zeros(ncol(dat_train), 1, sqrt(nrow(knots)), sqrt(nrow(knots)))
for(iter in 1:ncol(dat_train)){
  tmp_Z_vec <- matrix(log(dat_train[-1, iter]), sqrt(nrow(knots)), sqrt(nrow(knots)))
  dat_training_torch[iter,1,,] <- torch_tensor(tmp_Z_vec)
}

thetas_train <- read.csv("./train_Thetas_PS_vars.csv")
thetas_training_torch <- torch_zeros(ncol(dat_train), nrow(knots))
for(iter in 1:ncol(dat_train)){
  thetas_training_torch[iter,] <- torch_tensor(thetas_train[,iter])
}
Train_dl <- input_dataset(dat_training_torch, thetas_training_torch)
length(Train_dl)
Train_dl[1]



dat_valid <- read.csv("./valid_MEIs_PS_vars.csv")
dat_valid_torch <- torch_zeros(ncol(dat_valid), 1, sqrt(nrow(knots)), sqrt(nrow(knots)))
for(iter in 1:ncol(dat_valid)){
  tmp_Z_vec <- matrix(log(dat_valid[-1, iter]), sqrt(nrow(knots)), sqrt(nrow(knots)))
  dat_valid_torch[iter,1,,] <- torch_tensor(tmp_Z_vec)
}

thetas_valid <- read.csv("./valid_Thetas_PS_vars.csv")
thetas_valid_torch <- torch_zeros(ncol(dat_valid), nrow(knots))
for(iter in 1:ncol(dat_valid)){
  thetas_valid_torch[iter,] <- torch_tensor(thetas_valid[,iter])
}

Valid_dl <- input_dataset(dat_valid_torch, thetas_valid_torch)


##### Train the CNN
library(luz)
## Echo every other 100 epochs
print_callback <- luz_callback(
  name = "print_callback",
  on_epoch_end = function() {
    if(ctx$epoch %%100 == 0) cat("Epoch", ctx$epoch, "\n")
  }
)


fitted <- convnet %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_mse()
    )
  ) %>%
  fit(Train_dl,
      epochs = 10000,
      valid_data = Valid_dl,
      verbose = FALSE,
      callbacks = list(print_callback())
  )




##### Prediction
dat_test <- read.csv("./test_MEIs_PS_vars.csv")
dat_test_torch <- torch_zeros(ncol(dat_test), 1, sqrt(nrow(knots)), sqrt(nrow(knots)))
for(iter in 1:ncol(dat_test)){
  tmp_Z_vec <- matrix(log(dat_test[-1, iter]), sqrt(nrow(knots)), sqrt(nrow(knots)))
  dat_test_torch[iter,1,,] <- torch_tensor(tmp_Z_vec)
}

thetas_test <- read.csv("./test_Thetas_PS_vars.csv")
thetas_test_torch <- torch_zeros(ncol(dat_test), nrow(knots))
for(iter in 1:ncol(dat_test)){
  thetas_test_torch[iter,] <- torch_tensor(thetas_test[,iter])
}

test_dl <- input_dataset(dat_test_torch, thetas_test_torch)
preds <- fitted %>% predict(test_dl)
# fitted %>% evaluate(data = test_dl)
preds_cpu <- as_array(preds$to(device = "cpu"))

library(ggplot2)
cols <- rev(RColorBrewer::brewer.pal(n=9, "RdBu"))
which.rep <- 100
ggplot(knots) + geom_raster(aes(x=x, y=y, fill=preds_cpu[which.rep,])) +
  scale_fill_gradientn(colours = cols, name="predicted") + ggtitle(paste("Test Time #:", which.rep)) +
  theme(plot.title = element_text(face="bold", hjust=0.5))

ggplot(knots) + geom_raster(aes(x=x, y=y, fill=thetas_test[,which.rep])) +
     scale_fill_gradientn(colours = cols, name="true") + ggtitle(paste("Test Time #:", which.rep)) +
  theme(plot.title = element_text(face="bold", hjust=0.5))

ggplot(knots) + geom_raster(aes(x=x, y=y, fill=preds_cpu[which.rep,] - thetas_test[,which.rep])) +
  scale_fill_gradientn(colours = cols, name="pred - true") + ggtitle(paste("Test Time #:", which.rep)) +
  theme(plot.title = element_text(face="bold", hjust=0.5))


