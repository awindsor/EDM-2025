# Early version, produces same results as other Run all, without joined plot ans AKT

library(data.table); library(readr); library(dplyr); library(LKT); library(boot); library(bit64); library(pROC); library(elo)


#BKT function definition

# Define training arguments
trainingargs <- "-s 1.3.1 -m 1 -p 1 -e 0.0000001 -i 1000"

train_and_predict <- function(traindata, testdata, trainingargs) {
  traindata <- traindata[, .(
    observation = ifelse(CF..ansbin. == 1, 1L, 2L),     # 1 for correct, 2 for incorrect
    student     = Anon.Student.Id,
    problem     = gsub("\\s+", "_", `KC..Default.`), 
    skill       = gsub("\\s+", "_", `KC..Default.`)
  )]
  testdata <- testdata[, .(
    observation = ifelse(CF..ansbin. == 1, 1L, 2L),     # 1 for correct, 2 for incorrect
    student     = Anon.Student.Id,
    problem     = gsub("\\s+", "_", `KC..Default.`), 
    skill       = gsub("\\s+", "_", `KC..Default.`)
  )]
  # Define output directories and paths
  windows_dir <- "C:/Users/ppavl/OneDrive/Active projects/hmm-scalable"
  linux_dir <- "/mnt/c/Users/ppavl/OneDrive/Active projects/hmm-scalable"
  # Define file paths
  train_file <- file.path(windows_dir, "train_data.txt")
  test_file <- file.path(windows_dir, "test_data.txt")
  model_file <- file.path(linux_dir, "model.txt")
  predict_file <- file.path(linux_dir, "predict.txt")
  # Write traindata and testdata to files
  fwrite(
    traindata,
    file = train_file,
    sep = "\t",
    quote = FALSE,
    col.names = FALSE  )
  fwrite(
    testdata,
    file = test_file,
    sep = "\t",
    quote = FALSE,
    col.names = FALSE  )
  # Construct the training command
  train_cmd <- paste(
    "'/mnt/c/Users/ppavl/OneDrive/Active projects/hmm-scalable/trainhmm'",
    trainingargs,
    shQuote(file.path(linux_dir, "train_data.txt")),
    shQuote(model_file),
    shQuote(predict_file)  )
  # Run the training command
  train_result <- system2("wsl", args = c("bash", "-c", shQuote(train_cmd)), stdout = TRUE, stderr = TRUE)
  # Print the training output
  #cat(train_result, sep = "\n")
  # Construct the prediction command

  predict_cmd <- paste(
    "'/mnt/c/Users/ppavl/OneDrive/Active projects/hmm-scalable/predicthmm'",
    "-p 1",
    shQuote(file.path(linux_dir, "test_data.txt")),
    shQuote(model_file),
    shQuote(predict_file)  )
  # Run the prediction command
  predict_result <- system2("wsl", args = c("bash", "-c", shQuote(predict_cmd)), stdout = TRUE, stderr = TRUE)
  ## Print the prediction output
  cat(predict_result, sep = "\n")
  # Read the predictions
  predictions <- fread(file.path(windows_dir, "predict.txt"), header = FALSE)
  predictions <- predictions[[1]]
  test_actual <- ifelse(testdata$observation == 1L, 1L, 0L)
  return(list(predictions = predictions, test_actual,train_result,predict_result))
}


# Define the datasets
datasets <- list(  val,val2,val3)

# Initialize a list to store results for each dataset
results_list <- list()

# Loop over each dataset
for (dataset_index in 1:3) {


all_data <- setDT(datasets[[dataset_index]][order(datasets[[dataset_index]]$CF..Time.),])
all_data[, fold := cut(.I, breaks = 100, labels = 1:100)]
res1 <- res2 <- res3 <-res4 <-res5<-res6 <-res7<-res8<-res9<-res10 <- data.frame(RMSE = numeric(), LL = numeric(), N = numeric(), AUC = numeric())
train_pcts = 30

for (i in 1:train_pcts) {
  print(i)
  
  
  
  # BKT section
  # Prepare training and testing data
  train_indices <- which(all_data$fold %in% (1:i))
  test_indices <- which(all_data$fold %in% ((i + 1):min(100, i + 70)))
  
  traindata <- all_data[train_indices]
  testdata <- all_data[test_indices]
  
  # Run the train_and_predict function
  model10_results <- train_and_predict(traindata, testdata, trainingargs)
  
  # Extract predictions and actual values
  predictions <- model10_results[[1]]
  actual <- model10_results[[2]]
  
  # Calculate performance metrics
  rmse <- sqrt(mean((actual - predictions)^2))
  log_loss <- -mean(actual * log(predictions) + (1 - actual) * log(1 - predictions))
  n <- length(actual)
  auc <- as.numeric(roc(actual, predictions, quiet = TRUE)$auc)
  
  # Append results for Model 10
  res10 <- rbind(res10, c(rmse, log_loss, n, auc))
  
  # Model AFM (fixed effects)
  modelob1 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("intercept", "intercept", "lineafm$"))
  pred1 <- pmin(pmax(inv.logit(as.matrix(modelob1$predictors %*% modelob1$coefs)[, ]), 1e-5), 0.99999)[modelob1$newdata$fold %in% (i+1):min(100, i+70)]
  actual1 <- modelob1$newdata$CF..ansbin.[modelob1$newdata$fold %in% (i+1):min(100, i+70)]
  res1 <- rbind(res1, c(sqrt(mean((actual1 - pred1)^2)), -mean(actual1 * log(pred1) + (1 - actual1) * log(1 - pred1)), length(actual1), as.numeric(roc(actual1, pred1, quiet = TRUE)$auc)))
  
  # Model 2 AFM (with logitdec instead of student int)
  modelob2 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("logitdec", "intercept", "lineafm$"), fixedpars = c(0.98))
  pred2 <- pmin(pmax(inv.logit(as.matrix(modelob2$predictors %*% modelob2$coefs)[, ]), 1e-5), 0.99999)[modelob2$newdata$fold %in% (i+1):min(100, i+70)]
  actual2 <- modelob2$newdata$CF..ansbin.[modelob2$newdata$fold %in% (i+1):min(100, i+70)]
  res2 <- rbind(res2, c(sqrt(mean((actual2 - pred2)^2)), -mean(actual2 * log(pred2) + (1 - actual2) * log(1 - pred2)), length(actual2), as.numeric(roc(actual2, pred2, quiet = TRUE)$auc)))

  # Model 3 AFM (with logitdec evol instead of KC int)
  modelob3 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("intercept", "logitdecevol", "lineafm$"), fixedpars = c(.98))
  pred3 <- pmin(pmax(inv.logit(as.matrix(modelob3$predictors %*% modelob3$coefs)[, ]), 1e-5), 0.99999)[modelob3$newdata$fold %in% (i+1):min(100, i+70)]
  actual3 <- modelob3$newdata$CF..ansbin.[modelob3$newdata$fold %in% (i+1):min(100, i+70)]
  res3 <- rbind(res3, c(sqrt(mean((actual3 - pred3)^2)), -mean(actual3 * log(pred3) + (1 - actual3) * log(1 - pred3)), length(actual3), as.numeric(roc(actual3, pred3, quiet = TRUE)$auc)))

  # Model 4 AFM (with logitdec student and logitdec KC instead of ints)
  modelob4 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("logitdec", "logitdecevol", "lineafm$"), fixedpars = c(.98, .98))
  pred4 <- pmin(pmax(inv.logit(as.matrix(modelob4$predictors %*% modelob4$coefs)[, ]), 1e-5), 0.99999)[modelob4$newdata$fold %in% (i+1):min(100, i+70)]
  actual4 <- modelob4$newdata$CF..ansbin.[modelob4$newdata$fold %in% (i+1):min(100, i+70)]
  res4 <- rbind(res4, c(sqrt(mean((actual4 - pred4)^2)), -mean(actual4 * log(pred4) + (1 - actual4) * log(1 - pred4)), length(actual4), as.numeric(roc(actual4, pred4, quiet = TRUE)$auc)))
  
  
  # Model 6 Logitdec adapt
  modelob6 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("logitdec", "logitdecevol", "logsuc"), fixedpars = c(.98, .98))
  pred6 <- pmin(pmax(inv.logit(as.matrix(modelob6$predictors %*% modelob6$coefs)[, ]), 1e-5), 0.99999)[modelob6$newdata$fold %in% (i+1):min(100, i+70)]
  actual6 <- modelob6$newdata$CF..ansbin.[modelob6$newdata$fold %in% (i+1):min(100, i+70)]
  res6 <- rbind(res6, c(sqrt(mean((actual6 - pred6)^2)), -mean(actual6 * log(pred6) + (1 - actual6) * log(1 - pred6)), length(actual6), as.numeric(roc(actual6, pred6, quiet = TRUE)$auc)))
  
  

  
  modelob9 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11,
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default.", "KC..Default."),
                  features = c("logitdec", "logitdecevol", "logsuc","recency"), fixedpars = c(.98,.98,.25))
  pred9 <- pmin(pmax(inv.logit(as.matrix(modelob9$predictors %*% modelob9$coefs)[, ]), 1e-5), 0.99999)[modelob9$newdata$fold %in% (i+1):min(100, i+70)]
  actual9 <- modelob9$newdata$CF..ansbin.[modelob9$newdata$fold %in% (i+1):min(100, i+70)]
  res9 <- rbind(res9, c(sqrt(mean((actual9 - pred9)^2)),
                        -mean(actual9 * log(pred9) + (1 - actual9) * log(1 - pred9)),
                        length(actual9),
                        as.numeric(roc(actual9, pred9, quiet = TRUE)$auc)))
  
  
  
  
}

#Compute Elo
#This code finds K in training % of the data using the fold ID, then runs on rest with that K
#Test RMSE is only from folds after K is chosen

auc_test = rep(NA,train_pcts)
rmse_test = rep(NA,train_pcts)
K_options = c(.001,.1,.5,seq(1,100,1)) #Somewhat arbitrary range of choices, but about 10 seems usually good
chosen_k = c()
constant_k = FALSE
for(i in 1:train_pcts){
  print(i)
  train_idx = which(as.numeric(all_data$fold)<=i)
  test_idx = which(all_data$fold %in% c((i+1):(i+70)))
  if(constant_k==TRUE){
    k_chosen=15
  }else{
    dataK = all_data[train_idx,]
    auc_k = c()
    for(j in 1:length(K_options)){
      elo_fit <- elo.run(CF..ansbin. ~ Anon.Student.Id + KC..Default., data = dataK,
                         k = K_options[j])#, initial.elos = 0)
      
      auc_k[j] = suppressMessages(auc(dataK$CF..ansbin.,predict(elo_fit,dataK)))
    }
    k_chosen = K_options[which.max(auc_k)]
    chosen_k[i] = k_chosen
  }
  data3 = rbind(all_data[train_idx,],all_data[test_idx,])
  len3=dim(data3)[1]
  elo_fit <- elo.run(CF..ansbin. ~ Anon.Student.Id + KC..Default., data = data3,
                     k = k_chosen)#, initial.elos = 0)
  test_fold_idx = which(as.numeric(data3$fold)>i & as.numeric(data3$fold)<=(i+70))
  auc_test[i] = suppressMessages(auc(data3$CF..ansbin.[test_fold_idx],predict(elo_fit,data3[test_fold_idx,])))
  rmse_test[i] = suppressMessages(sqrt(mean((data3$CF..ansbin.[test_fold_idx]-predict(elo_fit,data3[test_fold_idx,]))^2)))
  print(len3)
}

# plot(auc_test,ylim=c(.65,.85),xlab="Percentage of Data Used in Training (%)",ylab="Test AUC",pch=16)
# lines(auc_test,ylim=c(.65,.85),xlab="Percentage of Data Used in Training (%)",ylab="Test AUC",pch=16,lwd=2)
# auc_test
# chosen_k
# plot(rmse_test,ylim=c(.25,.5),xlab="Percentage of Data Used in Training (%)",ylab="Test RMSE",pch=16)
# lines(rmse_test,ylim=c(.25,.5),xlab="Percentage of Data Used in Training (%)",ylab="Test RMSE",pch=16,lwd=2)
res_elo = data.frame(RMSE = rmse_test,
                     LL = NA,
                     N = NA,
                     AUC = auc_test)
# Add Model 9 to column names
colnames(res1) <- colnames(res2) <- colnames(res3) <- colnames(res4) <- colnames(res5) <- colnames(res6) <- colnames(res7) <- colnames(res8) <- colnames(res9)<- colnames(res10)  <- c("RMSE", "LL", "N", "AUC")

# Create original models list with static res references
models <- list(
  list(name = expression(int[S] + int[KC] + lineafm*"$"[KC]), res = res1, color = "blue", pch = 16),
  list(name = expression(int[S] + logitdecevol[KC] + lineafm*"$"[KC]), res = res3, color = "darkgreen", pch = 17),
  list(name = expression(logitdec[S] + int[KC] + lineafm*"$"[KC]), res = res2, color = "red", pch = 18),
  list(name = expression(logitdec[S] + logitdecevol[KC] + lineafm*"$"[KC]), res = res4, color = "purple", pch = 19),
  list(name = expression(logitdec[S] + logitdecevol[KC] + logsuc[KC]), res = res6, color = "brown", pch = 15),
  list(name = expression(logitdec[S] + logitdecevol[KC] + logsuc[KC] + recency[KC]), res = res9, color = "black", pch = 3),
  list(name = "Elo", res = res_elo, color = "gray", pch = 8),
  list(name = "BKT", res = res10, color = "gold3", pch = 4)
)


# Calculate mean AUC and sort models
for (i in seq_along(models)) {
  models[[i]]$mean_auc <- mean(models[[i]]$res$AUC)
}
models <- models[order(-sapply(models, function(x) x$mean_auc))]

# Calculate dynamic y-limits
rmse_ylim <- range(unlist(lapply(models, function(m) m$res$RMSE)))
auc_ylim <- range(unlist(lapply(models, function(m) m$res$AUC)))

# Generate filenames
rmse_filename <- paste0("RMSE_plot_dataset_", dataset_index, ".png")
auc_filename <- paste0("AUC_plot_dataset_", dataset_index, ".png")

# RMSE Plot
png(rmse_filename, width = 3.3 * 300, height = 3 * 300, res = 300)
par(mar = c(3, 3, 1, 1))
plot(models[[1]]$res$RMSE, type = "o", pch = 16, col = models[[1]]$color, 
     ylim = rmse_ylim, xlab = "Expanding Window (% Data Used for Training)", ylab = "RMSE Next 70% of Data", 
     cex.lab = 0.9,  # Reduce axis label font size
     cex.axis = 0.8, # Reduce tick label font size
     mgp = c(1.5, 0.5, 0))
for (i in 2:length(models)) {
  lines(models[[i]]$res$RMSE, type = "o", pch = 16, col = models[[i]]$color)
}
legend("topright", legend = sapply(models, function(m) m$name), 
       col = sapply(models, function(m) m$color), pch = 16, lty = 1, cex = 0.7)
dev.off()

# AUC Plot
png(auc_filename, width = 6.5 * 300, height = 3 * 300, res = 300)
par(mar = c(3, 3, 1, 1))
plot(models[[1]]$res$AUC, type = "o", pch = 16, col = models[[1]]$color,
     ylim = auc_ylim, xlab = "Expanding Window (% Data Used for Training)", ylab = "AUC Next 70% of Data", 
     cex.lab = 0.9,  # Reduce axis label font size
     cex.axis = 0.8, # Reduce tick label font size
     mgp = c(1.5, 0.5, 0))
for (i in 2:length(models)) {
  lines(models[[i]]$res$AUC, type = "o", pch = 16, col = models[[i]]$color)
}
legend("bottomright", legend = sapply(models, function(m) m$name), 
       col = sapply(models, function(m) m$color), pch = 16, lty = 1, cex = 0.7)
dev.off()


}
