##AKT not working in this version, comment out to run and produce plot from paper

library(data.table);  library(LKT); library(boot); library(bit64); library(pROC); library(elo)
library(jsonlite)

val$KC..Unique.step.<-val$KC..Default.
val2$KC..Unique.step.<-val2$Problem.Name


run_akt_simple <- function(train_df, test_df, akt_root, epochs = 1) {
  # 1) Define unified factor levels
  uid_lvls <- unique(c(train_df$Anon.Student.Id, test_df$Anon.Student.Id))
  qid_lvls <- unique(c(train_df$`KC..Unique.step.`, test_df$`KC..Unique.step.`))
  pid_lvls <- unique(c(train_df$`KC..Default.`,   test_df$`KC..Default.`))
  
  # 2) Encode train
  dt_train <- as.data.table(train_df)[, .(
    uid       = as.integer(factor(Anon.Student.Id,   levels = uid_lvls)) - 1,
    qid       = as.integer(factor(`KC..Unique.step.`, levels = qid_lvls)) - 1,
    pid       = as.integer(factor(`KC..Default.`,     levels = pid_lvls)) - 1,
    correct   = as.integer(CF..ansbin.),
    timestamp = as.integer(CF..Time.)
  )]
  
  # 3) Encode test
  dt_test <- as.data.table(test_df)[, .(
    uid       = as.integer(factor(Anon.Student.Id,   levels = uid_lvls)) - 1,
    qid       = as.integer(factor(`KC..Unique.step.`, levels = qid_lvls)) - 1,
    pid       = as.integer(factor(`KC..Default.`,     levels = pid_lvls)) - 1,
    correct   = as.integer(CF..ansbin.),
    timestamp = as.integer(CF..Time.)
  )]
  
  # 4) Write CSVs
  out_dir <- file.path(akt_root, "testdir")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  fwrite(dt_train, file.path(out_dir, "akt_train.csv"), col.names = FALSE)
  fwrite(dt_test,  file.path(out_dir, "akt_test.csv"),  col.names = FALSE)
  
  # 5) Call Python trainer with proper status capture
  old_wd <- setwd(akt_root); on.exit(setwd(old_wd), add = TRUE)
  
  # Use a temp file to capture both stdout & stderr
  output_file <- tempfile()
  status <- system2(
    "python",
    args   = c("run_akt_minimal2.py",
               "--epochs",   as.character(epochs),
               "--data_dir",  "testdir"),
    stdout = output_file,
    stderr = output_file
  )
  
  # Check exit status
  if (status != 0) {
    cat("Python execution error:\n")
    cat(readLines(output_file), sep = "\n")
    stop("Python script execution failed with status: ", status)
  }
  
  # 6) Read back payload.json
  payload_file <- file.path(out_dir, "payload.json")
  if (!file.exists(payload_file)) {
    stop("payload.json not found in ", out_dir)
  }
  
  # 7) Parse and return results
  res <- fromJSON(payload_file)
  
  # Return only relevant metrics
  return(list(
    epoch_auc_drop = res$epoch_auc_drop,
    train_loss     = res$train_loss,
    held_auc       = res$held_auc,
    test_pred      = res$test_pred,   # numeric vector of predictions
    test_true      = res$test_true    # integer vector of truths
  ))
}

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
  res1 <- res2 <- res3 <-res4 <-res5<-res6 <-res7<-res8<-res9<-res10 <-resAKT<- data.frame(RMSE = numeric(), LL = numeric(), N = numeric(), AUC = numeric())
  train_pcts = 30
  
  for (i in (1:train_pcts)) {
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
    
    # --- AKT Integration via run_akt_simple() ---
  
    # call new wrapper with separate train / test
    akt_payload <- run_akt_simple(
      train_df = traindata,
      test_df  = testdata,
      akt_root = "C:/Users/ppavl/OneDrive - The University of Memphis/AKT",
      epochs   = 60
    )
    
    # extract and compute metrics
    truth    <- akt_payload$test_true
    preds    <- akt_payload$test_pred
    rmse     <- sqrt(mean((truth - preds)^2))
    log_loss <- -mean(truth * log(preds) + (1 - truth) * log(1 - preds))
    n        <- length(truth)
    auc      <- akt_payload$held_auc
    
    # append into resAKT
    resAKT <- rbind(
      resAKT,
      data.frame(RMSE = rmse,
                 LL   = log_loss,
                 N    = n,
                 AUC  = auc)
    )
    
    
    
    
    
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
  colnames(res1) <- colnames(res2) <- colnames(res3) <- colnames(res4) <- colnames(res5) <- colnames(res6) <- colnames(res7) <- colnames(res8) <- colnames(res9)<- colnames(res10)  <- colnames(resAKT) <- c("RMSE", "LL", "N", "AUC")
  
  library(data.table)
  
  # ----- Save model results to CSV files -----
  model_results_list <- list(
    res1 = res1, res2 = res2, res3 = res3, res4 = res4,
    res6 = res6, res9 = res9, res_elo = res_elo, res10 = res10, resAKT = resAKT
  )
  model_names <- c("AFM1", "AFM2", "AFM3", "AFM4", "LKT6", "LKT9", "Elo", "BKT", "AKT")
  
  for (i in seq_along(model_results_list)) {
    fwrite(model_results_list[[i]], file = paste0(model_names[i], "_dataset_", dataset_index, ".csv"))
  }
  
}

plot_model_performance <- function(dataset_index) {
  library(data.table)
  
  # Model meta-info
  models_info <- list(
    list(name = expression(int[S] + int[KC] + lineafm*"$"[KC]), file = "AFM1", color = "blue", pch = 16),
    list(name = expression(int[S] + logitdecevol[KC] + lineafm*"$"[KC]), file = "AFM3", color = "darkgreen", pch = 17),
    list(name = expression(logitdec[S] + int[KC] + lineafm*"$"[KC]), file = "AFM2", color = "red", pch = 18),
    list(name = expression(logitdec[S] + logitdecevol[KC] + lineafm*"$"[KC]), file = "AFM4", color = "purple", pch = 19),
    list(name = expression(logitdec[S] + logitdecevol[KC] + logsuc[KC]), file = "LKT6", color = "brown", pch = 15),
    list(name = expression(logitdec[S] + logitdecevol[KC] + logsuc[KC] + recency[KC]), file = "LKT9", color = "black", pch = 3),
    list(name = "Elo", file = "Elo", color = "gray", pch = 8),
    list(name = "BKT", file = "BKT", color = "gold3", pch = 4),
    list(name = "AKT", file = "AKT", color = "orange", pch = 20)
  )
  
  # Load results from files and build models list
  models <- lapply(models_info, function(info) {
    res <- fread(paste0(info$file, "_dataset_", dataset_index, ".csv"))
    list(name = info$name, res = res, color = info$color, pch = info$pch)
  })
  
  # Calculate mean AUC and sort models
  for (i in seq_along(models)) {
    models[[i]]$mean_auc <- mean(models[[i]]$res$AUC, na.rm = TRUE)
  }
  models <- models[order(-sapply(models, function(x) x$mean_auc))]
  
  # Dynamic y-limits
  rmse_ylim <- range(unlist(lapply(models, function(m) m$res$RMSE)), na.rm = TRUE)
  auc_ylim <- range(unlist(lapply(models, function(m) m$res$AUC)), na.rm = TRUE)
  
  # Filenames
  rmse_filename <- paste0("RMSE_plot_dataset_", dataset_index, ".png")
  auc_filename <- paste0("AUC_plot_dataset_", dataset_index, ".png")
  
  # RMSE Plot
  png(rmse_filename, width = 3.3 * 300, height = 3 * 300, res = 300)
  par(mar = c(3, 3, 1, 1))
  plot(models[[1]]$res$RMSE, type = "o", pch = models[[1]]$pch, col = models[[1]]$color,
       ylim = rmse_ylim, xlab = "Expanding Window (% Data Used for Training)", ylab = "RMSE Next 70% of Data",
       cex.lab = 0.9, cex.axis = 0.8, mgp = c(1.5, 0.5, 0))
  for (i in 2:length(models)) {
    lines(models[[i]]$res$RMSE, type = "o", pch = models[[i]]$pch, col = models[[i]]$color)
  }
  legend("topright", legend = sapply(models, function(m) m$name),
         col = sapply(models, function(m) m$color), pch = sapply(models, function(m) m$pch), lty = 1, cex = 0.7)
  dev.off()
  
  # AUC Plot
  png(auc_filename, width = 6.5 * 300, height = 3 * 300, res = 300)
  par(mar = c(3, 3, 1, 1))
  plot(models[[1]]$res$AUC, type = "o", pch = models[[1]]$pch, col = models[[1]]$color,
       ylim = auc_ylim, xlab = "Expanding Window (% Data Used for Training)", ylab = "AUC Next 70% of Data",
       cex.lab = 0.9, cex.axis = 0.8, mgp = c(1.5, 0.5, 0))
  for (i in 2:length(models)) {
    lines(models[[i]]$res$AUC, type = "o", pch = models[[i]]$pch, col = models[[i]]$color)
  }
  legend("bottomright", legend = sapply(models, function(m) m$name),
         col = sapply(models, function(m) m$color), pch = sapply(models, function(m) m$pch), lty = 1, cex = 0.7)
  dev.off()
}
  # ---- Usage: call this after your modeling loop for each dataset ----
  plot_model_performance(dataset_index = 1,exclude_files = "AKT")

  plot_model_performance(dataset_index = 2)
  plot_model_performance(dataset_index = 3)
  
  plot_model_performance_tiled <- function(dataset_indexes = 1:3, exclude_files = character(0), 
                                           rmse_output_file = "RMSE_all_datasets_tiled.png", 
                                           auc_output_file = "AUC_all_datasets_tiled.png") {
    library(data.table)
    
    dataset_titles <- c("Cloze", "MATHia", "Interleaving")
    
    models_info <- list(
      list(name = expression(int[S] + int[KC] + lineafm*"$"[KC]), file = "AFM1", color = "blue", pch = 16),
      list(name = expression(int[S] + logitdecevol[KC] + lineafm*"$"[KC]), file = "AFM3", color = "darkgreen", pch = 17),
      list(name = expression(logitdec[S] + int[KC] + lineafm*"$"[KC]), file = "AFM2", color = "red", pch = 18),
      list(name = expression(logitdec[S] + logitdecevol[KC] + lineafm*"$"[KC]), file = "AFM4", color = "purple", pch = 19),
      list(name = expression(logitdec[S] + logitdecevol[KC] + logsuc[KC]), file = "LKT6", color = "brown", pch = 15),
      list(name = expression(logitdec[S] + logitdecevol[KC] + logsuc[KC] + recency[KC]), file = "LKT9", color = "black", pch = 3),
      list(name = "Elo", file = "Elo", color = "gray", pch = 8),
      list(name = "BKT", file = "BKT", color = "gold3", pch = 4),
      list(name = "AKT", file = "AKT", color = "orange", pch = 20)
    )
    
    models_info <- models_info[!sapply(models_info, function(info) info$file %in% exclude_files)]
    if (length(models_info) == 0) stop("No models left to plot after applying exclusions.")
    
    all_data <- lapply(dataset_indexes, function(dataset_index) {
      models <- lapply(models_info, function(info) {
        file_path <- paste0(info$file, "_dataset_", dataset_index, ".csv")
        if (!file.exists(file_path)) {
          warning(paste("File not found:", file_path, "- Skipping this model for dataset", dataset_index))
          return(NULL)
        }
        res <- tryCatch({
          fread(file_path)
        }, error = function(e) {
          warning(paste("Error reading file", file_path, ":", e$message))
          return(NULL)
        })
        if (is.null(res) || nrow(res) == 0) return(NULL)
        list(name = info$name, res = res, color = info$color, pch = info$pch)
      })
      models <- Filter(Negate(is.null), models)
      for (i in seq_along(models)) {
        models[[i]]$mean_auc <- mean(models[[i]]$res$AUC, na.rm = TRUE)
      }
      models[order(-sapply(models, function(x) x$mean_auc))]
    })
    
    if (all(sapply(all_data, length) == 0)) {
      stop("No data could be loaded for any dataset. Check file availability.")
    }
    
    # RMSE plot
    png(rmse_output_file, width = 6.8 * 600, height = 9 * 600, res = 600)
    par(mfrow = c(length(dataset_indexes), 1), mar = c(5, 5, 4, 2), oma = c(0, 0, 2, 0))
    for (i in seq_along(dataset_indexes)) {
      models <- all_data[[i]]
      title <- dataset_titles[dataset_indexes[i]]
      if (length(models) == 0) {
        plot(1, type = "n", xlab = "", ylab = "", main = paste("RMSE -", title, "(No Data)"),
             cex.lab = 1.25, cex.axis = 1.1, cex.main = 1.3, mgp = c(2.5, 1, 0))
        next
      }
      rmse_ylim <- range(unlist(lapply(models, function(m) m$res$RMSE)), na.rm = TRUE)
      plot(models[[1]]$res$RMSE, type = "o", pch = models[[1]]$pch, col = models[[1]]$color,
           ylim = rmse_ylim, xlab = "Expanding Window (% Data Used for Training)", 
           ylab = "RMSE Next 70% of Data", main = title,
           cex.lab = 1.25, cex.axis = 1.1, cex.main = 1.3, mgp = c(2.5, 1, 0))
      for (j in 2:length(models)) {
        lines(models[[j]]$res$RMSE, type = "o", pch = models[[j]]$pch, col = models[[j]]$color)
      }
      legend("bottomright", legend = sapply(models, function(m) m$name),
             col = sapply(models, function(m) m$color), pch = sapply(models, function(m) m$pch), 
             lty = 1, cex = 0.9, bg = "white")
    }
    dev.off()
    message("RMSE plot saved as: ", rmse_output_file)
    
    # AUC plot
    png(auc_output_file, width = 6.8 * 600, height = 9 * 600, res = 600)
    par(mfrow = c(length(dataset_indexes), 1), mar = c(4, 4, 2, 2), oma = c(0, 0, 0, 0))
    for (i in seq_along(dataset_indexes)) {
      models <- all_data[[i]]
      title <- dataset_titles[dataset_indexes[i]]
      if (length(models) == 0) {
        plot(1, type = "n", xlab = "", ylab = "", main = paste("AUC -", title, "(No Data)"),
             cex.lab = 1.25, cex.axis = 1.1, cex.main = 1.3, mgp = c(2.5, 1, 0))
        next
      }
      auc_ylim <- range(unlist(lapply(models, function(m) m$res$AUC)), na.rm = TRUE)
      plot(models[[1]]$res$AUC, type = "o", pch = models[[1]]$pch, col = models[[1]]$color,
           ylim = auc_ylim, xlab = "Expanding Window (% Data Used for Training)", 
           ylab = "AUC Next 70% of Data", main = title,
           cex.lab = 1.25, cex.axis = 1.1, cex.main = 1.3, mgp = c(2.5, 1, 0))
      for (j in 2:length(models)) {
        lines(models[[j]]$res$AUC, type = "o", pch = models[[j]]$pch, col = models[[j]]$color)
      }
      legend("bottomright", legend = sapply(models, function(m) m$name),
             col = sapply(models, function(m) m$color), pch = sapply(models, function(m) m$pch), 
             lty = 1, cex = 0.9, bg = "white")
    }
    dev.off()
    message("AUC plot saved as: ", auc_output_file)
  }
  
  plot_model_performance_tiled(exclude_files="AKT")
