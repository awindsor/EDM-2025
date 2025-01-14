library(data.table); library(readr); library(dplyr); library(LKT); library(boot); library(bit64); library(pROC); library(elo)

all_data <- setDT(val[order(val$CF..Time.),])
all_data[, fold := cut(.I, breaks = 100, labels = 1:100)]
res1 <- res2 <- res3 <-res4 <-res5<-res6 <-res7<-res8<-res9 <- data.frame(RMSE = numeric(), LL = numeric(), N = numeric(), AUC = numeric())

for (i in 1:30) {
  print(i)
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
  
  # Model 5 Logitdec AFM
  modelob5 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("logitdec", "logitdecevol", "lineafm"), fixedpars = c(.98, .98, .8))
  pred5 <- pmin(pmax(inv.logit(as.matrix(modelob5$predictors %*% modelob5$coefs)[, ]), 1e-5), 0.99999)[modelob5$newdata$fold %in% (i+1):min(100, i+70)]
  actual5 <- modelob5$newdata$CF..ansbin.[modelob5$newdata$fold %in% (i+1):min(100, i+70)]
  res5 <- rbind(res5, c(sqrt(mean((actual5 - pred5)^2)), -mean(actual5 * log(pred5) + (1 - actual5) * log(1 - pred5)), length(actual5), as.numeric(roc(actual5, pred5, quiet = TRUE)$auc)))
  
  # Model 6 Logitdec adapt
  modelob6 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("logitdec", "logitdecevol", "logsuc"), fixedpars = c(.98, .98, .8))
  pred6 <- pmin(pmax(inv.logit(as.matrix(modelob6$predictors %*% modelob6$coefs)[, ]), 1e-5), 0.99999)[modelob6$newdata$fold %in% (i+1):min(100, i+70)]
  actual6 <- modelob6$newdata$CF..ansbin.[modelob6$newdata$fold %in% (i+1):min(100, i+70)]
  res6 <- rbind(res6, c(sqrt(mean((actual6 - pred6)^2)), -mean(actual6 * log(pred6) + (1 - actual6) * log(1 - pred6)), length(actual6), as.numeric(roc(actual6, pred6, quiet = TRUE)$auc)))
  
  # Model 7 AFM (with logitdec instead of student int)
  modelob7 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("logitdec", "intercept", "lineafm"), fixedpars = c(0.98))
  pred7 <- pmin(pmax(inv.logit(as.matrix(modelob7$predictors %*% modelob7$coefs)[, ]), 1e-5), 0.99999)[modelob7$newdata$fold %in% (i+1):min(100, i+70)]
  actual7 <- modelob7$newdata$CF..ansbin.[modelob7$newdata$fold %in% (i+1):min(100, i+70)]
  res7 <- rbind(res7, c(sqrt(mean((actual7 - pred7)^2)), -mean(actual7 * log(pred7) + (1 - actual7) * log(1 - pred7)), length(actual7), as.numeric(roc(actual7, pred7, quiet = TRUE)$auc)))
  
  # Model 8 AFM (with logitdec evol instead of KC int)
  modelob8 <- LKT(verbose = FALSE, data = all_data, interc = TRUE, dualfit = FALSE, factrv = 1e11, 
                  usefolds = (1:i), components = c("Anon.Student.Id", "KC..Default.", "KC..Default."), 
                  features = c("intercept", "logitdecevol", "lineafm"), fixedpars = c(.98))
  pred8 <- pmin(pmax(inv.logit(as.matrix(modelob8$predictors %*% modelob8$coefs)[, ]), 1e-5), 0.99999)[modelob8$newdata$fold %in% (i+1):min(100, i+70)]
  actual8 <- modelob8$newdata$CF..ansbin.[modelob8$newdata$fold %in% (i+1):min(100, i+70)]
  res8 <- rbind(res8, c(sqrt(mean((actual8 - pred8)^2)), 
                        -mean(actual8 * log(pred8) + (1 - actual8) * log(1 - pred8)), 
                        length(actual8), 
                        as.numeric(roc(actual8, pred8, quiet = TRUE)$auc)))
  
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
train_pcts = 30
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

plot(auc_test,ylim=c(.65,.85),xlab="Percentage of Data Used in Training (%)",ylab="Test AUC",pch=16)
lines(auc_test,ylim=c(.65,.85),xlab="Percentage of Data Used in Training (%)",ylab="Test AUC",pch=16,lwd=2)
auc_test
chosen_k
plot(rmse_test,ylim=c(.25,.5),xlab="Percentage of Data Used in Training (%)",ylab="Test RMSE",pch=16)
lines(rmse_test,ylim=c(.25,.5),xlab="Percentage of Data Used in Training (%)",ylab="Test RMSE",pch=16,lwd=2)
res_elo = data.frame(RMSE = rmse_test,
                     LL = NA,
                     N = NA,
                     AUC = auc_test)
# Add Model 9 to column names
colnames(res1) <- colnames(res2) <- colnames(res3) <- colnames(res4) <- colnames(res5) <- colnames(res6) <- colnames(res7) <- colnames(res8) <- colnames(res9) <- c("RMSE", "LL", "N", "AUC")

# Dynamically calculate y-limits for RMSE
rmse_ylim <- range(c(res1$RMSE, res2$RMSE, res3$RMSE, res4$RMSE, res5$RMSE, res6$RMSE, res7$RMSE, res8$RMSE, res9$RMSE,res_elo$RMSE))

# Manually sorted models as a list
models <- list(
  list(name = "Model 1: fixed effects", res = res1, color = "blue"),
  list(name = "Model 3: int + logitdecevol + lineafm$", res = res3, color = "darkgreen"),
  list(name = "Model 8: int + logitdecevol + lineafm", res = res8, color = "cyan"),
  list(name = "Model 2: logitdec + int + lineafm$", res = res2, color = "red"),
  list(name = "Model 7: logitdec + int + lineafm1", res = res7, color = "pink"),
  list(name = "Model 4: logitdec + logitdecevol + lineafm$", res = res4, color = "purple"),
  list(name = "Model 5: logitdec + logitdecevol + lineafm1", res = res5, color = "orange"),
  list(name = "Model 6: logitdec + logitdecevol + logsuc1", res = res6, color = "brown"),
  list(name = "Model 9: logitdec + logitdecevol + logsuc1 + recency", res = res9, color = "black"),
  list(name = "Model 10: Elo", res = res_elo, color = "gray")
)

# RMSE Plot
par(mar = c(5, 4, 2, 2))
plot(models[[1]]$res$RMSE, type = "o", pch = 16, col = models[[1]]$color, ylim = rmse_ylim,
     xlab = "Percentage of Data Used in Training (%)", ylab = "RMSE")
for (i in 2:length(models)) {
  lines(models[[i]]$res$RMSE, type = "o", pch = 16, col = models[[i]]$color)
}
legend("topright", legend = sapply(models, function(m) m$name), col = sapply(models, function(m) m$color), pch = 16, lty = 1)

# Dynamically calculate y-limits for AUC
auc_ylim <- range(c(res1$AUC, res2$AUC, res3$AUC, res4$AUC, res5$AUC, res6$AUC, res7$AUC, res8$AUC, res9$AUC,res_elo$AUC))

# AUC Plot
par(mar = c(5, 4, 2, 2))
plot(models[[1]]$res$AUC, type = "o", pch = 16, col = models[[1]]$color, ylim = auc_ylim,
     xlab = "Percentage of Data Used in Training (%)", ylab = "AUC")
for (i in 2:length(models)) {
  lines(models[[i]]$res$AUC, type = "o", pch = 16, col = models[[i]]$color)
}
legend("bottomright", legend = sapply(models, function(m) m$name), col = sapply(models, function(m) m$color), pch = 16, lty = 1)


