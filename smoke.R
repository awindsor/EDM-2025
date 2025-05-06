library(data.table)
library(jsonlite)

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
    epoch_auc_drop = res$epoch_auc_drop,  # Dropout AUC (no dropout evaluation removed)
    held_auc = res$held_auc               # Held-out AUC
  ))
}

# Example usage:
akt_root <- "C:/Users/ppavl/OneDrive - The University of Memphis/AKT"

# Running the AKT model for the first dataset (val)
res <- run_akt_simple(
  train_df = val,
  test_df  = val,
  akt_root = akt_root,
  epochs   = 60
)

# Inspecting the resulting metrics from AKT
res$epoch_auc_drop
res$held_auc


val2$KC..Unique.step.<-val2$Problem.Name
val2<-val2[order(val2$CF..Time.),]
# Running the AKT model for the second dataset (val2)
res <- run_akt_simple(
  train_df = val2,
  test_df = val2,
  akt_root = akt_root,
  epochs = 60
)

# Inspecting the resulting metrics from AKT
res$epoch_auc_drop
res$held_auc

# Running the AKT model for the third dataset (val3)
res <- run_akt_simple(
  train_df = val3,
  test_df = val3,
  akt_root = akt_root,
  epochs = 60
)

# Inspecting the resulting metrics from AKT
res$epoch_auc_drop
res$held_auc




# Running the AKT model for the first dataset (val)
res <- run_akt_simple(
  train_df = val[1:15000,],
  test_df  = val,
  akt_root = akt_root,
  epochs   = 40
)

# Inspecting the resulting metrics from AKT
res$epoch_auc_drop
res$held_auc

# Running the AKT model for the second dataset (val2)
res <- run_akt_simple(
  train_df = val2[1:15000,],
  test_df = val2,
  akt_root = akt_root,
  epochs = 40
)

# Inspecting the resulting metrics from AKT
res$epoch_auc_drop
res$held_auc

# Running the AKT model for the third dataset (val3)
res <- run_akt_simple(
  train_df = val3[1:5000,],
  test_df = val3,
  akt_root = akt_root,
  epochs = 40
)

# Inspecting the resulting metrics from AKT
res$epoch_auc_drop
res$held_auc
