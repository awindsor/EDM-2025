
# This script demonstrates how to preprocess three different DataShop
# exports and compute spacing predictors for use with the LKT package.
# It prepares three datasets:
#   * val  - Cloze style fill‑in‑the‑blank data
#   * val2 - MATHia export
#   * val3 - Interleaving/Blocking dataset

library(LKT)
set.seed(41)                           # keep results reproducible
val <- largerawsample                  # example cloze dataset already loaded

#clean it up
val$KC..Default.<-val$Problem.Name
# make it a data table
# val= setDT(val)

# get the times of each trial in seconds from 1970
val$CF..Time.<-as.numeric(as.POSIXct(as.character(val$Time),format="%Y-%m-%d %H:%M:%S"))

#make sure it is ordered in the way the code expects
val<-val[order(val$Anon.Student.Id, val$CF..Time.),]

#create a binary response column to predict and extract only data with a valid value
val$CF..ansbin.<-ifelse(tolower(val$Outcome)=="correct",1,ifelse(tolower(val$Outcome)=="incorrect",0,-1))
val<-val[val$CF..ansbin.==0 | val$CF..ansbin.==1,]



# create durations
val$Duration..sec.<-(val$CF..End.Latency.+val$CF..Review.Latency.+500)/1000

# The computeSpacingPredictors() helper will generate recency, spacing
# and forgetting features used by several LKT models.  It requires the
# timestamps and optional trial durations computed above.
val <- computeSpacingPredictors(val, "KC..Default.")      # by knowledge component
val <- computeSpacingPredictors(val, "KC..Cluster.")      # by text cluster
val <- computeSpacingPredictors(val, "Anon.Student.Id")   # by student
val <- computeSpacingPredictors(val, "CF..Correct.Answer.") # by answer






# ---- Load and clean the MATHia export ----
# Replace the path below with the DataShop file on your machine
set.seed(41)
datafile <- "C:/Users/ppavl/OneDrive/Active projects/ds4845_tx_All_Data_6977_2021_0723_141809.txt"
val2 <- read.delim(colClasses = c("Anon.Student.Id"="character"),
                   datafile, sep="\t", header=TRUE, quote="")
val2=as.data.table(val2)
val2$CF..Time.<-as.numeric(as.POSIXct(as.character(val2$Time),format="%Y-%m-%d %H:%M:%S"))

#make sure it is ordered in the way the code expects
val2<-val2[order(val2$Anon.Student.Id, val2$CF..Time.),]

#create a binary response column to predict and extract only data with a valid value

val2$Outcome<-ifelse(tolower(val2$Outcome)=="ok","CORRECT","INCORRECT")
val2$CF..ansbin.<-ifelse(tolower(val2$Outcome)=="correct",1,0)
val2<-val2[val2$CF..ansbin.==0 | val2$CF..ansbin.==1,]

#subtot<-  aggregate(val2$CF..ansbin.,by=list(val2$Anon.Student.Id),FUN=length)
# subtot<- subtot[subtot$x<20,]
# val2<-val2[!(val2$Anon.Student.Id %in% subtot$Group.1),]
val2<-val2[val2$Attempt.At.Step==1,]
val2<-val2[val2$KC..MATHia.!="",]

colnames(val2)[colnames(val2) == "KC..MATHia."] <- "KC..Default."
# Generate the same time‑based features for different grouping factors
val2 <- suppressWarnings(computeSpacingPredictors(val2, "KC..Default."))  # by KC
val2 <- suppressWarnings(computeSpacingPredictors(val2, "Problem.Name"))   # by problem name
val2 <- suppressWarnings(computeSpacingPredictors(val2, "Anon.Student.Id")) # by student





# ---- Load the interleaving/blocking dataset ----
# Again replace the path with the location of the DataShop export
set.seed(41)
datafile <- "C:/Users/ppavl/OneDrive/Active projects/ds1706_tx_All_Data_3416_2017_0623_020504.txt"
val3 <- read.delim(colClasses = c("Anon.Student.Id"="character"),
                   datafile, sep="\t", header=TRUE, quote="")
val3=as.data.table(val3)
val3$CF..Time.<-as.numeric(as.POSIXct(as.character(val3$Time),format="%Y-%m-%d %H:%M:%S"))

#make sure it is ordered in the way the code expects
val3<-val3[order(val3$Anon.Student.Id, val3$CF..Time.),]

#create a binary response column to predict and extract only data with a valid value

#val3$Outcome<-ifelse(tolower(val3$Outcome)=="ok","CORRECT","INCORRECT")
val3$CF..ansbin.<-ifelse(tolower(val3$Outcome)=="correct",1,0)
val3<-val3[val3$CF..ansbin.==0 | val3$CF..ansbin.==1,]

#subtot<-  aggregate(val3$CF..ansbin.,by=list(val3$Anon.Student.Id),FUN=length)
# subtot<- subtot[subtot$x<20,]
# val3<-val3[!(val3$Anon.Student.Id %in% subtot$Group.1),]
val3<-val3[val3$Attempt.At.Step==1,]
val3<-val3[val3$KC..Field.!="",]
# make student stratified folds (for crossvalidation for unseen population)

colnames(val3)[colnames(val3) == "KC..Field."] <- "KC..Default."
val3 <- suppressWarnings(computeSpacingPredictors(val3, "KC..Default."))  # by KC field
val3 <- suppressWarnings(computeSpacingPredictors(val3, "Anon.Student.Id")) # by student




