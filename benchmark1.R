# library(devtools)
# install_github('yatzy/xgbimpute')
# install.packages( c('softImpute' , 'missRanger' ) )
# install.packages('bit64') 
# install.packages('microbenchmark')

library(xgbimpute)
library(softImpute)
library(missRanger)

library(data.table)
library(bit64)

library(microbenchmark)

################## matrix cases

cup_zip_file = download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/kddcup98-mld/epsilon_mirror/cup98lrn.zip",
              destfile='cup.zip' )
unz( temp_zip , 'cupdata.txt' )

cup_data = fread( paste(temp_dir,'cup98LRN.txt', sep ='/') )

fread('/tmp/RtmpwFsIYB/cup98LRN.txt' , skip = 1)


scania_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00414/to_uci.zip'
download.file(scania_data_url ,
              destfile='scania.zip' )
scania = fread('/home/yatzy/Applications/imputation_benchmark/to_uci/aps_failure_training_set.csv' 
               , na.strings = 'na'  )
scania[,class := ifelse(class == 'neg' , 0L , 1L)]
scania_matrix = as.matrix(scania)


scania_imputation = microbenchmark(
  xgb_imputation = impute_xgboost(scania_matrix , nrounds = 40 ) ,
  ranger_imputation = missRanger(scania , num.trees = 40 ) , 
  als_imputation = softImpute(scania_matrix , rank.max = 20   ) , 
  times = 3L
)
)