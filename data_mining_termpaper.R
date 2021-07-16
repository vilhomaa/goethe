# 
rm(list = ls())

library(mice)
library(Matrix)
library(xgboost)



setwd('your wd path here')

filename <- 'data_file_here'
xsell_raw <- get(load(filename))

tempData <- mice(xsell_raw,m=5,seed=2021)

xsell <- complete(tempData)

## Descriptive statistics
library(psych)
describe(xsell)

################## BEGINNING OF CODE ################## 

# custom profit-based loss function

# Based on:


profit_loss_function<- function(e_profit,fx,w){
  return(w*log(1+exp(-2*e_profit*fx)))
}
# Find the derivative with 'deriv' package:
# Deriv(w*log(1+exp(-2*e_profit*fx))",'fx')
gradient <- function(e_profit,x,w){
  .e2 <- exp(-(2 * (e_profit * x)))
  return(-(2 * (e_profit * w * .e2/(1 + .e2))))
}

# second order derivative gotten by:
# Deriv("w*log(1+exp(-2*e_profit*fx))",'fx',nderiv= 2)
hessian <- function(e_profit,fx,w){
  .e2 <- exp(-(2 * (e_profit * fx))) 
  .e3 <- 1 + .e2
  return(-(2 * (e_profit^2 * w * (2 * (.e2/.e3) - 2) * .e2/.e3)))
}

objective_function <- function(preds,dtrain){
  labels <- getinfo(dtrain,'label')
  # Calculating score -> transforming it to be 1 if the prediction is 
  # the same as the label. This also doesnt allow for values over 1
  fx <- ifelse(labels != 0, preds/labels,0)
  w = abs(labels)
  grad = gradient(labels,fx,w)
  hess = hessian(labels,fx,w)
  return(list(grad = grad, hess = hess))
}

eval_func <- function(preds,dtrain){
  labels <- getinfo(dtrain,'label')
  fx <- ifelse(labels != 0, preds/labels,0)
  w = abs(labels)
  psi <- profit_loss_function(labels,fx,w)
  err <- mean(psi)
  return(list(metric = "error", value = err))
}



# Calibration functio ( data, mikä column sorttaa, coefficienttenit)
#   returnnaa optimal n

profit_vector_given_sorcol <- function(df,df_col,clv_coef,uplift_coef,incresed_churn_prob_coef){
  profit_df <- df[order(df_col,decreasing = TRUE),]
  profit_df <- profit_df[which((profit_df$volume_debit >10 & profit_df$volume_debit_6months>10)),]
  profit_df$uplift_if_xbuy <- profit_df$volume_debit*clv_coef*uplift_coef
  profit_df$downlift_if_no_xbuy <- -profit_df$volume_debit*clv_coef*incresed_churn_prob_coef
  profit_df$actual_profit <- ifelse(profit_df$xsell == 1, profit_df$uplift_if_xbuy,profit_df$downlift_if_no_xbuy)
  return(profit_df$actual_profit)
}

# data prep func -> arguments: data, model



run_analysis <- function(xsell,seed,depth,lr,iterations,stop_n_rounds,min_leaf_size){
  
  # Coefficient's assumed for model
  uplift_coef = 0.3 # -> if a would-buy customer is contacted, what is the % uplift of clv
  clv_coef = 0.01 # clv = this * debit_amount
  incresed_churn_prob_coef = 0.05 # increased chance of churning for non-would-buy customer


  
  ## Data manipulations for making algorithm work
  xsell$gender<- as.factor(xsell$gender)
  xsell$maritial_status<- as.factor(xsell$maritial_status)
  xsell$occup<- as.factor(xsell$occup)
  xsell$xsell <- ifelse(xsell$xsell == 1,1,0)
  
  set.seed(seed) 
  xsell_random <- xsell[order(runif(100000)),] 
  xsell_train_unedited <- xsell_random[1:33333, ]       
  xsell_calibration_unedited <- xsell_random[33334:66666, ]  
  xsell_test_unedited <- xsell_random[66667:100000, ] 
  
  
  # benchmark models
  
  model = xsell ~.-volume_debit-volume_debit_6months
  
  
  train_m <- sparse.model.matrix(model, data = xsell_train_unedited)
  calib_m <- sparse.model.matrix(model, data = xsell_calibration_unedited)
  test_m <- sparse.model.matrix(model, data = xsell_test_unedited)
  
  dtrain <- xgb.DMatrix(train_m,label = xsell_train_unedited$xsell)
  dcalib <- xgb.DMatrix(calib_m,label = xsell_calibration_unedited$xsell)
    

  print("Training benchmark model")
  set.seed(seed)
  watchlist <- list(train = dtrain,validation = dcalib)
  param <- list(max_depth = depth,objective = "binary:logistic",eval_metric = 'auc',min_child_weight = min_leaf_size, eta = lr, verbose = 1) # ,objective = objective_function,eval_metric = eval_func
  xgb <- xgb.train(param, dtrain,nrounds = iterations,watchlist ,early_stopping_rounds=stop_n_rounds,maximize = TRUE) 

  
  # Make predictions
  xsell_calibration <- xsell_calibration_unedited
  xsell_test <- xsell_test_unedited
  xsell_calibration$pred_xgb <- predict(xgb, newdata = calib_m)
  xsell_test$pred_xgb <- predict(xgb, newdata = test_m)
  
  # Benchmark profit of ranking customers on the basis of xsell
  cumsum_profit_xsell_calib <- cumsum(profit_vector_given_sorcol(xsell_calibration,xsell_calibration$pred_xgb,clv_coef,uplift_coef,incresed_churn_prob_coef))
  optimal_n_contacted_customers_xsell <- which.max(cumsum_profit_xsell_calib)
  cumsum_profit_xsell_test <- cumsum(profit_vector_given_sorcol(xsell_test,xsell_test$pred_xgb,clv_coef,uplift_coef,incresed_churn_prob_coef))
  profit_xsell <- cumsum_profit_xsell_test[optimal_n_contacted_customers_xsell]
    
  
  # Benchmark profit by ranking customers with e_uplift = xsell*clv|contacted
  # Expected uplift
  xsell_calibration$uplift_if_xbuy <- xsell_calibration$volume_debit*clv_coef*uplift_coef
  xsell_test$uplift_if_xbuy <- xsell_test$volume_debit*clv_coef*uplift_coef

  xsell_calibration$downlift_if_no_xbuy = -xsell_calibration$volume_debit*clv_coef*incresed_churn_prob_coef
  xsell_test$downlift_if_no_xbuy = -xsell_test$volume_debit*clv_coef*incresed_churn_prob_coef

  xsell_calibration$e_uplift <- xsell_calibration$pred_xgb*xsell_calibration$uplift_if_xbuy+(1-xsell_calibration$pred_xgb)*xsell_calibration$downlift_if_no_xbuy
  xsell_test$e_uplift <- xsell_test$pred_xgb*xsell_test$uplift_if_xbuy+(1-xsell_test$pred_xgb)*xsell_test$downlift_if_no_xbuy

  cumsum_profit_e_uplift_calib <- cumsum(profit_vector_given_sorcol(xsell_calibration,xsell_calibration$e_uplift,clv_coef,uplift_coef,incresed_churn_prob_coef))
  optimal_n_contacted_customers_e_uplift <- which.max(cumsum_profit_e_uplift_calib)
  cumsum_profit_e_uplift_test <-cumsum(profit_vector_given_sorcol(xsell_test,xsell_test$e_uplift,clv_coef,uplift_coef,incresed_churn_prob_coef)) 
  profit_e_uplift <- cumsum_profit_e_uplift_test[optimal_n_contacted_customers_e_uplift]
  
  
  ##
  xsell_train <- xsell_train_unedited
  xsell_calibration <- xsell_calibration_unedited
  xsell_test <- xsell_test_unedited
  
  xsell_train$e_profit <- ifelse(xsell_train$xsell == 1,xsell_train$volume_debit*clv_coef*uplift_coef,-xsell_train$volume_debit*clv_coef*incresed_churn_prob_coef)
  xsell_calibration$e_profit <- ifelse(xsell_calibration$xsell == 1,xsell_calibration$volume_debit*clv_coef*uplift_coef,-xsell_calibration$volume_debit*clv_coef*incresed_churn_prob_coef)
  
  # problem: R is not willing to calculate numbers as big as the profitloss function wants
  # -> transformation: x = x, x€[-1,1], x = log(x), x>1, x = -log(abs(x)), x<1
  
  # xsell_train$e_profit <- xsell_train$e_profit/1000 # dividing by 1000 - otherwise values too big for profitloss func
  xsell_train$e_profit <- ifelse(abs(xsell_train$e_profit) <=exp(1),xsell_train$e_profit,ifelse(xsell_train$e_profit>exp(1),log(xsell_train$e_profit),-log(abs(xsell_train$e_profit))))
  xsell_calibration$e_profit <- ifelse(abs(xsell_calibration$e_profit) <=exp(1),xsell_calibration$e_profit,ifelse(xsell_calibration$e_profit>exp(1),log(xsell_calibration$e_profit),-log(abs(xsell_calibration$e_profit))))
  xsell_test$e_profit <- 0
  
  model <- e_profit ~ .-xsell-volume_debit -volume_debit_6months
  
  
  train_m <- sparse.model.matrix(model, data = xsell_train)
  calib_m <- sparse.model.matrix(model, data = xsell_calibration)
  test_m <- sparse.model.matrix(model, data = xsell_test)
  
  dtrain <- xgb.DMatrix(train_m,label = xsell_train$e_profit)
  dcalib <- xgb.DMatrix(calib_m,label = xsell_calibration$e_profit)
  
  print("Training profit-loss model")
  set.seed(seed)
  watchlist <- list(train = dtrain,validation = dcalib)
  param <- list(max_depth = depth,objective = objective_function,eval_metric = 'mae',min_child_weight = min_leaf_size, eta = lr, verbose = 1) # ,objective = objective_function,eval_metric = eval_func
  bst <- xgb.train(param, dtrain,nrounds = iterations,watchlist ,early_stopping_rounds=stop_n_rounds,maximize = FALSE) 
  

  xsell_calibration$pred_xgb <- predict(bst, newdata = calib_m)
  xsell_test$pred_xgb <- predict(bst, newdata = test_m)
  
  cumsum_profit_profitloss_calib <- cumsum(profit_vector_given_sorcol(xsell_calibration,xsell_calibration$pred_xgb,clv_coef,uplift_coef,incresed_churn_prob_coef))
  optimal_n_contacted_customers_profitloss <- which.max(cumsum_profit_profitloss_calib)
  cumsum_profit_profitloss_test <- cumsum(profit_vector_given_sorcol(xsell_test,xsell_test$pred_xgb,clv_coef,uplift_coef,incresed_churn_prob_coef))
  profit_profitloss <- cumsum_profit_profitloss_test[optimal_n_contacted_customers_profitloss]
  
  return(list(
    cumsum_profit_xsell_calib,
    cumsum_profit_e_uplift_calib,
    cumsum_profit_profitloss_calib,
    cumsum_profit_xsell_test,
    cumsum_profit_e_uplift_test,
    cumsum_profit_profitloss_test,
    optimal_n_contacted_customers_xsell,
    optimal_n_contacted_customers_e_uplift,
    optimal_n_contacted_customers_profitloss,
    profit_xsell,
    profit_e_uplift,
    profit_profitloss))
}

# Grid searching for optimal parameters etc

hyperparameters = vector(mode="list", length=3)
names(hyperparameters) <- c('lr','maxrounds','stoprounds')
hyperparameters$lr <- c(0.1,0.01,0.001,0.0001)
hyperparameters$maxrounds <- c(100,1000,10000,20000)
hyperparameters$stoprounds <- c(10,20,30,40)




xsell_calib_cumsums <- list()
e_uplift_calib_cumsums <- list()
profitloss_calib_cumsums <- list()
xsell_test_cumsums <- list()
e_uplift_test_cumsums <- list()
profitloss_test_cumsums <- list()

profit_storage <- list()

named_vector <- function(x,name){
  names(x) <- name
  return(x)
}


for (i in 1:4) { 
  counter <- 0
  
  lr <- hyperparameters$lr[i]
  maxrounds <- hyperparameters$maxrounds[i]
  stoprounds <- hyperparameters$stoprounds[i]
  
  optimal_n_contacted_customers_xsell_buffer <- c()
  optimal_n_contacted_customers_e_uplift_buffer <- c()
  optimal_n_contacted_customers_profitloss_buffer <- c()
  profit_xsell_buffer <- c()
  profit_e_uplift_buffer <- c()
  profit_profitloss_buffer <- c()
  for (j in 2022:2123) {
    counter <- counter + 1
    results_buffer <- run_analysis(xsell,j,8,lr,maxrounds,stoprounds,10)
    xsell_calib_cumsums[paste0('params_',i,'_cumsum_xsell_calib_seed_',j)] <- list(results_buffer[[1]])
    e_uplift_calib_cumsums[paste0('params_',i,'_cumsum_e_uplift_calib_seed_',j)] <- list(results_buffer[[2]])
    profitloss_calib_cumsums[paste0('params_',i,'_cumsum_profitloss_calib_seed_',j)] <- list(results_buffer[[3]])
    xsell_test_cumsums[paste0('params_',i,'_cumsum_xsell_test_seed_',j)] <- list(results_buffer[[4]])
    e_uplift_test_cumsums[paste0('params_',i,'_cumsum_e_uplift_test_seed_',j)] <- list(results_buffer[[5]])
    profitloss_test_cumsums[paste0('params_',i,'_cumsum_profitloss_test_seed_',j)] <- list(results_buffer[[6]])
    
    optimal_n_contacted_customers_xsell_buffer[counter] <- results_buffer[[7]]
    optimal_n_contacted_customers_e_uplift_buffer[counter] <- results_buffer[[8]]
    optimal_n_contacted_customers_profitloss_buffer[counter] <- results_buffer[[9]]
    profit_xsell_buffer[counter] <- results_buffer[[10]]
    profit_e_uplift_buffer[counter] <- results_buffer[[11]]
    profit_profitloss_buffer[counter] <- results_buffer[[12]]
    

  }
  
  profit_storage[paste0('OnCC_xsell_params_',i)] <- list(optimal_n_contacted_customers_xsell_buffer)
  profit_storage[paste0('OnCC_e_uplift_params_',i)] <- list(optimal_n_contacted_customers_e_uplift_buffer)
  profit_storage[paste0('OnCC_profitloss_params_',i)] <- list(optimal_n_contacted_customers_profitloss_buffer)
  profit_storage[paste0('profit_xsell_params_',i)] <- list(profit_xsell_buffer)
  profit_storage[paste0('profit_e_uplift_params_',i)] <- list(profit_e_uplift_buffer)
  profit_storage[paste0('profit_profitloss_params_',i)] <- list(profit_profitloss_buffer)
}


equalize_lengths <- function(lists){
  minlen <- min(lengths(lists))
  for (name in names(lists)) {
    lists[[name]] <- lists[[name]][1:minlen]
  } 
  return(lists)
}

xsell_calib_cumsums <- equalize_lengths(xsell_calib_cumsums)
e_uplift_calib_cumsums <- equalize_lengths(e_uplift_calib_cumsums)
profitloss_calib_cumsums <- equalize_lengths(profitloss_calib_cumsums)
xsell_test_cumsums <- equalize_lengths(xsell_test_cumsums)
e_uplift_test_cumsums <- equalize_lengths(e_uplift_test_cumsums)
profitloss_test_cumsums <- equalize_lengths(profitloss_test_cumsums)

xsell_calib_cumsums_df <- data.frame(xsell_calib_cumsums)
e_uplift_calib_cumsums_df <- data.frame(e_uplift_calib_cumsums)
profitloss_calib_cumsums_df <- data.frame(profitloss_calib_cumsums)
xsell_test_cumsums_df <- data.frame(xsell_test_cumsums)
e_uplift_test_cumsums_df <- data.frame(e_uplift_test_cumsums)
profitloss_test_cumsums_df <- data.frame(profitloss_test_cumsums)


profit_storage_df <- data.frame(profit_storage)


write.csv(xsell_calib_cumsums_df,'xsell_calib_cumsums_df.csv')
write.csv(e_uplift_calib_cumsums_df,'e_uplift_calib_cumsums_df.csv')
write.csv(profitloss_calib_cumsums_df,'profitloss_calib_cumsums_df.csv')
write.csv(xsell_test_cumsums_df,'xsell_test_cumsums_df.csv')
write.csv(e_uplift_test_cumsums_df,'e_uplift_test_cumsums_df.csv')
write.csv(profitloss_test_cumsums_df,'profitloss_test_cumsums_df.csv')


write.csv(profit_storage_df,'profit_storage_df.csv')



### ANALYSIS OF THE OPTIMAL CALIBRATION SIZES
# TAKE AVERAGE
# -> PLOT ALL 4 AVERAGES


plot_single_cumprofit <- function(Cumulative_profit,Cumulative_profit2){
  ylim_upper <- ifelse(max(Cumulative_profit) > max(Cumulative_profit2),max(Cumulative_profit),max(Cumulative_profit2))
  ylim_lower <- ifelse(min(Cumulative_profit) < min(Cumulative_profit2),min(Cumulative_profit),min(Cumulative_profit2))
  Customers_contacted <- seq(1,25000,1)
  plot(jana,Cumulative_profit[Customers_contacted],type="l",col="red",ylim = c(ylim_lower,ylim_upper),ylab = 'Cumulative profit',xlab = 'Customers contacted')
  lines(jana,Cumulative_profit2[Customers_contacted],col="green")
  abline(v = which.max(Cumulative_profit))
}


plot_single_cumprofit(xsell_calib_cumsums_df[,11],xsell_test_cumsums_df[,11])
plot_single_cumprofit(e_uplift_calib_cumsums_df[,4],e_uplift_test_cumsums_df[,4])
plot_single_cumprofit(profitloss_calib_cumsums_df[,4],profitloss_test_cumsums_df[,4])

aggregate_cumprofit_plot <- function(df1,df2,params_no){
  colname_to_search <- paste0('params_',params_no)
  plot_single_cumprofit(rowMeans(df1[,grepl(colname_to_search, names( df1 ) )]),rowMeans(df2[,grepl(colname_to_search, names( df2 ) )]))
}

par(mfrow=c(1,1))
aggregate_cumprofit_plot(xsell_calib_cumsums_df,xsell_test_cumsums_df,4,0.8)

par(mfrow=c(2,2))
aggregate_cumprofit_plot(e_uplift_calib_cumsums_df,e_uplift_test_cumsums_df,1)
aggregate_cumprofit_plot(e_uplift_calib_cumsums_df,e_uplift_test_cumsums_df,2)
aggregate_cumprofit_plot(e_uplift_calib_cumsums_df,e_uplift_test_cumsums_df,3)
aggregate_cumprofit_plot(e_uplift_calib_cumsums_df,e_uplift_test_cumsums_df,4)
mtext('Density plots of profits', side = 3, line = -2, outer = TRUE, cex = 1.5)
mtext('Red: Calibration data, Green: Test data, Black: Optimal n customers', side = 3, line = -3, outer = TRUE, cex = 1)


par(mfrow=c(1,1))
aggregate_cumprofit_plot(profitloss_calib_cumsums_df,profitloss_test_cumsums_df,4,0.8)



### PLOTS OF THE OPTIMAL PROFITS NUMBER
# -> plot distributions

dist_plot_of_profits <- function(profit_storage_df,paramset){
  profit_storage_params_x <- profit_storage_df[,grepl(paramset, names(profit_storage_df) )]
  profit_storage_profit_params_x <- profit_storage_params_x[,grepl('profit_', names(profit_storage_params_x) )]
  
  
  # Compute the density data
  dens <- density(profit_storage_profit_params_x[,1])
  dens2 <- density(profit_storage_profit_params_x[,2])
  dens3 <- density(profit_storage_profit_params_x[,3])
  # plot density
  title <- paste0('Parameter set ', paramset)
  plot(dens, frame = FALSE, col = "steelblue",main=title) 
  lines(dens2,col = 'red')
  lines(dens3,col = 'darkgreen')
}


par(mfrow=c(2,2))
dist_plot_of_profits(profit_storage_df,1)
dist_plot_of_profits(profit_storage_df,2)
dist_plot_of_profits(profit_storage_df,3)
dist_plot_of_profits(profit_storage_df,4)
mtext('Density plots of profits', side = 3, line = -2, outer = TRUE, cex = 1.5)
mtext('Rankings: \n Blue: xsell, Red: e_uplift, Green: profitloss', side = 3, line = -5, outer = TRUE, cex = 1)




oncc_split_to_tables <- function(profit_storage_df,paramset){
  profit_storage_df_params_x <- profit_storage_df[,grepl(paramset, names(profit_storage_df) )]
  profit_storage_df_oncc_params_x <- profit_storage_df_params_x[,grepl('OnCC', names(profit_storage_df_params_x) )]
  profit_storage_df_colmeans_paramset_x <- colMeans(profit_storage_df_oncc_params_x)
  values <- unname(profit_storage_df_colmeans_paramset_x)
  ranking_type <- c('xsell','e_uplift','profitloss')
  parameter_set <- rep(paramset,3)
  
  data.frame(values,ranking_type,parameter_set)
}

hist_data <- rbind(
  oncc_split_to_tables(profit_storage_df,1),
  oncc_split_to_tables(profit_storage_df,2),
  oncc_split_to_tables(profit_storage_df,3),
  oncc_split_to_tables(profit_storage_df,4)
)


p <- ggplot(data=hist_data, aes(x=parameter_set, y=values, fill=ranking_type)) +
  geom_bar(stat="identity", color="black", position=position_dodge())+
  ggtitle('Optimal number of contacted customers')+
  theme_minimal()
p




############ REGRESSION ############


reg_func <- function(profit_storage_df,paramset){
  profit_storage_df_params_x <- profit_storage_df[,grepl(paramset, names(profit_storage_df) )]
  profit_storage_df_profit_params_x <- profit_storage_df_params_x[,grepl('profit_', names(profit_storage_df_params_x) )]
  
  reg_value_df <- data.frame()
  rankings <- c('xsell','e_uplift','profitloss')
  for (i in 1:3) {
    buffer_df <- data.frame(profit_storage_df_profit_params_x[,i],rep(rankings[i],length(profit_storage_df_profit_params_x[,i])))
    reg_value_df <- rbind(reg_value_df,buffer_df)
  }
  colnames(reg_value_df) <- c('profit','ranking_algorithm')
  
  reg_value_df['xsell'] <- ifelse(reg_value_df[,2]=='xsell',1,0)
  reg_value_df['e_uplift'] <- ifelse(reg_value_df[,2]=='e_uplift',1,0)
  model <- lm(profit ~ xsell + e_uplift,reg_value_df)
  return(model)
}


reg1 <- reg_func(profit_storage_df,1)
summary(reg1)

reg2 <- reg_func(profit_storage_df,2)
summary(reg2)

reg3 <- reg_func(profit_storage_df,3)
summary(reg3)

reg4 <- reg_func(profit_storage_df,4)
summary(reg4)






