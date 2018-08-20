library(dplyr)
library(pROC)
library(randomForest)
library(foreach)
library(doParallel)

source("active_learning_lib_random_forest.R")
set.seed(1)

FEATURIZED_DATA_FILE <- "featurized_wiki_comments_attack.Rds"

FEATURIZED_DATA <- readRDS(FEATURIZED_DATA_FILE)
FEATURIZED_DATA <- FEATURIZED_DATA[complete.cases(FEATURIZED_DATA),]

TEST_SET_SIZE <- 10000
test_set_ids <- sample(FEATURIZED_DATA$rev_id, TEST_SET_SIZE)
TEST_SET <- FEATURIZED_DATA %>% filter(rev_id %in% test_set_ids)

TRAINING_CANDIDATES <- FEATURIZED_DATA %>% filter(!(rev_id %in% test_set_ids) )
  

inputs <- grep("^V", names(FEATURIZED_DATA), value=TRUE)
outcome <- "flagged"
FORM <- formula(paste(outcome, paste(inputs, collapse="+"), sep="~"))
# parameters <- list(seed=1, mu=0.65, sigma=0.1, initial_examples_per_class=20, examples_to_label_per_iteration=20, num_iterations=20, presample_size=5000)

PARAMETER_TABLE <- expand.grid(seed=101:115, # 1:3,
                               mu=seq(0.05, 0.95, by=0.05), 
                               sigma=c(0.025, 0.05, 0.10, 0.20),
                               initial_examples_per_class=20, #c(15, 20), 
                               examples_to_label_per_iteration=20, #c(10, 20), 
                               num_iterations=30, 
                               presample_size=15000)

run_param_set <- function(params){ # FEATURIZED_DATA, FORM
  set.seed(params$seed)

  in_labeled_set <- sample(c(TRUE, FALSE), nrow(TRAINING_CANDIDATES), prob=c(0.09, 0.91), replace=TRUE)
  
  labeled_data_df <- TRAINING_CANDIDATES[in_labeled_set,]
  unlabeled_data_df <- TRAINING_CANDIDATES[!in_labeled_set,]
  
  initial_training_set <- labeled_data_df %>%
    group_by(flagged) %>%
    do(sample_n(., params$initial_examples_per_class)) %>%
    ungroup %>%
    as.data.frame
  
  
  select_cases <- function(model, available_cases, 
                           mu=params$mu, 
                           sigma=params$sigma,
                           N=params$examples_to_label_per_iteration, 
                           presample_size=params$presample_size){
    presample_size <- min(nrow(available_cases), presample_size)
    candidate_cases <- available_cases[sample(1:nrow(available_cases), presample_size),]
    
    votes_vec <- predict(model, candidate_cases, type='vote')[,'TRUE']
    predictions_df <- data.frame(rev_id=candidate_cases$rev_id,
                                 flagged=candidate_cases$flagged,
                                 predicted=votes_vec > 0.5,
                                 estimated_probability=votes_vec)
  
    p <- predictions_df$estimated_probability
    w <- dnorm(p, mean=mu, sd=sigma)
    s <- sample(predictions_df$rev_id, N, prob=w, replace=FALSE)
    selected <- predictions_df %>% filter(rev_id %in% s)
  
    return(selected)
  }   
  
  initial_model_results <- fit_and_evaluate_model(initial_training_set, form=FORM, test_set=TEST_SET)
  initial_model_results$selected <- select_cases(initial_model_results$model, unlabeled_data_df)
  initial_model_results$model <- NULL  # randomForest models are too big to save
  
  new_sample <- initial_model_results$selected %>% get_new_pseudolabeled_sample(unlabeled_data_df)
  
  current_training_set <- rbind(initial_training_set, new_sample[names(initial_training_set)])
  
  ALREADY_EVALUATED <- initial_model_results$selected$rev_id
  
  iteration_results <- lapply(1:params$num_iterations, function(i){
    results <- fit_and_evaluate_model(current_training_set, form=FORM, test_set=TEST_SET)
    
    candidate_cases <- unlabeled_data_df[(unlabeled_data_df$rev_id %in% setdiff(unlabeled_data_df$rev_id,
                                                                                ALREADY_EVALUATED)),]
    results$selected <- select_cases(results$model, candidate_cases)
    
    ALREADY_EVALUATED <<- c(ALREADY_EVALUATED, results$selected$rev_id)
    
    next_sample <- results$selected %>% get_new_pseudolabeled_sample(unlabeled_data_df)
    
    current_training_set <<- rbind(current_training_set, next_sample[names(current_training_set)])
    
    results$model <- NULL  # randomForest models are too big to save
    results
  })
  
  c(list(initial_model_results), iteration_results)
}


# RESULTS <- lapply(1:3, function(i) run_param_set(PARAMETER_TABLE[i,]))
NUM_CORES <- detectCores(all.tests = FALSE, logical = TRUE) - 1
cl <- makeCluster(NUM_CORES)
registerDoParallel(cl)
packages <- c('dplyr', 'randomForest', 'pROC')
t0 <- Sys.time()
RESULTS <- foreach(i = 1:nrow(PARAMETER_TABLE), .packages=packages) %dopar% run_param_set(PARAMETER_TABLE[i,])
t1 <- Sys.time()
print(t1 - t0)

# saveRDS(RESULTS, file="RESULTS.Rds")
# saveRDS(PARAMETER_TABLE, file="PARAMETER_TABLE.Rds")

length(RESULTS) # parameter sets
length(RESULTS[[1]]) # iterations
length(RESULTS[[1]][[1]]) # components


summarize_iteration_results <- function(itres, itnums=0:30){
  library(pROC)
  setNames(sapply(itnums+1, function(i) auc(itres[[i]]$roc)), sprintf("round_%02d", itnums))
}

AUC_mat <- sapply(RESULTS, summarize_iteration_results)

AUC_df <- as.data.frame(t(AUC_mat))

PARAMETER_RESULTS <- cbind(PARAMETER_TABLE, AUC_df)

# saveRDS(PARAMETER_RESULTS, file="PARAMETER_RESULTS.Rds")

library(ggplot2)
PARAMETER_RESULTS %>%
  ggplot(aes(x=mu, y=round_30, col=factor(sigma))) + geom_point() + geom_smooth() +
  ggtitle("30 rounds")

library(tidyr)
PARAMETER_RESULTS %>%
  select(mu, sigma, round_05, round_10, round_15, round_20, round_25, round_30) %>%
  gather("round", "auc", -mu, -sigma) %>% 
  ggplot(aes(x=mu, y=auc, col=factor(sigma))) + geom_point(size=0.2) + geom_smooth() +
  facet_wrap('round')

PARAMETER_RESULTS %>%
  select(mu, sigma, round_01, round_02, round_03, round_04, round_05, round_06) %>%
  gather("round", "auc", -mu, -sigma) %>% 
  ggplot(aes(x=mu, y=auc, col=factor(sigma))) + geom_point(size=0.2) + geom_smooth() +
  facet_wrap('round')

#TO DO: calculate lift by decile

get_iteration_decile_lift <- function(res, i){
  pred_df <- res[[i+1]]$test_predictions
  pred_df$decile <- cut(pred_df$estimated_probability, 
                        breaks=quantile(pred_df$estimated_probability, probs=(0:10)/10), 
                        labels=sprintf("decile_%02d", 10:1), 
                        include.lowest=TRUE)
  overall_prevalence <- with(pred_df, sum(flagged)/length(flagged))
  lift_tbl <- pred_df %>% 
    group_by(decile) %>% 
    summarize(num_flagged=sum(flagged), n=n(), decile_outcome=sum(flagged)/length(flagged), lift=decile_outcome/overall_prevalence)
  
  with(lift_tbl, setNames(lift, decile))
}

get_decile_lift_history <- function(){
  keeper_deciles <- c("decile_01", "decile_10")
  dlh <- sapply(RESULTS, get_iteration_decile_lift, 0) %>% t %>% as.data.frame
  dlh <- dlh[keeper_deciles]
  names(dlh) <- sprintf("%srnd%02d", names(dlh), 0)
  for (i in 1:30){
    lift_df <- sapply(RESULTS, get_iteration_decile_lift, i) %>% t %>% as.data.frame
    lift_df <- lift_df[keeper_deciles]
    names(lift_df) <- sprintf("%srnd%02d", names(lift_df), i)
    dlh <- cbind(dlh, lift_df)
  }
  dlh
}


# NOTE: there are only 15 distinct values of each decile lift in iteration 1

# with(lift_df, plot(decile_01, decile_10))
decile_lift_history <- get_decile_lift_history()

PARAMETER_RESULTS2 <- cbind(PARAMETER_RESULTS, decile_lift_history)

saveRDS(PARAMETER_RESULTS2, file="PARAMETER_RESULTS_FIXED_TRAINING_SET.Rds")

PARAMETER_RESULTS2 %>%
  select(mu, sigma, decile_01rnd05, decile_01rnd10, decile_01rnd15, decile_01rnd20, decile_01rnd25, decile_01rnd30) %>%
  gather("round", "lift_decile_01", -mu, -sigma) %>% 
  ggplot(aes(x=mu, y=log(lift_decile_01), col=factor(sigma))) + geom_point(size=0.2) + geom_smooth() +
  facet_wrap('round')

PARAMETER_RESULTS2 %>%
  select(mu, sigma, decile_10rnd05, decile_10rnd10, decile_10rnd15, decile_10rnd20, decile_10rnd25, decile_10rnd30) %>%
  gather("round", "lift_decile_10", -mu, -sigma) %>% 
  ggplot(aes(x=mu, y=log(lift_decile_10), col=factor(sigma))) + geom_point(size=0.2) + geom_smooth() +
  facet_wrap('round')
