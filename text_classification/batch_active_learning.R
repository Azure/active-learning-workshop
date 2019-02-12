library(dplyr)
library(glmnet)
library(feather)
library(doParallel)

IN_DIR <- "/Users/rmhorton/Documents/Strata2019"
ALL_DATA <- read_feather(file.path(IN_DIR, "wiki_attacks_use_encoded_30k.feather"))
OUT_DIR <- "lambda_scans4"
PARAMS_FILE <- "parameter_sets4.csv"

if (!file.exists(OUT_DIR)) dir.create(OUT_DIR)

run_learning_curve <- function(params, all_data, output_dir){
  params_tag <- with(params, sprintf("run%d_%dx%d_rep%d_seed%d_%svs%s_mu%0.2f_sigma%0.2f_lambda%0.2f_alpha%0.2f", 
                                     params_id,
                                     examples_per_iteration, 
                                     num_iterations,
                                     replicate_num,
                                     seed,
                                     candidate_group,
                                     test_group,
                                     mu,
                                     sigma,
                                     lambda,
                                     alpha))
  
  message(params_tag)
  
  set.seed(params$seed)
  
  subset_name <- sample(rep(LETTERS[1:3], times=nrow(all_data)/3))
  table(subset_name)
  
  CANDIDATE_SET <- all_data[subset_name==params$candidate_group,]
  TEST_SET <- all_data[subset_name==params$test_group,]
  
  INPUTS <- names(CANDIDATE_SET)[1:512]
  OUTCOME <- "flagged"
  
  INITIAL_TRAINING_CASES <- with(CANDIDATE_SET, {
    sample(rev_id, 100)
  })
  
  TRAINING_CASES <- INITIAL_TRAINING_CASES
  
  set.seed(100*params$seed + params$replicate_num) # re-seed RNG for each replicate
  
  # Be sure that there are some examples from each class!
  CANDIDATE_SET[CANDIDATE_SET$rev_id %in% INITIAL_TRAINING_CASES, 'flagged'] %>% table
  
  NULL_MODEL <- "placeholder for method dispatch."
  class(NULL_MODEL) <- "null_model"
  predict.null_model <- function(object, newx, type) matrix(runif(nrow(newx)), ncol=1)
  
  select_new_cases <- function(current_model, n, sigma, mu){
    available_cases <- CANDIDATE_SET[!(CANDIDATE_SET$rev_id %in% TRAINING_CASES),]
    X_available <- as.matrix(available_cases[INPUTS])
    category_prob <- predict(current_model, newx=X_available, type="response")[,1]
    selection_weight <- dnorm(category_prob, mean=mu, sd=sigma)
    sample(available_cases$rev_id, size=n, prob=selection_weight)
  }
  
  fit_and_select <- function(iteration_num, cases_per_iteration, selection_mode="active"){
    training_set <- CANDIDATE_SET[CANDIDATE_SET$rev_id %in% TRAINING_CASES,]
    
    alpha <- params$alpha
    lambda <- params$lambda
    X_train <- training_set[INPUTS] %>% as.matrix
    y_train <- training_set[[OUTCOME]]
    model <- glmnet(X_train, y_train, 
                    alpha=alpha, lambda=lambda, 
                    family="binomial")
    
    X_test <- TEST_SET[INPUTS] %>% as.matrix
    y_test <- TEST_SET[[OUTCOME]]
    
    test_pred <- predict(model, X_test, type="response")[,1]
    test_set_auc <- pROC::auc(factor(y_test), test_pred, direction='<') %>% as.numeric
    
    selection_model <- if (selection_mode == "active") model else NULL_MODEL
    new_cases <- select_new_cases(selection_model, n=params$examples_per_iteration, sigma=params$sigma, mu=params$mu)
    TRAINING_CASES <<- c(TRAINING_CASES, new_cases)
    
    list(model=model, 
         iteration=iteration_num, 
         lambda = lambda,
         alpha=alpha,
         auc = test_set_auc,
         tss = model$nobs,
         selection_mode=selection_mode, 
         new_cases=paste(new_cases, collapse=','))
  }
  
  do_training_run <- function(num_iterations, ...){
    TRAINING_CASES <<- INITIAL_TRAINING_CASES
    run_results <- lapply(1:num_iterations, fit_and_select, ...)
  }
  
  learning_results <- do_training_run(num_iterations=params$num_iterations, 
                                  cases_per_iteration=params$examples_per_iteration, 
                                  selection_mode=params$selection_mode)
  
  learning_results_file <- sprintf("learning_curve_%s.Rds", params_tag)
  
  saveRDS(learning_results, file.path(output_dir, learning_results_file))
}

# Expand parameter set table
param_ranges <- list(
  seed = 2,
  initial_examples = 100,
  examples_per_iteration = 100,
  num_iterations = 99,
  replicate_num = 1:5,
  mu = c(0.5),
  sigma = 0.1,
  lambda=10^seq(-4, 1, len=16),
  alpha=0,
  candidate_group = c('A', 'B'),
  selection_mode=c("active","passive")
)

next_group <- function(group_vec, groups=c('A', 'B', 'C')){
  group_idx <- sapply(group_vec, function(grp) which(groups==grp))
  groups[((group_idx) %% length(groups)) + 1]
}

parameter_sets <- expand.grid(param_ranges) %>% 
  mutate(test_group=next_group(candidate_group),
         params_id=100000 + (1:n()))

write.csv(parameter_sets, file.path(OUT_DIR, PARAMS_FILE), row.names=FALSE, quote=FALSE)

# run_learning_curve(parameter_sets[1,], ALL_DATA, OUT_DIR)

num_cores <- detectCores(logical=TRUE)
cl <- makeCluster(num_cores)
registerDoParallel(cl)

meh <- foreach (idx = 1:nrow(parameter_sets), 
                .packages=c('dplyr', 'glmnet', 'feather')) %dopar% {
  run_learning_curve(parameter_sets[idx,], ALL_DATA, OUT_DIR)
}

stopCluster(cl)

