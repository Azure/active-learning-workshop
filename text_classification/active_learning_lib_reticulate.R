
get_new_pseudolabeled_sample <- function(labeled_filenames, unlabeled_data_df){
  unlabeled_data_df[unlabeled_data_df$rev_id %in% labeled_filenames$rev_id, ]
}

fit_and_evaluate_model <- function(candidate_cases, form=FORM, test_set=TEST_SET){

  library(reticulate)
  library(pROC)
  
  use_python("C:/ProgramData/Miniconda3'python.exe")
  use_condaenv("base")
  
  candidate_cases$flagged <- factor(candidate_cases$flagged)
  
  compute_roc <- function(pred_df){
    with(pred_df, roc(flagged, estimated_probability, direction = "<"))
  }
  
  # for binary, NA means bad example
  training_set_new <- candidate_cases %>% filter(!is.na(flagged))
  
  # fit_new <- randomForest(form, training_set_new, type='classification', keep.forest=TRUE)
  # votes_vec <- predict(fit_new, test_set, type='vote')[,'TRUE']
  
  sken <- import("sklearn.ensemble")
  model <- sken$RandomForestClassifier(n_estimators=101L, n_jobs=-1L, max_depth=8L)
  # we need to remove Intercept column from model matrix; be sure `form` is a character string
  form1 <- formula(paste0(FORM, ' - 1'))
  X_train <- model.matrix(form1, training_set_new)
  model$fit(X=X_train, y=training_set_new[['flagged']])
  
  X_test <- model.matrix(form1, test_set)
  pred_prob <- model$predict_proba(X_test)[,2]

  pred_test <- data.frame(rev_id=test_set$rev_id,
                          flagged=test_set$flagged,
                          predicted=pred_prob > 0.5,
                          estimated_probability=pred_prob)

  roc_obj = compute_roc(pred_test)

  results <- list(
    model=model,
    test_predictions=pred_test,
    roc = roc_obj,
    performance = c(tss = nrow(training_set_new),
                    accuracy = with(pred_test, sum(flagged == predicted)/nrow(pred_test)),
                    auc = auc(roc_obj)
    ),
    confusion = with(pred_test, table(flagged, predicted))
  )
  
  return(results)
}

get_auc <- function(roc_obj){
  pROC::auc(roc_obj)
}

plot_roc_history <- function(initial_results, results_history, ...){
  roc0 <- initial_results$roc
  with(roc0, plot(1 - specificities, sensitivities, type='l', col="black", lwd=2, ...))
  text(0.75, 0, sprintf("AUC: %0.3f", get_auc(roc0)), col="black", pos=3, cex=0.8)
  rbow <- rainbow(length(results_history), end=2/3, v=0.75)
  for (i in seq_along(results_history)){
    roc_N <- results_history[[i]]$roc
    with(roc_N, lines(1 - specificities, sensitivities, col=rbow[i]))
    text(0.75, i * 0.04, sprintf("AUC: %0.3f", get_auc(roc_N)), col=rbow[i], pos=3, cex=0.8)
  }
}

plot_probability_distributions <- function(res, title){
  res$test_predictions %>% 
  ggplot(aes(x=estimated_probability, fill=flagged)) + 
  geom_density(bw=0.02, alpha=0.5) + ggtitle(title)
}


get_uncertainty_function <- function(pred_df, params){
  function(x){
    dnorm(x, mean=params$mu, sd=params$sigma)
  }
}

select_cases <- function(current_results, available_cases, params){
  
  N=params$examples_to_label_per_iteration
  presample_size=params$presample_size
  
  model <- current_results$model
  presample_size <- min(nrow(available_cases), presample_size)
  candidate_cases <- available_cases[sample(1:nrow(available_cases), presample_size),]
  
  # votes_vec <- predict(model, candidate_cases, type='vote')[,'TRUE']'
  X_candidates <- model.matrix(formula(FORM), candidate_cases)
  votes_vec <- model$predict_proba(X_candidates)[,2]
  predictions_df <- data.frame(rev_id=candidate_cases$rev_id,
                               flagged=candidate_cases$flagged,
                               predicted=votes_vec > 0.5,
                               estimated_probability=votes_vec)
  
  uncertainty <- get_uncertainty_function(current_results$test_predictions, params)
  
  p <- predictions_df$estimated_probability
  u <- uncertainty(p)
  s <- sample(predictions_df$rev_id, N, prob=u, replace=FALSE)
  selected <- predictions_df %>% filter(rev_id %in% s)
  
  return(selected)
}

run_passive_learning_curve <- function(seed, ts_sizes, unlabeled_data_df, FEATURIZED_DATA, params){
  set.seed(seed)
  
  random_training_set <- FEATURIZED_DATA %>%
    group_by(flagged) %>%
    do(sample_n(., params$initial_examples_per_class)) %>%
    ungroup %>%
    as.data.frame

  random_training_set_results <- lapply(c(0,diff(ts_sizes)), function(tss){
    new_ids <- sample(setdiff(unlabeled_data_df$rev_id, random_training_set$rev_id), tss)
    new_cases <- unlabeled_data_df %>% filter(rev_id %in% new_ids)
    random_training_set <<- rbind(random_training_set, new_cases)
    fit_and_evaluate_model(random_training_set)
  })
  
  random_ts_performance <- random_training_set_results %>%
    lapply("[[", "performance") %>%
    do.call(bind_rows, .)
  
  random_ts_performance$mode <- "random"
  random_ts_performance$seed <- seed
  
  random_ts_performance
}

run_active_learning_curve <- function(params, unlabeled_data_df, FEATURIZED_DATA){
  set.seed(params$seed)
  
  initial_training_set <- FEATURIZED_DATA %>%
    group_by(flagged) %>%
    do(sample_n(., params$initial_examples_per_class)) %>%
    ungroup %>%
    as.data.frame
  
  initial_model_results <- fit_and_evaluate_model(initial_training_set)
  initial_model_results$selected <- select_cases(initial_model_results, unlabeled_data_df, params)
  
  new_sample <- initial_model_results$selected %>% get_new_pseudolabeled_sample(unlabeled_data_df)
  
  current_training_set <- rbind(initial_training_set, new_sample[names(initial_training_set)])
  
  ALREADY_EVALUATED <- initial_model_results$selected$rev_id
  
  iteration_results <- lapply(1:params$num_iterations, function(i){
    results <- fit_and_evaluate_model(current_training_set)
    
    candidate_cases <- unlabeled_data_df[(unlabeled_data_df$rev_id %in% setdiff(unlabeled_data_df$rev_id,
                                                                                ALREADY_EVALUATED)),]
    results$selected <- select_cases(results, candidate_cases, params)
    
    ALREADY_EVALUATED <<- c(ALREADY_EVALUATED, results$selected$rev_id)
    
    next_sample <- results$selected %>% get_new_pseudolabeled_sample(unlabeled_data_df)
    
    current_training_set <<- rbind(current_training_set, next_sample[names(current_training_set)])
    
    results
  })
  
  iteration_results <- list(initial_model_results) %>% 
    append(iteration_results)  %>% 
    lapply(function(ires) ires$performance %>% as.list %>% as.data.frame) %>% 
    bind_rows
  iteration_results$seed <- params$seed
  iteration_results
}
