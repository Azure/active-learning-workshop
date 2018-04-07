entropy <- function(p1){
  LITTLE_BIT <- 1e-12
  p1 <- (p1 + LITTLE_BIT)/( 1 + 2*LITTLE_BIT)
  H <- function(p){
    p[0>p] <- 0
    p[0==p] <- 1e-100
    -p * log2(p)
  }
  H(p1) + H(1 - p1)
}

get_new_pseudolabeled_sample <- function(labeled_filenames){
  unlabeled_data_df[unlabeled_data_df$rev_id %in% labeled_filenames$rev_id, ]
}

fit_and_evaluate_model <- function(candidate_cases, form=FORM, test_set=TEST_SET){
  
  compute_roc <- function(pred_df){
    LITTLE_BIT <- 1e-12
    pred_df$Probability <- (pred_df$Probability + LITTLE_BIT)/( 1 + 2*LITTLE_BIT)
    rxRoc("flagged", "Probability", pred_df, numBreaks=1000)
  }
  
  prediction_logloss <- function(pred_df){
    LITTLE_BIT <- 1e-12
    pred_df$Probability <- (pred_df$Probability + LITTLE_BIT)/( 1 + 2*LITTLE_BIT)
    log_probs <- with(pred_df, log(Probability[flagged]))
    -sum(log_probs)/nrow(pred_df)
  }
  
  # for binary, NA means bad example
  training_set_new <- candidate_cases %>% filter(!is.na(flagged))
  
  progress_messages <- capture.output({
    fit_new <- rxFastTrees(form, training_set_new,
                           type="binary", 
                           reportProgress=0, verbose=0)
    
    pred_test <- rxPredict(fit_new, test_set, extraVarsToWrite=c("rev_id", "flagged"))
    
    roc_obj = compute_roc(pred_test)
    
    results <- list(
      model=fit_new,
      test_predictions=pred_test,
      roc = roc_obj,
      performance = c(tss=nrow(training_set_new),
                      accuracy=with(pred_test, sum(flagged == PredictedLabel)/length(flagged)),
                      neg_logloss = -prediction_logloss(pred_test),
                      auc = rxAuc(roc_obj)
      ),
      confusion = with(pred_test, table(flagged, PredictedLabel))
    )
  })
  
  return(results)
}

get_auc <- function(roc_obj){
  library(pROC)
  if ('rxRoc' %in% class(roc_obj)){
    rxAuc(roc_obj)
  } else {
    pROC::auc(roc_obj)
  }
}

plot_roc_history <- function(initial_results, results_history){
  roc0 <- initial_results$roc
  with(roc0, plot(1 - specificity, sensitivity, type='l', col="black"))
  text(0.75, 0, sprintf("AUC: %0.3f", get_auc(roc0)), col="black", pos=3, cex=0.8)
  rbow <- rainbow(length(results_history), end=2/3, v=0.75)
  for (i in seq_along(results_history)){
    roc_N <- results_history[[i]]$roc
    with(roc_N, lines(1 - specificity, sensitivity, col=rbow[i]))
    text(0.75, i * 0.04, sprintf("AUC: %0.3f", get_auc(roc_N)), col=rbow[i], pos=3, cex=0.8)
  }
}

plot_probability_distributions <- function(res, title){
  res$test_predictions %>% 
  ggplot(aes(x=Probability, fill=flagged)) + 
  geom_density(bw=0.01, alpha=0.5)
}

run_ngram_learning_curve <- function(seed, ts_sizes, training_candidates, test_set){
  set.seed(seed)
  initial_training_set_ids <- sample(training_candidates$rev_id, ts_sizes[[1]])
  training_set_ids <- initial_training_set_ids
  
  train_with_additional_cases <- function(num_new_cases){
    new_ids <- sample(setdiff(training_candidates$rev_id, training_set_ids), num_new_cases)
    training_set_ids <<- c(training_set_ids, new_ids) 
    
    random_training_set <- training_candidates %>% filter(rev_id %in% training_set_ids)
    
    model <- rxFastTrees(is_attack ~ ngrams, random_training_set, type="binary",
                         mlTransforms = list(
                           featurizeText(vars = c(ngrams = "comment"),
                                         wordFeatureExtractor = ngramCount(ngramLength = 2, weighting="tfidf"),
                                         charFeatureExtractor = ngramCount(ngramLength=3))))
    pred <- rxPredict(model, data=test_set, extraVarsToWrite="is_attack")
    with(pred, data.frame(tss=nrow(random_training_set), accuracy=sum(PredictedLabel==is_attack)/nrow(test_set), auc=rxAuc(rxRoc("is_attack", "Probability", pred))))
    
  }
  
  performance_list <- lapply(c(0, diff(ts_sizes)), train_with_additional_cases)
  
  performance_df <- bind_rows(performance_list)
  performance_df$ts_sizes <- ts_sizes # to double check against tss
  performance_df$seed <- seed
  performance_df
}
