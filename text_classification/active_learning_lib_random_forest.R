
get_new_pseudolabeled_sample <- function(labeled_filenames, unlabeled_data_df){
  unlabeled_data_df[unlabeled_data_df$rev_id %in% labeled_filenames$rev_id, ]
}

fit_and_evaluate_model <- function(candidate_cases, form=FORM, test_set=TEST_SET){
  library(randomForest)
  library(pROC)
  
  candidate_cases$flagged <- factor(candidate_cases$flagged)
  
  compute_roc <- function(pred_df){
    with(pred_df, roc(flagged, estimated_probability, direction = "<"))
  }
  
  # for binary, NA means bad example
  training_set_new <- candidate_cases %>% filter(!is.na(flagged))
  
  fit_new <- randomForest(form, training_set_new, type='classification', keep.forest=TRUE)
  
  votes_vec <- predict(fit_new, test_set, type='vote')[,'TRUE']
  pred_test <- data.frame(rev_id=test_set$rev_id,
                          flagged=test_set$flagged,
                          predicted=votes_vec > 0.5,
                          estimated_probability=votes_vec)

  roc_obj = compute_roc(pred_test)

  results <- list(
    model=fit_new,
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

