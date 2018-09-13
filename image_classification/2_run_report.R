
param_table <- expand.grid(seed=1:3, sigma=c(0.2))

for (i in 1:nrow(param_table)){
  rmarkdown::render("1_woodknots_reticulate.Rmd", params=param_table[i,])
  system(sprintf("mv 1_woodknots_reticulate.html 1_woodknots_reticulate%04d.html", i))
}

