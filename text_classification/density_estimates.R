# density_estimates.R
# JMA 12 sept 2018

FEATURE_DIM <- length(inputs)

distance_mat <- function(subset_data=SUBSET_DATA, presample_size=params$SUBSET_SIZE) {
  
  euclid_distance <-function(u, v) {
    sqrt(sum((u[1:FEATURE_DIM] -v[1:FEATURE_DIM])^2))/sqrt(FEATURE_DIM)
  } 
  
  pairwise_distances <- matrix(0, nrow=presample_size, ncol=presample_size)
  
  for (i in 1:(presample_size-1) ){  # row
    for (j in (i+1):(presample_size)){  # col
      pairwise_distances[i,j] <- euclid_distance(subset_data[i,], subset_data[j,])
    }
  }
  pairwise_distances
}

# pairwise_distances <- distance_mat(SUBSET_DATA, params$SUBSET_SIZE)

compact_support <- function(pd=pairwise_distances) {
  pdist_v <- as.vector(pd)
  pdist_v <- pdist_v[pdist_v > 0]
  support <- mean(pdist_v, na.rm=TRUE) + 4* sd(pdist_v, na.rm=TRUE)
  support
}

# Triangular distribution.
tri_neighborhood <- function(pd, SUPPORT){
  # cat('pd ', pd)
  if ( pd >= SUPPORT) {
    ret <- 0
  } else {
    ret <- (SUPPORT - pd)/SUPPORT
  }
  # cat('ne', ret, '\n')
  ret * 0.5 * SUPPORT
}

# Epanechnikov kernel (Wasserman, p. 312)
K <- function(x, z) {
  w <- 0
  if ( abs(x) < sqrt(5)) {
    w <- 0.75 * (1 - 0.2*x*x)/sqrt(5)
  }
  w
}

weight <- function(the_point, pairwise_distances, subset_size=params$SUBSET_SIZE, knl=tri_neighborhood) {
  # Scan row of distance matrix, not including the point itself
  nhood_sup <- compact_support(pairwise_distances)
  the_density <- 0
  if ( the_point == 1) {
    # Special case the first and last row of the distance matrix. 
    for (j in 2:subset_size ) {
      the_density <- the_density + knl(pairwise_distances[1, j], nhood_sup )
    }
  } else if ( the_point == subset_size) {
    for (i in 1:(subset_size-1) ) {
      the_density <- the_density + knl(pairwise_distances[i,subset_size], nhood_sup )
    }
  } else {
    for (i in 1:(the_point-1)){
      # cat('i',i, '\n')
      the_density <- the_density + knl( pairwise_distances[i,the_point], nhood_sup )
    }
    for (j in (the_point+1):(subset_size)){
      # cat('j', j, '\n')
      the_density <- the_density + knl( pairwise_distances[the_point,j], nhood_sup )
    }
  }
  # SUBSET_DATA$density[the_point] <- the_density/(params$SUBSET_SIZE -1)
  # cat('SUBSET_DATA', the_point, SUBSET_DATA$density[the_point], '\n')
  the_density/(subset_size -1)
}

feature_density_est <- function(pairwise_distances, subset_data=SUBSET_DATA, subset_size=params$SUBSET_SIZE) {
  subset_data$density<- 0
  for (k in 1:subset_size) {
    # w <- weight(k)
    subset_data$density[k] <- weight(k, pairwise_distances, subset_size=subset_size)
  }
  subset_data
}
