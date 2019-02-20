
Kareem Rogers
=============

validation accuracy
===================

``` r
library(keras)

imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb
```

perparing the data
------------------

The argument `num_words = 10000` means that we will only keep the top 10,000 most frequently occurring words in the training data. Rare words will be discarded. This allows us to work with vector data of manageable size.

The variables `train_data` and `test_data` are lists of reviews, each review being a list of word indices (encoding a sequence of words). `train_labels` and `test_labels` are lists of 0s and 1s, where 0 stands for "negative" and 1 stands for "positive":

``` r
str(train_data[[1]])
```

    ##  int [1:218] 1 14 22 16 43 530 973 1622 1385 65 ...

Since we restricted ourselves to the top 10,000 most frequent words, no word index will exceed 10,000:

``` r
max(sapply(train_data, max))
```

    ## [1] 9999

For kicks, here's how you can quickly decode one of these reviews back to English words:

``` r
vectorize_sequences <- function(sequences, dimension = 10000) {
  # Create an all-zero matrix of shape (len(sequences), dimension)
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences)) {
    # Sets specific indices of results[i] to 1s
    results[i, sequences[[i]]] <- 1
  results
}

return(results)

}

# Our vectorized training data
x_train <- vectorize_sequences(train_data)
# Our vectorized test data
x_test <- vectorize_sequences(test_data)
```

we should vector the label.

``` r
y_train <- as.numeric(train_labels)

y_test <- as.numeric(test_labels)
```

Validating our approach
-----------------------

In order to monitor during training the accuracy of the model on data that it has never seen before, we will create a "validation set" by setting apart 10,000 samples from the original training data:

``` r
val_indices <- 1:10000

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]
```

Rebuilding the model
--------------------

``` r
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, kernel_regularizer = regularizer_l1(l=0.0001), activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, kernel_regularizer = regularizer_l1(l=0.0001), activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")


model %>% compile(
  optimizer = "adamax",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
```

``` r
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 4,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)
```

``` r
plot(history)
```

![](Assignment1_DL_kroger27_files/figure-markdown_github/unnamed-chunk-9-1.png)

The Validation set results
--------------------------

``` r
results_val <- model %>% evaluate(x_val, y_val)
```

``` r
print(results_val)
```

    ## $loss
    ## [1] 0.3627627
    ## 
    ## $acc
    ## [1] 0.8889
