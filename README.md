# tf-yolor
An attempt at implementing YoloR in TensorFlow

## About This Branch

This branch just contains the kaggle notebook(s) that were used to develop thisframework.

## How did it score?

I only got about `0.27` on the private test set, but I only had one submission that went out literally a minute before the deadline (and then it didn't even count T_T).

I found out the hard way that the tensorflow keras APIs don't nicely build resnet graphs when using the `+` version of the add operation and require a keras `Add` layer instead.

This issue was uncovered the morning of the deadline, so I'm pretty happy with that score all things considered.

I assume it could have done better with other tricks like upsampling the image or Slicing-Aided Hyper-Inference (SAHI), or just experimenting with training parameters a bit more.

