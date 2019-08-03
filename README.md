# So bad, they're good: a movie recommendation system

## 1. The hypothesis

It may be possible to identify "good" bad movies by looking at the distribution of user review scores. With this recommendation system I first set out to identify these movies, then build out a system to recommend other similar "good" bad movies to users.


Below is the distribution of a movie generally as good, The Godfather Part II:

![](https://github.com/jmcneilkeller/Good_Bad_Movies/blob/master/images/good_movie.png)

This is Gigli, generally regarded as, well, not very good:

![](https://github.com/jmcneilkeller/Good_Bad_Movies/blob/master/images/bad_bad_movie.png)

The below is Neighbors, a fairly average movie:

![](https://github.com/jmcneilkeller/Good_Bad_Movies/blob/master/images/normal_movie.png)

And below is The Room, generally regarded as one of the "best" worst movies:

![](https://github.com/jmcneilkeller/Good_Bad_Movies/blob/master/images/good_bad_movie.png)

As you can see, The Room's rating distribution is much heavier at the tails than any of the others. The ideal "good" bad movie should be concave, with relatively high amounts of both very low and very high ratings.

## 2. The data

I used data from the full Movielens dataset available [here.](https://grouplens.org/datasets/movielens/latest/)

* The original dataset contained 27,000,000+ ratings from 58,000+ movies.
* I removed any movie with less than 30 reviews, leaving me with a final dataset of 15,751 movies.

## 3. Creating the good/bad list

After minor data cleaning and exploration, I created two features to allow me to select for the distribution which I had previously identified:

* Percent Polarity: This is defined as the difference between percentage of reviews that were either 0.5 and 1.0 and the percentage of reviews that were either 4.5 and 5.0. Ideally this number should be positive, but below 0.5.
* Total Tails: This feature sums the percentage of reviews that were either 0.5 or 1.0 and the percentage of reviews that were either 4.5 or 5.0. Ideally this should value should be above 0.5 for our target.

Once I had established those features, I used Agglomerative Clustering from scikit-learn to separate the dataset into three clusters (after checking the dendogram), with the second cluster being the "good" bad cluster.

In order to evaluate my list, I used Movielens's relevancy scores. Movielens generates these scores based on how often users flag a particular movie with particular tag. In this case I used the "So bad its good" and "So bad its funny" tags. The mean relevancy tag scores for my sorted list significantly outperformed the full dataset mean scores.

![](https://github.com/jmcneilkeller/Good_Bad_Movies/blob/master/images/relevancy_scores.png)

## 4. The recommendation system

Once I had created my final "good" bad list, I used Surprise's KNNBaseline model to create the recommendations. My final model RMSE was 1.009. Below is a sample of a recommendation as produced by the engine:

Recommendations for Friday the 13th Part VII: The New Blood (1988):

1. 'Friday the 13th Part VIII: Jason Takes Manhattan (1989)',
2. 'Leatherface: Texas Chainsaw Massacre III (1990)',
3. 'Amityville: A New Generation (1993)',
4. 'Jaws 3-D (1983)',
5. 'Texas Chainsaw Massacre: The Next Generation (a.k.a. The Return of the Texas Chainsaw Massacre) (1994)']
