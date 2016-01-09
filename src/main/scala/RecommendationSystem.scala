import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object RecommendationSystem extends App {

  def computeMSE(model: MatrixFactorizationModel, data: RDD[Rating]) = {
    val usersProducts = data.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions = model.predict(usersProducts).map {
      case Rating(user, product, rate) => ((user, product), rate)
    }
    val ratesAndPreds = data.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)
    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()
    MSE
  }

  val conf = new SparkConf()
    .setAppName("movie")
    .setMaster("local[*]")
  val sc = new SparkContext(conf)
  sc.setLogLevel("ERROR")

  val parseRatings: (String) => Rating = (line: String) => {
    line.split("::").take(3) match {
      case Array(userID, movieID, rating) => Rating(userID toInt, movieID toInt, rating toDouble)
    }
  }

  val parseMovie: (String) => (Int, String) = (line: String) => {
    line.split("::").take(2) match {
      case Array(movieID, movieTitle) => (movieID toInt, movieTitle)
    }
  }

  val movieDataset = getClass.getResource("/dataset/movies.dat") toString
  val personalRatingDataset = getClass.getResource("/dataset/personalRatings.dat") toString
  val ratingsDataset = getClass.getResource("/dataset/ratings.dat") toString

  val movieRDD: RDD[(Int, String)] = sc.textFile(movieDataset).map(parseMovie).persist
  val personalRatingRDD: RDD[Rating] = sc.textFile(personalRatingDataset).map(parseRatings).persist
  val ratingsRDD: RDD[Rating] = sc.textFile(ratingsDataset).map(parseRatings).persist

  val numRatings = ratingsRDD.count()
  val numUsers = ratingsRDD.map(_.user).distinct().count()
  val numMovies = ratingsRDD.map(_.product).distinct().count()

  println(s"Got $numRatings ratings from $numUsers users on $numMovies movies")

  // train %60, validation %20 and test %20
  val (training, validation, test) = ratingsRDD.randomSplit(Array(.6, .2, .2)) match {
    case Array(training, validation, test) => (training.union(personalRatingRDD).persist, validation.persist, test.persist)
  }

  println(s"Training : ${training.count}, validation : ${validation.count}, test : ${test.count}")

  val (bestModel, bestValidatipnMSE, bestRank, bestLambda, bestNumIter) = {
    for (rank <- Array(8, 12);
         lambda <- Array(.1, 10);
         numIter <- Array(10, 20)
    ) yield {
      val model = ALS.train(training, rank, numIter, lambda)
      val validationMSE = computeMSE(model, validation)
      println(s"MSE(validation)= $validationMSE rank= $rank, lambda=$lambda, numIter= $numIter")
      (model, validationMSE, rank, lambda, numIter)
    }
  }.minBy(_._2)
  val testMSE = computeMSE(bestModel, test)
  println(s"The best model was trained with rank = $bestRank and lambda = $bestLambda and numIter = $bestNumIter, and its MSE on the test set is $testMSE.")

  val candidateRDD = movieRDD.subtractByKey(personalRatingRDD.map(rating => (rating.product, Some)))
  bestModel.predict(candidateRDD.map(can => (0, can._1)))
    .map(rating => (rating.product, rating.rating))
    .join(movieRDD)
    .sortBy(_._2._1, false)
    .take(10).map(_._2._2).foreach(println)

  sc.stop
}
