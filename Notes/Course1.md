# Natural Language Processing with Classification and Vector Spaces

## Week 1: Sentiment Analysis with Logistic Regression (Normal)

- __Logistic regression is a statistical method for modeling a binary outcome__ (i.e. a outcome with two possible values, such as success or failure) as a function of one or more predictor variables. It is a type of generalized linear model (GLM) that uses a logistic function to model the probability of the outcome being a certain value (usually "success") given the values of the predictor variables.

    The logistic function, also known as the sigmoid function, is an S-shaped curve that maps any real-valued number to a value between 0 and 1. In logistic regression, this function is used to model the probability of the outcome being a "success" (usually represented by the value 1) given the values of the predictor variables.

    The coefficients of the predictor variables in the logistic regression model are estimated from the data using a method called maximum likelihood estimation. Once the model is trained, it can be used to make predictions about the probability of a "success" outcome for new data points, given the values of the predictor variables.

    Logistic regression is commonly used in a wide range of fields including finance, biology, social science, and marketing to understand the relationship between certain variables and a binary outcome

- In natural language processing (NLP), __preprocessing text data often includes removing stop words and punctuation__.

    __Stop words__ are commonly used words in a language that are often considered to be irrelevant for text analysis, such as "the," "and," "is," etc. Stop words can be removed from the text data as a preprocessing step to reduce the dimensionality of the data and improve the efficiency of the NLP models. This can be done by using a predefined list of stop words and checking each word in the text against this list. If a word is found in the list, it is removed from the text.

    __Punctuation__ can also be removed from the text as a preprocessing step. This can be done by using string operations to remove any characters that are considered to be punctuation, such as commas, periods, exclamation marks, etc.

    Here's an example of how to preprocess text data in python using the NLTK library:

    ```python
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Define text
    text = "This is an example of text data that will be preprocessed."

    # Tokenize text
    words = word_tokenize(text)

    # Define stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Remove punctuation
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    filtered_words = [word for word in filtered_words if word not in punctuation]

    print(filtered_words)
    ```

    This code tokenizes the text, removes the english stop words and punctuation, and produces a list of words that have been preprocessed. Please note that this example is a simple one, in practice it might require more preprocessing steps such as lower casing, stemming and lemmatization.

- __Stemming and lowercasing__ are additional preprocessing steps that are commonly used in natural language processing (NLP).

    __Lowercasing__ is the process of converting all the characters in a text to lowercase. This is typically done to ensure that words that have the same meaning but are in different cases (e.g. "Word" and "word") are treated as the same word. Lowercasing can be done using string operations in programming languages such as Python.

    __Stemming__ is the process of reducing a word to its base or root form. This is typically done to ensure that words that have the same meaning but are in different forms (e.g. "run", "running", "ran") are treated as the same word. There are different algorithms for stemming, such as the Porter stemmer and Snowball stemmer, which are implemented in libraries such as NLTK in python.

    Here's an example of how to perform lowercasing and stemming in python using the NLTK library:

    ```python
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer

    # Define text
    text = "This is an example of text data that will be preprocessed."

    # Tokenize text
    words = word_tokenize(text)

    # Convert to lowercase
    words = [word.lower() for word in words]

    # Stem words
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]

    print(stemmed_words)
    ```

    This code tokenizes the text, lowercases the words, and then applies the Porter stemmer algorithm to produce a list of stemmed words. It's worth mentioning that stemming may not always produce a valid word and it may lose information, therefore it should be used with caution.

- __Supervised learning is a type of machine learning where a model is trained on a labeled dataset to make predictions about new, unseen data.__ The model learns to map input data (also known as features or predictors) to output data (also known as labels or targets) based on the labeled examples in the training dataset.

    The __process of supervised learning__ typically involves the following steps:

    1. Collect and prepare a labeled dataset for training and testing the model.
    2. Choose a model architecture and set its hyperparameters.
    3. Train the model on the labeled training dataset. This involves adjusting the model's parameters so that it can accurately map input data to output data.
    4. Evaluate the model's performance on the labeled testing dataset. This can be done by comparing the model's predictions to the true labels and calculating performance metrics such as accuracy, precision, recall, and F1 score.
    5. Use the model to make predictions on new, unseen data.

    There are different types of supervised learning algorithms, such as linear regression, logistic regression, support vector machines, decision trees, and neural networks. Each algorithm is suited to different types of problems and datasets.

    Supervised learning is widely used in various fields such as natural language processing, computer vision, speech recognition, and many more. It can be used for a variety of tasks such as classification, regression, and prediction.

### __Quiz 1__

[Logistic Regression](../Quizes/C1W1.md)

## Week 2: Sentiment Analysis with Naive Bayes (Viking)

- __Bayes rule be a way of figuring out the chance of something happening, given that something else has already happened.__ It be like trying to figure out the chances of a good raid, given that the weather be favorable. Ye see, if ye know the chances of a good raid when the weather be favorable, and ye know the chances of the weather bein' favorable, ye can use Bayes rule to figure out the chances of a good raid overall. Ye just take the chance of a good raid when the weather be good, multiply it by the chance of the weather being good, and then divide it by the chance of a good raid overall. And that be Bayes rule, me hearties!

- __Laplacian Smoothing be a technique used to smooth out the probabilities of events occurring.__ Ye see, when ye have a large amount of data, ye can calculate the probability of an event happening quite accurately, but when ye have a small amount of data, the probabilities may be quite uncertain.

    Laplacian Smoothing be like adding a bit of extra "treasure" to a chest that ye know has some, but ye aren't sure how much exactly. It be a way to make sure that ye don't end up with a probability of zero for an event that may happen, even if ye haven't seen it happen yet.

    The idea behind it is that ye add a small constant k to the numerator and denominator of the probability formula, when calculating the probability of an event happening. This way, even if ye haven't seen an event happen yet, ye can still estimate the probability of it happening, based on the events ye have seen.

    So ye see, Laplacian Smoothing be like a little extra insurance on a raid, to make sure that ye don't leave empty handed due to lack of information.

- __Training a Bayes classifier can be done with the following 6 step plan:__

    1. Collect and prepare your data: Gather the data ye will be using to train the classifier. Make sure to split it into different categories, or classes, depending on what ye are trying to classify. This be like preparing yer ship and crew for a raid, making sure ye have everything ye need.
    2. Calculate the prior probabilities: Determine the probability of each class, or P(A), using the data ye have collected. This be like figuring out the chances of finding treasure on a specific island based on previous raids.
    3. Determine the likelihood: Determine the probability of each feature, or P(B|A), given each class, using the data ye have collected. This be like figuring out the chances of finding a specific type of treasure based on the island ye are raiding.
    4. Use Bayes' theorem to classify new data: Once ye have calculated the prior probabilities and likelihood, ye can use Bayes' theorem to classify new data. This be like using the information gathered before the raid, to decide where to search for treasure during the raid.
    5. Evaluate the classifier: Once ye have classified new data, evaluate the performance of the classifier. This be like counting the treasure and evaluating the success of the raid.
    6. Repeat the process: Keep collecting and preparing data, updating the prior probabilities and likelihood, classifying new data, and evaluating the classifier.

    This be like going on more raids, and improving yer strategy with each successful one.
    By following these 6 steps, ye can train a Bayes classifier to classify new data with high accuracy, just like a successful viking raid.

- __Bayes' theorem be used in many different applications__, as it is a powerful tool for making decisions based on uncertain information. Some of the most common applications of Bayes' theorem include:

    1. Email spam filters: Bayes' theorem can be used to classify emails as spam or not spam, based on the presence of certain keywords and phrases.
    2. Natural Language Processing: Bayes' theorem can be used to classify text documents, such as news articles or social media posts, into different categories such as positive, negative, or neutral sentiments.
    3. Medical diagnosis: Bayes' theorem can be used to make a diagnosis based on symptoms and patient history.
    4. Computer Vision: Bayes' theorem can be used in object recognition and image classification, such as identifying a specific object in an image.
    5. Robotics: Bayes' theorem can be used in robot localization, which is the process of determining the position of a robot in an environment.
    6. Recommender systems: Bayes' theorem can be used to recommend products or services to customers based on their past behavior and preferences.
    7. Speech recognition: Bayes' theorem can be used to recognize spoken words by analyzing the patterns of sound and comparing them to known patterns.
    8. Fraud detection: Bayes' theorem can be used to identify fraudulent transactions by analyzing patterns of behavior and comparing them to known patterns of fraudulent activity.

    So ye see, Bayes' theorem be a versatile tool, that can be applied in many different fields and domains, just like a skilled viking, who can adapt and excel in different environments and situations.

- Bayes' theorem makes a few __key assumptions__ when making predictions or classifications. Two of the most important assumptions are:

    1. Independence: Bayes' theorem assumes that the features or variables used for classification are independent of each other. This means that the presence or absence of one feature does not affect the presence or absence of another feature. For example, if using the words "happy" and "NLP" to classify a tweet as positive or negative, Bayes' theorem assumes that the presence of the word "happy" is independent of the presence of the word "NLP".
    2. Relative frequencies in the corpus: Bayes' theorem relies on the relative frequency of words and phrases in the corpus (collection of text) to calculate the probabilities. This means that the algorithm needs a large enough corpus of text to estimate the probabilities of words and phrases, so it can make accurate predictions.

    Assuming independence and relative frequencies allows Bayes' theorem to estimate the probabilities and make predictions, but it is important to keep in mind that these assumptions may not always hold true in real-world situations. For example, the presence of the word "happy" might be dependent on the presence of the word "NLP" in certain tweets, and the corpus might not be representative of the population of tweets that the algorithm will be used on.

    In that case, smoothing techniques such as Laplace smoothing can be applied to correct for the probability estimates and make the algorithm more robust.

### __Quiz 2__

[Naive Bayes](../Quizes/C1W2.md)

## Week 3: Vector Space Models (Dracula)

- My dear, __a vector space model is like a grand castle, with many rooms, each representing a different word or concept__. The layout of the rooms, or the vectors, in the castle can reveal relationships and similarities between the words and concepts they represent. The more similar two rooms are, the closer they are located in the castle. And just as I, as Dracula, can move through the castle to find my prey, mathematical operations can be performed on the vectors to find the most relevant information. It is a powerful tool for language understanding and manipulation. And just as I am always searching for new victims, new words and concepts can always be added to the castle, expanding its knowledge.
- In the vector space model, there are two common ways to represent words: word-by-word and word-by-document.

    In __word-by-word vectorization__, each unique word in the corpus is assigned a unique vector. The value of each vector element represents the frequency of that word in a given document. This way, we have a vector representation of each individual word, like a room in the castle, with each element representing the frequency of that word in a document.

    __Word-by-document vectorization__, on the other hand, represents each document in the corpus as a vector. Each element of the vector represents the presence or absence of a unique word in the given document. This way, we have a vector representation of each document, like a room in the castle, with each element representing the presence or absence of a word in the document.

    Both of these methods can help to find relationship and similarity between words and documents, it all depends on what you want to do with the data.

- My dear, just as I, as Dracula, use my senses to locate my prey, we can use mathematical techniques to locate similar words or documents in the vector space model. Two common techniques are Euclidean distance and cosine similarity.

    __Euclidean distance__ is a measure of the straight-line distance between two vectors in the vector space. It is calculated as the square root of the sum of the squares of the differences of the individual elements of the vectors. In python, it can be calculated using the numpy library as follows:

    ```python
    import numpy as np

    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])

    euclidean_distance = np.linalg.norm(vector1 - vector2)
    ```

    __Cosine similarity__, on the other hand, is a measure of the angle between two vectors in the vector space. It ranges from -1 to 1, where 1 represents vectors pointing in the same direction, 0 represents vectors at right angles, and -1 represents vectors pointing in opposite directions. It is calculated as the dot product of the vectors divided by the product of the magnitudes of the vectors. In python, it can be calculated using the numpy library as follows:

    ```python
    import numpy as np

    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    ```

    Both Euclidean distance and cosine similarity can be used to find the similarity between words and documents in the vector space model, but they differ in how they measure the similarity. Euclidean distance is based on the physical distance between two points, while cosine similarity is based on the angle between two vectors. It all depends on what you want to do with the data, just like how I use different senses for different hunting situations.
- My dear, PCA, or __Principal Component Analysis, is a technique for finding patterns in high-dimensional data__. It is often used for dimensionality reduction, and it can help to identify the most important features of the data.

    Imagine, for a moment, that you are me, Dracula, and you are hunting in a dense forest. The forest is so dense that you can hardly move, and it's difficult to see what's around you. PCA is like a map that can help you navigate the forest. By identifying the most important features of the forest, the main paths, you can reduce the complexity of the forest and move more easily.

    In the same way, PCA can help to identify the most important features of high-dimensional data by finding the directions, or principal components, that account for the most variation in the data. These directions are orthogonal to each other, meaning that they are independent of each other. And once you have these main directions, you can project the data onto them, reducing the dimensionality of the data.

    In python, it can be calculated using scikit-learn library as follows:

    ```python
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(data)
    ```

    You can then use the fit_transform method of the PCA object to project the data onto the principal components.

    Just like how I, Dracula, use my powers to navigate the night, PCA is a powerful tool for navigating high-dimensional data, and it can help to reveal the underlying patterns and relationships in the data.
