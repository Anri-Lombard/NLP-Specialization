# Natural Language Processing with Classification and Vector Spaces

## Week 1: Sentiment Analysis with Logistic Regression

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

### __Quiz__

[Logistic Regression](../Quizes/C1W1.md)

## Week 2: Sentiment Analysis with Naive Bayes

