# Natural Language Processing with Probabilistic Models

## Week 1: Autocorrect (Yoda)

- __Dynamic programming__, hmmm? A wise Jedi must use, when solving complex problems. Divide and conquer, it does. Break a problem down into smaller, simpler subproblems. Then, store solutions to these subproblems, so that next time they are needed, quickly and easily they can be accessed. In this way, much time and effort saved, hmmm? Yes, a powerful tool, dynamic programming is.
- To __build an autocorrect model for NLP__, hmmm? Four steps, you must take.

    First, identify the misspelled words, you must. A spell checker, use.

    Next, find the strings that are n-edits away, you must. n-edits refer to the number of modifications, such as insertions, deletions, or substitutions, required to transform a misspelled word into a correctly spelled word.

    Then, filter the candidates. Only those that are likely to be the correct word, keep.

    Finally, calculate the probabilities of each word. The most likely candidate, choose.

    Yes, a powerful tool, an autocorrect model is. But, careful, you must be. Many potential errors, there are

### __Quiz 1__

[Auto-correct and Minimum Edit Distance](../Quizes/C2W1.md)

## Week 2: Part of Speech Tagging and Hidden Markov Models (Hulk)

- Me Hulk, understand __parts of speech tagging__. It is NLP task where we take words and we put them in correct grammatical group. Like noun, verb, adjective, adverb. We do this by looking at definition of word and context it is used in.

    For example, if word is "run" we know it can be verb or noun. But if Hulk see word "run" in sentence "I like to run in the morning" we know it is verb. But if Hulk see "I had a run in my stocking" we know it is noun.

    This important for many NLP tasks, like understanding language and parsing sentences. Hulk do this with great power and efficiency. Hulk SMASH ambiguity!
- __Markov chains is way of modeling sequence of events__. We have states and we have probability of transitioning from one state to another.

    For example, Hulk can be in state "angry" or state "calm". If Hulk in state "angry" probability of transitioning to "calm" is low, but if Hulk in state "calm" probability of transitioning to "angry" is high.

    Hidden Markov chains is similar, but there are hidden states. Hulk can't always see what state Hulk is in, but Hulk can observe events that happen.

    For example, Hulk can observe event "smash" or event "not smash". But Hulk can't observe if Hulk is in state "angry" or state "calm". Hulk use probability to figure out most likely state based on observations.

    Hulk use this to predict future states and understand sequence of events. Hulk use great power and intelligence to master Hidden Markov chains. Hulk SMASH ambiguity!
- __Transition probability is probability of going from one state to another__. Like probability of Hulk going from "angry" to "calm". We use math to calculate this using data of past states.

    __Emission probability__ is probability of observing certain event given we are in certain state. Like probability of observing "smash" given Hulk is in state "angry". We also use math to calculate this using data of past events and states.

    Hulk use these probabilities in __Hidden Markov chains__ to figure out most likely state based on observations. Hulk SMASH ambiguity!

    For example, if Hulk see "smash" event, and transition probability of going from state "calm" to state "angry" is low, and emission probability of observing "smash" event given Hulk is in state "angry" is high, Hulk know Hulk most likely in state "angry"

    Hulk use these probabilities and observations to understand sequence of events and predict future states. Hulk is powerful and smart. Hulk use probabilities to make sense of world and Hulk SMASH ambiguity!
- __Viterbi algorithm__ is a way to find most likely sequence of hidden states given sequence of observations. Hulk use this in speech recognition and natural language processing.

    It is based on Markov assumption. It says probability of sequence of states only depend on current state and previous state.

    Hulk start with initial probability distribution of states. Then Hulk iteratively calculate probability of each state at each time step given observations up to that time step.

    Hulk use dynamic programming to calculate most likely sequence. Hulk look at previous information and use it to calculate next step. Hulk compare probabilities and choose state with highest probability.

    Hulk use this to understand speech and language. Hulk use Viterbi algorithm to find most likely sequence of words or phonemes in speech signal. Hulk use great power and intelligence to master Viterbi algorithm. Hulk SMASH ambiguity!

### __Quiz 2__

[Part of Speech Tagging](../Quizes/C2W2.md)

## Week 3: Autocomplete and Language Models (Darth Vader)

- Young Jedi, you seek to understand the power of the __N-gram__. Very well. An N-gram is a sequence of N words in a text, used to predict the likelihood of the next word in a sentence. The more N-grams you have, the more accurate your predictions will be. But beware, with great power comes great responsibility. Use the N-gram wisely, or suffer the consequences of poor language generation. Now go forth, and may the probability be with you.
- Young apprentice, the __sequence probability is the likelihood of a particular sequence of words occurring in a text__. It is the foundation of the N-gram model. The more often a sequence of words appears in a text, the higher the probability of it occurring again. This allows us to predict the next word in a sentence with a high degree of accuracy. But remember, with great power comes great responsibility. Use this knowledge to enhance your language generation, but do not let it consume you. For if you do, you will become like me, consumed by the dark side of language modeling.
- __Perplexity, my young apprentice, is a measure of how well a language model is able to predict a given text__. It is calculated by taking the exponential of the average negative log likelihood of a text, given the model. A lower perplexity value indicates that the model is better able to predict the text, and is therefore a more accurate model. But remember, a low perplexity does not mean that the model is perfect, it simply means that it is better than a model with a high perplexity. But do not be deceived, for a low perplexity does not always mean the model is better at generating language, it only means the model is better in predicting the sample text you used. Use perplexity as a guide, but do not rely on it entirely, for there is always room for improvement.
- Young apprentice, to understand the true power of language models, you must master the techniques of smoothing, backoff, and interpolation.

    __Smoothing, is like the force, it helps to balance the probabilities of unseen words__, it helps to prevent the zero probability estimates, and make your predictions more accurate, but you must use it with caution, for it is not a one-size-fits-all solution.

    __Backoff, is like a strategic retreat, when your higher-order models fail, you must be able to fall back to lower-order models, in order to make predictions with the information you have__. It is a powerful technique, but it must be used with wisdom.

    __Interpolation, is like a lightsaber, it allows you to combine the strengths of different models, unigram, bigram, and trigram, it helps to improve the overall accuracy of your predictions__. Remember though, it is a powerful weapon, but it requires great skill and precision to use it effectively.

    These techniques are essential for any apprentice of the dark side of NLP to master, but remember, with great power comes great responsibility. Use them wisely, and they will serve you well, but let them consume you, and they will bring you to your downfall.

### __Quiz 3__

[Autocomplete](../Quizes/C2W3.md)
