# Mind Map
- Text Preprocessing
  - Tokenization
  - Stop Words Removal
  - Text Normalization (lowercasing, removing punctuations, special characters)
  - Lemmatization/Stemming
- Word Representation/Embeddings
  - Standard 
    - One Hot Encoding
    - Label Encoding
    - Bag of words
    - TF IDF
  - Deep Learning Based
    - Word2Vec
    - Skip-Gram
    - CBOW
    - Glove
    - Contextual Embeddings
      - BERT
      - GPT
  - Statistical-Based:
    - n-Gram
- Tasks
  - POS Tagging
    - Hidden Markov's Model
    - Verterbi Algorithm
  - Named Entity Recognition (NER)
  - Text Generation
  - Text classification
  - Text Summarization - Abstractive vs Extractive Summarization
  - Question Answering - Answering based on given text
  - Coreference Resolution - Link pronouns to entities
  - Speech to text (ASR) - converting spoken words into text
  - Text to speech (TTS) - converting text into spoken words
  - Dialogue System - Chatbots, Virtual Assistants
 

# 1. Word Representation/Embeddings
## Mind Map
- Standard 
  - One Hot Encoding
  - Label Encoding
  - Bag of words
  - TF IDF
- Deep Learning Based
  - Word2Vec
    - Skip-Gram
    - CBOW
  - Glove
  - Contextual Embeddings
    - BERT
    - GPT
- Statistical-Based:
  - n-Gram
## 1.1 Standard 
### 1.1.1 One Hot Encoding
- It Created binary vectors for each word in a vocab.
- Each word is represented by a vector that is all zeroes except for a single 1, corresponding to the index of that word in vocab.
- Example:-
  - cat: [1, 0, 0, 0]
  - dog: [0, 1, 0, 0]
  - fish: [0, 0, 1, 0]
  - bird: [0, 0, 0, 1]
### 1.1.2 Bag of words
- Bag of Words (BoW) is a method where each document (or sentence) is represented by a vector. The vector indicates the frequency (or presence) of each word in the document.
- The problem with BoW is that it creates a very large, sparse vector (since most words won't appear in a given document) and doesn't capture word semantics (meaning).
- Example:-
  - Suppose we have two documents:
    - Doc 1: "The cat sat on the mat."
    - Doc 2: "The dog lay on the mat."
  - First, we create a vocabulary: ["the", "cat", "sat", "on", "mat", "dog", "lay"].
  - The Bag of Words vector for each document would look like this:
    - Doc 1: [2, 1, 1, 1, 1, 0, 0] (2 occurrences of "the", 1 occurrence of "cat", etc.)
    - Doc 2: [2, 0, 0, 1, 1, 1, 1]
### 1.1.3 TF IDF
TF-IDF improves on BoW by counting word frequency and considering the importance of a word in a corpus of documents.
- Words that appear frequently in a document get a higher score (TF), but words that appear in many documents get a lower score (IDF). This helps filter out common words (like "the") that don't have much meaning.
- **Formula**
```math
 TF(w, d) = \frac{number of times w appears in d}{total number of words in d}
IDF(w) = log\frac{total number of documents}{Number of documents containing w}
TF-IDF(w, d) = TF(w, d) * IDF(w)
```
- Example
  - For the word "cat" in Doc 1 from the BoW example:
    - TF: 1/6(it appears once in a document with 6 words)
    - IDF: If "cat" appears in only one of the two documents, IDF=log(2/1)=log(2)
    - TF-IDF for "cat" would be 1/6×log(2).
### 1.1.4 Label Encoding
- Label Encoding is a technique to convert categorical data (data that can be divided into discrete categories) into numerical values. Each unique category or label in the data is assigned a unique integer. It is a simple method often used for converting labels (e.g., words or categories) into numerical form so that machine learning algorithms, which generally operate on numerical data, can process them.
- Label Encoding assigns a distinct integer to each category given a set of categories. The assignment is typically arbitrary, meaning it doesn't represent any inherent relationship between the categories.
- Example
  - apple = 0, banana = 1, cherry = 2
## 1.2 Deep Learning Based
### 1.2.1 Word2Vec
- For a given word it returns a vector such that these vectors are semantically similar to the similar words.
- **Target Word** The word for which we want to predict the surrounding context words.
- **Context Word** The words surrounding the target word within a defined window size.
- Contextual words are very useful in understanding the target words and vice versa.
- **Comparison between CBOW & skip-gram:**
  1. CBOW is comparatively faster to train than skip-gram (as CBOW has to train only one softmax).
  2. CBOW is better for frequently occurring words (because if a word occurs more often it will have more training words to train).
  3. Skip-gram is slower but works well for the smaller amount of data then CBOW.
  4. Skip-gram works well for less frequently occurring words than CBOW.
  5. CBOW is a simpler problem than the Skip-gram (because in CBOW we just need to predict the one focus word given many context words).
#### Mind Map
- CBOW
- Skip-Gram
- ![image](https://github.com/user-attachments/assets/624af3fe-e589-4ede-a8c5-a839f4ae7db7)
#### 1.2.1.1 CBOW
- Vocabulary of words V of size v and we use **one-hot encoding to represent each word**.
  - The vector for word w is of v dimensions.
  - **Core Idea:-** Given the **K** context words can we predict the target word?
  - **Input:-** **K** context words each of dimension v
  - **Output:-** Target word of dimension v 
#### 1.2.1.2 Skip-Gram
- Vocabulary of words V of size v and we use **one-hot encoding to represent each word**.
  - The vector for word w is of v dimensions.
  - **Core Idea:-** Given the **K** context words can we predict the target word?
  - **Input:-** Target word of dimension v
  - **Output:-** **K** context words each of dimension v
  
 
### 1.2.2 Glove
- 
### 1.2.3 Contextual Embeddings
- Word representations that consider the context in which a word appears, allowing the same word to have different meanings depending on its usage.
- BERT: Bidirectional, great for understanding context in tasks that require comprehension.
- GPT: Unidirectional, focused on generating coherent text based on the previous context.
#### 1.2.4.1 BERT
- Architecture: Uses a transformer architecture with attention mechanisms.
- Bidirectional: Considers the context from both the left and right sides of a word, allowing it to understand the word's meaning in a sentence more comprehensively.
- Pre-training:
  - Masked Language Model (MLM): Randomly masks words in a sentence and trains the model to predict them.
  - Next Sentence Prediction (NSP): Trains the model to determine if two sentences are consecutive.
- Use Cases: Great for tasks like text classification, question answering, and named entity recognition.
- https://github.com/premanshsharma/Generative_AI_Notes/blob/main/BERT.md
#### 1.2.4.2 GPT
- Architecture: Also uses a transformer architecture but is unidirectional.
- Unidirectional: Reads text from left to right, predicting the next word based on previous words.
- Pre-training: Trained using a language modeling objective, focusing solely on predicting the next word in a sentence.
- Use Cases: Excellent for text generation, dialogue systems, and creative writing tasks.
- https://github.com/premanshsharma/Generative_AI_Notes/blob/main/GPT.md
## 1.3 Statistical-Based:
### 1.3.1 n-Gram
- An n-gram is a contiguous sequence of n items (words, characters, etc.) from a given sample of text or speech. The value of n determines the type of n-gram:
  - Unigram (n=1): Individual words.
  - Bigram (n=2): Pairs of consecutive words.
  - Trigram (n=3): Triplets of consecutive words.
  - n-gram (n=k): General case for k consecutive words.
- **Theoretical Foundation of n-Grams**
  - The n-gram model is based on the Markov assumption, which states that the probability of a word depends only on the previous n-1 words.
  ```math
  P(w1,w2,......,wN) = \prod_{i=1}^{N} P(wi|wi-n+1,..........,wi-1)
  ```
  - where P(wi|wi-n+1,..........,wi-1) is the joint probability of the word sequence
  - wi represents the ith word in the sequence
  - n is the size of the n-gram
- **Estimating n-Gram Probabilities**
  - To estimate the probabilities P(wi|wi-n+1,..........,wi-1), we can use frequency counts:
    ```math
    P(wi|wi-n+1,..........,wi-1) = \frac{C(wi-n+1,..........,wi-1,wi)}{C(wi-n+1,........,wi-1)}
    ```
  - C(A) denotes the count of occurrences of the sequence A.
  - To address the problem of zero probabilities for unseen n-grams, various smoothing techniques can be applied:
  ```math
    P(wi|wi-n+1,..........,wi-1) = \frac{C(wi-n+1,..........,wi-1,wi)+1}{C(wi-n+1,........,wi-1)+V}
    ```
- Generating Word Embeddings Using n-grams
  - Vectorization: Once n-grams are identified and counted, they can be vectorized using techniques such as:
    - Count Vectorization: Directly representing n-grams as feature vectors based on their counts in the text.
    - TF-IDF Vectorization: Using term frequency-inverse document frequency (TF-IDF) to weigh n-grams, thus emphasizing more informative n-grams while down-weighting common ones.
  - Embedding Space: After vectorization, these feature vectors can be fed into embedding models or neural networks (like CNNs or RNNs) to learn dense representations. This allows for more complex relationships between n-grams and their contexts to be learned.
- Types of n-Gram Models
  - Language Modeling: Predicting the next word in a sequence based on the previous n−1 words. Used in applications like predictive text input and speech recognition.
  - Text Classification: Utilizing n-grams as features in classification algorithms, where the presence or frequency of n-grams can help classify documents or sentiments.
  - Machine Translation: The sequence of words helps translate phrases while preserving context.
- Challenges with n-Grams
  - Sparsity: As n increases, the number of possible n-grams grows exponentially, leading to sparsity in data.
  - Memory Usage: Storing n-grams, especially for higher values of n, can consume significant memory.
  - Context Limitation: The Markov assumption limits the model to only consider a fixed context window.
- Basic Example
  - Example: Bigram Model
    - Consider the following sentence:
      - "The cat sat on the mat."
    - Tokenization: We split the sentence into words:
      - Words=[The, cat, sat, on, the, mat]
    - Generate Bigrams: We can form bigrams by taking pairs of consecutive words:
      - Bigrams=[(The, cat),(cat, sat),(sat, on),(on, the),(the, mat)]
    - Count Bigrams: Let's count occurrences in a larger corpus:
      - C(The, cat)=2
      - C(cat, sat)=3
      - C(sat, on)=1
      - C(on, the)=4
      - C(the, mat)=1
    - Calculate Probabilities: For example, the probability of "cat" given "The" is calculated as:
```math
P(cat|The) = \frac{C(The, cat)}{C(The) = frac{2}{5} = 0.4}
```
---------------------------------------

- It is a neural network architecture
- It aims to predict the context words surrounding a given target word.
- Model is designed to learn word representations (embeddings) by maximizing the probability of context words given a target word.
- The Skip-Gram model predicts context words given a target word.
- It maximizes the probability of context words using the softmax function.
- The model is trained using negative log-likelihood and updates word vectors through backpropagation.
- The resulting word embeddings capture semantic relationships based on context.
- **Example**
  - Consider the sentence: "The cat sat on the mat."
    - Target Word: "cat"
    - Context Window Size: 2
  - Context Words: "The", "sat", "on", "the".
    - For the target word "cat", the Skip-Gram model will try to maximize:
      - P("The"∣"cat")×P("sat"∣"cat")×P("on"∣"cat")×P("the"∣"cat")
- **Mathematical Representation**
  - **Objective:** Given a target word w<sub>t</sub>, the model predicts context words w<sub>t-j</sub> and w<sub>t+j</sub> where j is within a certain range defined by the context window size c.
  - **Probability Calculation**
    - ∏<sub>-c<=j<=c, j!= 0</sub> P(w<sub>t+j</sub>|w<sub>t</sub>)
    - P(w<sub>t+j</sub>|w<sub>t</sub>   represents the probability of observing a context word w<sub>t+j</sub> given the target word w<sub>t</sub>.
  - **Softmax Function** To compute P(w<sub>t+j</sub>|w<sub>t</sub>) we use softmax function

![image](https://github.com/user-attachments/assets/d2bd7289-adb1-4b79-a0c5-75987e5d36eb)
  - **Training the Model**:-
    - The training objective is to minimize the negative log-likelihood of the predicted context words, given the target word:
    - ![image](https://github.com/user-attachments/assets/e5d9e20f-2258-4afd-b4bd-33552a024f29)
    - **Backpropagation:** The model uses backpropagation and stochastic gradient descent to update the word vectors based on the loss calculated. The weights of the network (word embeddings) are adjusted iteratively to minimize the loss function.
