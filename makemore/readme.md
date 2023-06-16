# Makemore Language Model From Scratch

MakeMore is an advanced character-level language model that generates unique names like Dontel, Irot, Zhendi, and others. While these names sound authentic, they are not actual names. The model operates by treating each line as an individual example, analyzing them as sequences of individual characters. For instance, 'Reese' serves as one example, represented by a sequence of characters. This character-level approach forms the basis of MakeMore, as it models and predicts the subsequent character in a given sequence.

To accomplish this, MakeMore implements various character-level language models, ranging from basic bi-gram and back-of-work models to sophisticated approaches such as multilingual perceptrons, recurrent neural networks, and modern transformers. The transformer model we are developing is comparable to the renowned **GPT-2** transformer, signifying its cutting-edge capabilities

We can feed any database of strings to generate more of it. 

## Bi-gram
A bigram language model focuses on predicting the next character in a sequence based on the current character. It operates by considering only two characters at a time, disregarding any additional context or information. The model assumes that the likelihood of a character following another character depends solely on the preceding character.

The model's simplicity lies in its limited scope, as it only looks at the previous character to make predictions. This approach neglects the broader context and dependencies present in language. Therefore, it can be considered a weak language model.

Introduced the bigram character-level language model and explored various aspects such as model training, sampling, and evaluating model quality using negative log likelihood loss. Additionally, we trained the model using two distinct approaches that ultimately yielded the same outcome.

In the first approach, we computed the frequencies of all the bigrams and performed normalization. On the other hand, the second approach utilized the negative log likelihood loss as a guide to optimize the counts matrix or counts array, minimizing the loss within a gradient-based framework. Remarkably, both methods produced identical results.

It is worth noting that the gradient-based framework offers greater flexibility. Currently, our neural network is quite basic, involving a single previous character input passed through a single linear layer to compute the logits. However, this framework allows for future expansion and enhancement of the neural network architecture to tackle more complex tasks. 


## References

Inspiration, code snippets, etc.
* [The spelled-out intro to language modeling: building makemore by Andrej Karpathy](https://youtu.be/PaCmpygFfXo)

## makemore github repo:
* [micrograd](https://github.com/karpathy/makemore)

* [Bigram Wiki](https://en.wikipedia.org/wiki/Bigram)

* [One-hot Encoding](https://en.wikipedia.org/wiki/One-hot)

### [Pytorch](https://pytorch.org/)

