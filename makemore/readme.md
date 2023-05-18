# Makemore Language Model From Scratch

MakeMore is an advanced character-level language model that generates unique names like Dontel, Irot, Zhendi, and others. While these names sound authentic, they are not actual names. The model operates by treating each line as an individual example, analyzing them as sequences of individual characters. For instance, 'Reese' serves as one example, represented by a sequence of characters. This character-level approach forms the basis of MakeMore, as it models and predicts the subsequent character in a given sequence.

To accomplish this, MakeMore implements various character-level language models, ranging from basic bi-gram and back-of-work models to sophisticated approaches such as multilingual perceptrons, recurrent neural networks, and modern transformers. The transformer model we are developing is comparable to the renowned GPT-2 transformer, signifying its cutting-edge capabilities

We can feed any database of strings to generate more of it. 




## References

Inspiration, code snippets, etc.
* [The spelled-out intro to language modeling: building makemore by Andrej Karpathy](https://youtu.be/PaCmpygFfXo)

## makemore github repo:
* [micrograd](https://github.com/karpathy/makemore)

### [Pytorch](https://pytorch.org/)