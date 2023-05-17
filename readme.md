# Micrograd

Micrograd from scratch. A tiny Autograd engine. Short form of automatic gradient which implements the backpropagation algorithm. Backpropagation is a method to efficiently calculate the gradient of a loss function with respect to the weights of a neural network. This gradient calculation allows for iterative weight tuning, minimizing the loss function and improving the network's accuracy.


```python
from micrograd.engine import Value

# Initialize input values
a = Value(-4.0)
b = Value(2.0)

# Build the expression graph
c = a + b
d = a * b + b ** 3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e ** 2
g = f / 2.0
g += 10.0 / f

# Print the value of g after the forward pass
print(f'{g.data:.4f}')  # prints 24.7041, the outcome of this forward pass

# Perform backpropagation
g.backward()

# Print the gradients of a and b
print(f'{a.grad:.4f}')  # prints 138.8338, i.e., the numerical value of dg/da
print(f'{b.grad:.4f}')  # prints 645.5773, i.e., the numerical value of dg/db

```
Micrograd allows the construction of mathematical expressions using input values, such as **a** and **b**. These values are wrapped in **Value** objects provided by micrograd. The library supports various operations like addition, multiplication, exponentiation, offsetting, negation, squashing, and division.
By building an expression graph with **a** and **b** as inputs, the library creates a chain of operations that transform them into **c**, **d**, **e**, **f**, and **g**. Micrograd understands the structure of the expression graph, recognizing that c is the result of an addition operation and has a and b as child nodes.
#### In summary performs backpropagation to calculate the gradients of *g* with respect to *a* and *b*


## References

Inspiration, code snippets, etc.
* [The spelled-out intro to neural networks and backpropagation: building micrograd by Andrej Karpathy](https://youtu.be/VMj-3S1tku0)

## micrograd github repo:
* [micrograd](https://github.com/karpathy/micrograd)