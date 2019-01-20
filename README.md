
# Deep Neuroevolution
Using genetic algorithms to fit neural network parameters in RL setting based on [Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning](https://arxiv.org/abs/1712.06567).
<p float="left">
  <img src="https://github.com/hannesliik/deep_neuroevolution/blob/master/examples/untrained.gif" width="275" />
  <img src="https://github.com/hannesliik/deep_neuroevolution/blob/master/examples/generation_10.gif" width="275" /> 
  <img src="https://github.com/hannesliik/deep_neuroevolution/blob/master/examples/generation_50.gif" width="275" />
</p>
We provide a framework for optimization with genetic algorithms. We have also provided implementations for use with OpenAI gym environments.

For an example of use, see `impl/run_lunarlander.py`.

For environments implementing the gym.Env interface, we have a parallelized evaluator. We implemented the framework for testing with PyTorch models, se we have implemented evolutionary strategies for PyTorch in particular, but left an abstract base class that implements common pipelines without PyTorch dependencies.

### Installation
`git clone https://github.com/hannesliik/deep_neuroevolution.git`

`pip install -e .`
## Dependencies
### PyTorch
Follow instructions from https://pytorch.org/get-started/locally/
### Box2D for LunarLander
`pip install box2d-py`
