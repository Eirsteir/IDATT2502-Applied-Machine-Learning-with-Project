import torch

W_init = torch.tensor([[0.0]], requires_grad=True)
b_init = torch.tensor([[0.0]], requires_grad=True)

class SigmoidModel:
	
	def __init__(self, W=W_init, b=b_init):
		self.W = W
		self.b = b
	
	def logits(self, x):
		return x @ self.W + self.b

	# Predictor
	def f(self, x):
		return torch.sigmoid(self.logits(x))

	# Cross Entropy loss
	def loss(self, x, y):
	  return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x),y)