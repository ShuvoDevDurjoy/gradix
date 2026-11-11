import numpy as np
import torch

from gradix.nn.modules import Conv2D
from gradix.tensor import Tensor


def check_case(kernel_size, stride, padding):
	seed = 465
	rng = np.random.default_rng(seed)
	input_data = rng.normal(size=(2, 2, 5, 6)).astype(np.float32)
	weight_data = rng.normal(size=(3, 2, kernel_size, kernel_size)).astype(np.float32)
	bias_data = rng.normal(size=(3,)).astype(np.float32)

	output_height = (5 + 2 * padding - kernel_size) // stride + 1
	output_width = (6 + 2 * padding - kernel_size) // stride + 1
	upstream_grad = np.broadcast_to(
		rng.normal(size=(2, 3, output_height, output_width)).astype(np.float32),
		(2, 3, output_height, output_width),
	).copy()

	engine = Conv2D(2, 3, kernel_size, stride=stride, padding=padding, use_bias=True)
	engine.weight = Tensor(weight_data.copy(), requires_grad=True)
	engine.bias = Tensor(bias_data.copy(), requires_grad=True)
	engine_input = Tensor(input_data.copy(), requires_grad=True)
	engine_output = engine(engine_input)
	(engine_output * Tensor(upstream_grad)).sum().backward()

	reference = torch.nn.Conv2d(
		2, 3, kernel_size, stride=stride, padding=padding, bias=True
	)
	with torch.no_grad():
		reference.weight.copy_(torch.from_numpy(weight_data))
		reference.bias.copy_(torch.from_numpy(bias_data))
	torch_input = torch.tensor(input_data, requires_grad=True)
	torch_output = reference(torch_input)
	(torch_output * torch.from_numpy(upstream_grad)).sum().backward()

	comparisons = {
		"output": (engine_output.data, torch_output.detach().numpy()),
		"input gradient": (engine_input.grad, torch_input.grad.numpy()),
		"weight gradient": (engine.weight.grad, reference.weight.grad.numpy()),
		"bias gradient": (engine.bias.grad, reference.bias.grad.numpy()),
	}
	for name, (actual, expected) in comparisons.items():
		if not np.allclose(actual, expected, rtol=1e-4, atol=1e-5):
			max_error = np.max(np.abs(actual - expected))
			raise AssertionError(
				f"{name} mismatch for kernel={kernel_size}, stride={stride}, "
				f"padding={padding}; max absolute error={max_error}"
			)

	print(
		f"PASS kernel={kernel_size}, stride={stride}, padding={padding}: "
		f"output shape={engine_output.shape}"
	)


if __name__ == "__main__":
	check_case(3, 1, 0)
	check_case(2, 2, 1)
	check_case(3, 1, 2)
	print("All Conv2D comparisons against PyTorch passed.")