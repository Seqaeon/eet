import torch
from nanochat.gpt import Linear

def test_linear_bias_fix():
    print("Testing Linear bias fix...")
    # Create a Linear layer with a specific weight and bias
    lin = Linear(2, 2)
    lin.weight.data = torch.eye(2)
    lin.bias.data = torch.tensor([1.0, 2.0])
    
    x = torch.zeros(1, 2)
    y = lin(x)
    
    # If bias is working, y should be [1.0, 2.0]
    expected = torch.tensor([[1.0, 2.0]])
    print(f"Input: {x}")
    print(f"Output: {y}")
    print(f"Expected: {expected}")
    
    assert torch.allclose(y, expected), f"Linear bias fix failed! Expected {expected}, got {y}"
    print("Linear bias fix verified!")

if __name__ == "__main__":
    test_linear_bias_fix()
