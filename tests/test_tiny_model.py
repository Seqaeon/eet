import torch
from nanochat.gpt import GPT, GPTConfig

def test_tiny_model_forward():
    # Use a tiny n_embd that would have triggered the bug (n_embd < 32)
    config = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16, 
        vocab_size=128,
        sequence_len=32
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = GPT(config).to(device)
    model.init_weights()
    
    # Create dummy input
    idx = torch.randint(0, config.vocab_size, (1, 8), device=device)
    
    print("Running forward pass...")
    try:
        logits = model(idx)
        print("Forward pass successful!")
        assert logits.shape == (1, 8, config.vocab_size)
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        raise e

if __name__ == "__main__":
    test_tiny_model_forward()
