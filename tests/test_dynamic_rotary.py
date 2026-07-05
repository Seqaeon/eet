import torch
from nanochat.gpt import GPT, GPTConfig

def test_dynamic_rotary():
    # Use a small sequence_len for initialization
    config = GPTConfig(
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=16, 
        vocab_size=128,
        sequence_len=32 # This will precompute only 320 tokens (32 * 10)
    )
    
    device = "cpu"
    model = GPT(config).to(device)
    model.init_weights()
    
    initial_cache_size = model.cos.size(1)
    print(f"Initial rotary cache size: {initial_cache_size}")
    
    # Create input longer than the initial cache (e.g., 512 tokens)
    long_seq_len = 512
    idx = torch.randint(0, config.vocab_size, (1, long_seq_len), device=device)
    
    print(f"Running forward pass with sequence length {long_seq_len}...")
    try:
        logits = model(idx)
        print("Forward pass successful!")
        
        new_cache_size = model.cos.size(1)
        print(f"New rotary cache size: {new_cache_size}")
        
        assert new_cache_size >= long_seq_len, "Cache should have grown to at least long_seq_len"
        assert logits.shape == (1, long_seq_len, config.vocab_size)
        
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        raise e

if __name__ == "__main__":
    test_dynamic_rotary()
