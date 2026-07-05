import torch
import torch.nn.functional as F
from nanochat.gpt import GPTConfig
from nanochat.eet import EarlyExitGPT

def test_eet_depth_affine():
    device = "cpu"
    print("Testing EET Depth-Conditional Affine Scaling...")
    
    config = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_global_router=True,
        eet_depth_affine=True,
        eet_loss_variant='quality',
    )
    
    model = EarlyExitGPT(config).to(device)
    model.train()
    
    # Check parameters exist
    assert hasattr(model, 'exit_gamma'), "exit_gamma not registered"
    assert hasattr(model, 'exit_beta'), "exit_beta not registered"
    assert len(model.exit_gamma) > 0, "exit_gamma is empty"
    
    B, T = 2, 8
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    model.zero_grad(set_to_none=True)
    loss = model(x, y, eet_do_route=True, eet_phase=2)
    assert not torch.isnan(loss)
    loss.backward()
    
    # Check that gamma and beta received gradients
    assert model.exit_gamma[0].grad is not None, "exit_gamma did not receive gradients"
    assert model.exit_beta[0].grad is not None, "exit_beta did not receive gradients"
    print("✓ Depth-Conditional Affine verified successfully!")


def test_eet_learned_schedule():
    device = "cpu"
    print("Testing EET Learned Scheduling Priors...")
    
    config = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_global_router=True,
        eet_learned_schedule=True,
        eet_loss_variant='quality',
    )
    
    model = EarlyExitGPT(config).to(device)
    model.train()
    
    assert hasattr(model, 'exit_schedule_logits'), "exit_schedule_logits not registered"
    assert model.exit_schedule_logits is not None
    assert model.exit_schedule_logits.shape == (model.n_exits,), "exit_schedule_logits shape mismatch"
    
    B, T = 2, 8
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    model.zero_grad(set_to_none=True)
    loss = model(x, y, eet_do_route=True, eet_phase=2)
    assert not torch.isnan(loss)
    loss.backward()
    
    assert model.exit_schedule_logits.grad is not None, "exit_schedule_logits did not receive gradients"
    print("✓ Learned Exit Schedule verified successfully!")


def test_eet_departure_summary():
    device = "cpu"
    print("Testing EET Departure Summary Injection...")
    
    config = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_global_router=True,
        eet_compute_skip=True,
        eet_departure_summary=True,
        eet_loss_variant='quality',
    )
    
    model = EarlyExitGPT(config).to(device)
    model.train()
    
    assert hasattr(model, 'departure_gate'), "departure_gate not registered"
    assert len(model.departure_gate) > 0, "departure_gate is empty"
    
    B, T = 2, 8
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    model.zero_grad(set_to_none=True)
    loss = model(x, y, eet_do_route=True, eet_phase=3)
    assert not torch.isnan(loss)
    loss.backward()
    
    assert model.departure_gate[0].grad is not None, "departure_gate did not receive gradients"
    print("✓ Departure Summary Injection verified successfully!")


def test_eet_route_consistency():
    device = "cpu"
    print("Testing EET Route Consistency Loss...")
    
    config = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_route_consistency_lambda=0.5,
        eet_loss_variant='quality',
    )
    
    model = EarlyExitGPT(config).to(device)
    model.train()
    
    assert hasattr(model, 'vocab_route_ema'), "vocab_route_ema not registered"
    assert model.vocab_route_ema is not None
    assert model.vocab_route_ema.shape == (config.vocab_size, model.n_exits), "vocab_route_ema shape mismatch"
    
    # Initialize EMA buffer with uniform exit probability
    model.vocab_route_ema.fill_(1.0 / model.n_exits)
    
    B, T = 2, 8
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    model.zero_grad(set_to_none=True)
    loss = model(x, y, eet_do_route=True, eet_phase=2)
    assert not torch.isnan(loss)
    loss.backward()
    
    # Check that route consistency did not produce NaN
    assert not torch.isnan(loss)
    print("✓ Route Consistency Loss verified successfully!")


def test_eet_dense_distill():
    device = "cpu"
    print("Testing EET Concurrent Dense Distillation...")
    
    config = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_dense_distill_interval=1,
        eet_dense_distill_lambda=0.5,
        eet_loss_variant='quality',
    )
    
    model = EarlyExitGPT(config).to(device)
    model.train()
    
    B, T = 2, 8
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    # Get dense targets
    with torch.no_grad():
        dense_x = model._compute_dense_logits(x)
        
    assert dense_x.shape == (B, T, config.n_embd), "dense_x shape mismatch"
    
    model.zero_grad(set_to_none=True)
    loss = model(x, y, eet_do_route=True, eet_phase=2, eet_dense_x=dense_x)
    assert not torch.isnan(loss)
    loss.backward()
    
    print("✓ Concurrent Dense Distillation verified successfully!")


def test_eet_capacity_anneal():
    device = "cpu"
    print("Testing EET Capacity Annealing...")
    
    config = GPTConfig(
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        vocab_size=128,
        sequence_len=32,
        use_eet=True,
        eet_min_exit_layer=0,
        eet_global_router=True,
        eet_compute_skip=True,
        eet_loss_variant='quality',
    )
    
    model = EarlyExitGPT(config).to(device)
    model.train()
    
    assert model.config.eet_target_active_frac == 0.125
    
    # Test updating the target active fraction config
    model.config.eet_target_active_frac = 0.4
    assert model.config.eet_target_active_frac == 0.4
    
    B, T = 2, 8
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    model.zero_grad(set_to_none=True)
    loss = model(x, y, eet_do_route=True, eet_phase=3)
    assert not torch.isnan(loss)
    loss.backward()
    
    print("✓ Capacity Annealing verified successfully!")
