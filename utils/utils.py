import os

import torch


def log_gradient_norm(model, writer, step, mode, norm_type=2):
    """Writes model param's gradients norm to tensorboard"""
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    writer.add_scalar(f"Gradient/{mode}", total_norm, step)


def save_checkpoint(model, optimizer, start_time, epoch):
    """Saves specified model checkpoint."""
    target_dir = os.path.join("checkpoints", str(start_time))
    os.makedirs(target_dir, exist_ok=True)
    # Save model weights
    save_path_model = os.path.join(target_dir, f"model_{epoch}.pth")
    save_path_optimizer = os.path.join(target_dir, f"optimizer_{epoch}.pth")
    torch.save(model.state_dict(), save_path_model)
    torch.save(optimizer.state_dict(), save_path_optimizer)
    print("Model saved.")
