# src/rl/callbacks.py
import logging
from tqdm import tqdm
try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    # Fallback if building a purely custom training loop without SB3
    class BaseCallback:
        def __init__(self, verbose=0): pass

class ProgressBarCallback(BaseCallback):
    """
    Custom callback for displaying a tqdm progress bar during RL training.
    """
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self) -> None:
        """Fired before the first environment step."""
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Agent", unit="step")

    def _on_step(self) -> bool:
        """Fired at every environment step."""
        # Update the progress bar by 1 step
        self.pbar.update(1)
        
        # Optional: You can add custom metrics to the progress bar here
        # Example: self.pbar.set_postfix({'Current Best IC': self.locals.get('best_ic', 0)})
        
        return True

    def _on_training_end(self) -> None:
        """Fired when training is completed."""
        if self.pbar:
            self.pbar.close()