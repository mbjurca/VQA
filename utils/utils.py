from torch.optim.lr_scheduler import _LRScheduler

class WarmupThenScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, after_warmup_scheduler, final_lr):
        self.warmup_epochs = warmup_epochs
        self.after_warmup_scheduler = after_warmup_scheduler
        self.final_lr = final_lr
        super(WarmupThenScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs + 1 and self.warmup_epochs != 0:
            # Linearly increase the learning rate during warm-up
            initial_lr = self.final_lr * 0.001
            final_lr = self.final_lr
            alpha = self.last_epoch / self.warmup_epochs
            return [initial_lr + (final_lr - initial_lr) * alpha]
        else:
            # After warm-up, use the LR schedule of the provided scheduler
            self.after_warmup_scheduler.last_epoch = self.last_epoch - self.warmup_epochs  # Increment last_epoch
            return self.after_warmup_scheduler.get_lr()
