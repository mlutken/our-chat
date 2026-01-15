import torch
import terminedia as TM
from nnutils import *

# NOTE: Work in progress. Not used currently. The training code in nnutils is used.

class BaseTrainer():
    def __init__(self, model, train_loader, val_loader, optimizer, device):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.eval_freq = 5
        self.eval_iter = 5
        self.do_validation_loss = True
        self.trainingStopRequested = False

    def evaluateModel(self):
        self.model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(self.train_loader, self.model, self.device, num_batches=self.eval_freq)
            val_loss = -1
            if self.do_validation_loss:
                val_loss = calc_loss_loader(self.val_loader, self.model, self.device, num_batches=self.eval_freq)
        self.model.train()
        return train_loss, val_loss

    def trainModel(self, num_epochs, start_context):
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        self.trainingStopRequested = False
        for epoch in range(num_epochs):
            if self.trainingStopRequested:
                break

            self.model.train()
            for input_batch, target_batch in self.train_loader:
                if self.trainingStopRequested:
                    break
                # print(f"FIXMENM train input_batch: {input_batch.shape}, target_batch: {target_batch.shape}")
                self.optimizer.zero_grad()
                loss = self.model.calcLossBatch(input_batch, target_batch, self.device)
                loss.backward()
                self.optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % self.eval_freq == 0:
                    train_loss, val_loss = self.evaluateModel()
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, "
                          f"Val loss {val_loss:.3f}"
                          )

                with TM.keyboard:
                    if (pressed := TM.inkey()) == "q":
                        print(f"INFO: Training stop requested!")
                        self.train_loader.dataset.forceStop()
                        self.val_loader.dataset.forceStop()
                        self.trainingStopRequested = True

            self.model.generateAndPrintSample(self.device, start_context)
        return train_losses, val_losses, track_tokens_seen


class FoundationTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        super().__init__(model, train_loader, val_loader, optimizer, device)

