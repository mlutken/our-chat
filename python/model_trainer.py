import torch
import terminedia as TM
from nnutils import *
from datetime import datetime as dt

# NOTE: Work in progress. Not used currently. The training code in nnutils is used.

class ModelTrainer():
    def __init__(self, model, train_loader, eval_train_loader, eval_validation_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.eval_train_loader = eval_train_loader
        self.eval_validation_loader = eval_validation_loader
        self.optimizer = optimizer
        self.device = device
        self.model_test_strings = []
        self.start_context = "<prompt> How to stay healthy? </prompt>"
        self.eval_freq = 100
        self.eval_batches = 5
        self.do_validation_loss = True
        self.trainingStopRequested = False
        self.train_losses       = []
        self.val_losses         = []
        self.track_tokens_seen  = []
        self.last_eval_records_count = 0
        self.total_records_processed = 0
        self.tokens_seen = 0
        self.cur_epoch = 0
        self.global_step = 0
        self.training_stop_requested_ = False
        self.training_active_ = False

    def training_completed(self):
        return not self.training_active_ and not self.training_stop_requested_

    def append_model_test_string(self, model_test_string):
        self.model_test_strings.append(model_test_string)

    def train_model_simple(self, num_epochs, start_epoch):
        self.train_losses       = []
        self.val_losses         = []
        self.track_tokens_seen  = []
        self.tokens_seen = 0
        self.total_records_processed = 0
        self.last_eval_records_count = self.train_loader.dataset.totalRecordsProcessed()
        self.global_step = -1
        self.cur_epoch = 0
        self.training_active_ = True

        self.model_evaluate_and_print()

        self.training_stop_requested_ = False
        for epoch in range(start_epoch, num_epochs):
            self.cur_epoch = epoch
            print(f"--------------------------------------------------------")
            print(f"--- Start epoch {epoch}  eval_freq: {self.eval_freq} ---")
            print(f"--------------------------------------------------------")
            if self.training_stop_requested_:
                break

            self.train_loader.dataset.epoch_started(epoch)
            self.model.train()
            current_batch_number = -1
            for input_batch, target_batch in self.train_loader:

                self.total_records_processed = self.train_loader.dataset.totalRecordsProcessed()
                current_batch_number += 1
                if self.training_stop_requested_:
                    break
                self.optimizer.zero_grad()
                loss = self.model.calcLossBatch(input_batch, target_batch, self.device)
                loss.backward()
                self.optimizer.step()
                self.tokens_seen += input_batch.numel()
                self.global_step += 1

                should_evaluate = self.total_records_processed >= (self.last_eval_records_count + self.eval_freq)
                # print(f"FIXMENM [{should_evaluate}] total records processed: [{self.total_records_processed} ] self.global_step: {self.global_step}, loss: {loss} tokens_seen: {self.tokens_seen}")

                if should_evaluate:
                    self.last_eval_records_count = self.total_records_processed
                    self.model_evaluate_and_print()

                with TM.keyboard:
                    if (pressed := TM.inkey()) == "q":
                        print(f"INFO: Training stop requested!")
                        self.train_loader.dataset.forceStop()
                        if self.eval_train_loader is not None:
                            self.eval_train_loader.dataset.forceStop()
                        if not self.eval_validation_loader is None:
                            self.eval_validation_loader.dataset.forceStop()
                        self.training_stop_requested_ = True

        self.training_active_ = False
        return self.train_losses, self.val_losses, self.track_tokens_seen

    def model_evaluate_and_print(self):
        train_loss, val_loss = evaluate_model(self.model, self.eval_train_loader, self.eval_validation_loader,
                                              self.device, self.eval_batches, True)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.track_tokens_seen.append(self.tokens_seen)
        ts_now = dt.now()
        date_time = ts_now.strftime('%Y-%m-%d T %H:%M:%S')

        print("******************************************************************************************************")
        print(
            f"EVALUATE [{date_time}]: Train loss {train_loss:.3f}, Total records: {self.train_loader.dataset.totalRecordsProcessed()} Epoch: {self.cur_epoch} (Step {self.global_step:06d}): "
            f"Rec index / processed in epoch: [{self.train_loader.dataset.recordsReadThisIteration()} / {self.train_loader.dataset.recordsProcessedThisIteration()}] "
            f"Val loss {val_loss:.3f}"
            )
        for test_string in self.model_test_strings:
            self.model.generateAndPrintSample(self.device, test_string)
        print("******************************************************************************************************")

    def calculate_train_loss(self, do_print=False):
        self.model.to(self.device)
        train_loss = float("nan")
        val_loss = float("nan")
        with torch.no_grad():
            train_loss = calc_loss_loader(self.eval_train_loader, self.model, self.device, num_batches=self.eval_batches)
            val_loss = calc_loss_loader(self.eval_validation_loader, self.model, self.device, num_batches=self.eval_batches)
            if do_print:
                print(f"Training loss   : {train_loss}")
                print(f"Validation loss : {val_loss}")
        return train_loss, val_loss



