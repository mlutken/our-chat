import json
import os
import torch

class TrainContinue():
    def __init__(self, cmd_args, continue_state_filename: str = "continue.our-chat.json"):
        self.cmd_args_ = cmd_args
        self.cmd_args_dict_ = vars(cmd_args)
        self.continue_state_filename_ = continue_state_filename
        self.state_dict_ = {}
        self.model_ = None
        self.device_ = "cpu"

        if self.cmd_args_dict_["train_uri"] is None:
            print ("ERROR: TrainContinue self.cmd_args_dict_.train_uri is None")
            self.cmd_args_dict_["train_uri"] = "ERROR_MISSING_TRAIN_URI"

        if not os.path.isfile(self.continue_state_filename_):
            self.write_new_state_dict_file()

        self.read_and_update_state_dict()


    def set_model(self, model, device):
        self.model_ = model
        self.device_ = device

    def save_model_weights(self):
        if self.model_ is None:
            return
        device = next(self.model_.parameters()).device
        print(f"Saving model to {self.cmd_args_.save_path} ...", end='')
        self.model_.to("cpu")
        torch.save(self.model_.state_dict(), self.cmd_args_.save_path)
        self.model_.to(device)
        print(f" Done saving model weights!")

    def can_continue(self):
        return self.cmd_args_.mode == "continue" and self.get_current_state_dict()['current_epoch'] < self.cmd_args_.epochs

    def get_continue_parameters(self):
        cd = self.get_current_state_dict()
        return cd['current_epoch'], cd['current_records_read']

    def update_callback(self, data_loader):
        # print(f"FIXMENM TrainContinue [{data_loader.epochNumber()}: {data_loader.recordsReadThisIteration()} / {data_loader.recordsProcessedThisIteration()}] TO process this iteration: {data_loader.recordsToProcessThisIteration()} TOTAL: {data_loader.totalRecordsProcessed()}")
        if data_loader.totalRecordsProcessed() % self.cmd_args_.eval_freq == 0:
            self.save_model_weights()
            self.state_dict_[self.cmd_args_.train_uri]["total_records_processed"] = data_loader.totalRecordsProcessed()
            self.state_dict_[self.cmd_args_.train_uri]["current_records_read"] = data_loader.recordsReadThisIteration()
            self.state_dict_[self.cmd_args_.train_uri]["current_epoch"] = data_loader.epochNumber()
            self.write_current_state_dict()

    def mark_training_done(self):
        self.state_dict_[self.cmd_args_.train_uri]["current_records_read"] = 0
        self.state_dict_[self.cmd_args_.train_uri]["current_epoch"] = self.cmd_args_.epochs
        self.write_current_state_dict()

    def read_and_update_state_dict(self):
        with open(self.continue_state_filename_, 'r') as json_file:
            self.state_dict_ = json.load(json_file)

        current_run_dict = self.state_dict_.get(self.cmd_args_.train_uri, None)

        if (not current_run_dict) or (self.cmd_args_.mode == "train"):
            self.state_dict_[self.cmd_args_.train_uri] = self.create_new_state_dict()
            self.write_current_state_dict()

    def write_current_state_dict(self):
        json_str = json.dumps(self.state_dict_, indent=4)
        with open(self.continue_state_filename_, "w") as f:
            f.write(json_str)

    def get_current_state_dict(self):
        return self.state_dict_[self.cmd_args_.train_uri]

    def create_new_state_dict(self):
        return {"initial_params": self.cmd_args_dict_, "current_epoch": 0, "current_records_read": 0}

    def write_new_state_dict_file(self):
        state_dict = {self.cmd_args_dict_["train_uri"]: self.create_new_state_dict()}
        json_str = json.dumps(state_dict, indent=4)
        # print(f"FIXMENM json_str:\n{json_str}")
        with open(self.continue_state_filename_, "w") as f:
             f.write(json_str)

