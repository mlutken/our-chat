import json
import hashlib
import sys
from pathlib import Path

class TrainContinue():
    def __init__(self, cmd_args, continue_state_filename: str = "continue.our-chat.json"):
        self.cmd_args_ = cmd_args
        self.cmd_args_dict_ = vars(cmd_args)
        self.continue_state_filename_ = Path(continue_state_filename)
        self.state_dict_ = {}

        if self.cmd_args_dict_["train_uri"] is None:
            print ("FIXMENM self.cmd_args_dict_.train_uri is None")
            self.cmd_args_dict_["train_uri"] = "ERROR_MISSING_TRAIN_URI"

        if not self.continue_state_filename_.exists():
            self.write_new_state_dict_file()

        self.read_and_update_state_dict()

    def info_message(self):
        s = ""
        if self.cmd_args_.mode == "continue":
            d = self.get_current_state_dict()
            s += f"\n--------------- Continuing run from epoch: {d['current_epoch']}, record: {d['current_records_read']}\n----------------------\n"
        return s

    def modify_args(self):
        if self.cmd_args_.mode != "continue":
            return self.cmd_args_

        return self.cmd_args_   # FIXMENM TODO

    def update_callback(self, data_loader):
        #print(f"FIXMENM TrainContinue [{data_loader.recordsReadThisIteration()} / {data_loader.recordsProcessedThisIteration()}] ")
        if data_loader.recordsProcessedThisIteration() % 100 == 0:
            self.state_dict_[self.cmd_args_.train_uri]["current_records_read"] = data_loader.recordsReadThisIteration()
            self.state_dict_[self.cmd_args_.train_uri]["current_epoch"] = data_loader.epochNumber()
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

