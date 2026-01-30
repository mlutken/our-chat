import torch
import os
from sympy import false
from sympy.logic.boolalg import Boolean
from torch.utils.data import Dataset, IterableDataset, DataLoader
import collections
import datasets
from tokenizer_utils import *
from dataloader_pre_processors import *




class IterDataset_Base(IterableDataset):
    def __init__(self):
        super().__init__()
        self.forced_stop_ = False
        self.total_records_processed_ = 0
        self.records_processed_since_iter_create_ = 0
        self.debug_data_file_name_ = '/tmp/_our_streaming_records_debug.txt'
        self.write_text_to_debug_file_ = False
        self.dbg_print_text_ = False
        self.process_callback_ = None

    def processCallbackSet(self, process_callback):
        self.process_callback_ = process_callback

    def forceStop(self):
        self.forced_stop_ = True

    def debugPrintText(self, do_dbg_print_text):
        self.dbg_print_text_ = do_dbg_print_text

    def writeTextToDebugFile(self, do_write):
        self.write_text_to_debug_file_ = do_write

    def setDebugDataFileName(self, debug_data_file_name):
        self.debug_data_file_name_ = debug_data_file_name


    def doCheckEndPrematurely(self):
        if self.forced_stop_:
            print(f"FIXMENM doCheckEndPrematurely FORCED STOP!")
            return True

        if self.records_to_process_ == -1:  # -1 means process all records
            return false

        return self.total_records_processed_ >= self.records_to_process_

    def endPrematurely(self):
        if self.doCheckEndPrematurely():
            self.total_records_processed_ = 0
            return True

        return False


# ---------------------------------------
# --- IterDataset_TextFile_PreTrain ---
# ---------------------------------------
# https://medium.com/@amit25173/how-to-use-dataloader-with-iterabledataset-in-pytorch-an-advanced-practical-guide-898a49ace81c
# https://docs.python.org/3/library/collections.html#collections.deque
class IterDataset_TextFile_PreTrain(IterDataset_Base):
    def __init__(self, tokenizer, text_file_path, records_to_process, max_length, stride):
        super().__init__()
        self.tokenizer_ = tokenizer
        self.text_file_path_ = text_file_path
        self.records_to_process_ = records_to_process
        self.max_length_ = max_length
        self.stride_ = stride
        self.read_chunk_size_ = int(max_length/4)
        self.token_queue_capacity_ = 100000
        self.token_queue_ = collections.deque([], maxlen=self.token_queue_capacity_)

        # self.token_queue_ = collections.deque([])

        # print (f"FIXMENM queue_len: {self.queue_len()}")
        # print (f"FIXMENM queue capacity: {self.token_queue_.maxlen}")


    def queue_len(self):
        return len(self.token_queue_)

    def queue_empty(self):
        return len(self.token_queue_) > 0

    def __iter__(self):
        # print(" **** FIXMENM TextFile __iter__ create")
        self.records_processed_since_iter_create_ = 0
        # Open file in read mode and yield each line
        file_handle = open(self.text_file_path_, 'r')

        for line in file_handle:
            if self.endPrematurely():
                file_handle.close()
                print("--------- FIXMENM TextFile endPrematurely() ----------")
                return None, None
                break

            self.total_records_processed_ += 1
            self.records_processed_since_iter_create_ += 1

            if self.dbg_print_text_:
                print (f"TextFile.RECORD[{self.records_processed_since_iter_create_} / {self.total_records_processed_}] text[0:20]: '{line[0:20]}'")

            if self.process_callback_ is not None:
                line = self.process_callback_.process(line)

            tokens = self.tokenizer_.encode(line)

            for token in tokens:
                self.token_queue_.appendleft(token)

            while self.queue_len() > self.max_length_ +1:
                yield self.process_output()


        while not self.queue_empty():
            yield self.process_output()

        file_handle.close()

        # print("--------- FIXMENM TextFile Iteration done !!!  ----------")
        return None, None

    def preprocess(self, line):
        # Custom preprocessing logic here (e.g., parse JSON or CSV if necessary)
        return line # Simple example, removing newline characters

    def process_output(self):
        input_chunk = []
        target_chunk = []
        last_index =  self.max_length_ -1
        second_last_index =  last_index -1
        n = 0
        while self.queue_len() > 0 and (n < self.max_length_):
            token = self.token_queue_.pop()

            if  idIsNumber(token) and ( (n == 0) or (n >= second_last_index)):
                # print(f"A FIXMENM number token[{n}]: {token}, queue len: {self.queue_len()}")
                input_chunk.append(PADDING_ID)
                self.token_queue_.append(token) # Put NUMBER_ID token back into buffer
                # print(f"B FIXMENM number token[{n}]: {token}, queue len: {self.queue_len()}")
            else:
                input_chunk.append(token)
            n += 1

        while len(input_chunk) < self.max_length_:
            input_chunk.append(PADDING_ID)

        target_chunk = input_chunk[1:]
        target_chunk.append(PADDING_ID)

        input_chunk_tensor = torch.tensor(input_chunk)
        target_chunk_tensor = torch.tensor(target_chunk)

        # return input_chunk_tensor
        return input_chunk_tensor, target_chunk_tensor


def create_iter_loader_TextFile_PreTrain(tokenizer, textFilePath, records_to_process = -1, batch_size=4, max_length=256,
                                           stride=128, shuffle=False, drop_last=True,
                                           num_workers=0):
    if not os.path.isfile(textFilePath):
        return None

    dataset = IterDataset_TextFile_PreTrain(tokenizer, textFilePath, records_to_process, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# ------------------------------------------
# --- IterDataset_HuggingFace_PreTrain ---
# ------------------------------------------
# https://medium.com/@amit25173/how-to-use-dataloader-with-iterabledataset-in-pytorch-an-advanced-practical-guide-898a49ace81c
# https://docs.python.org/3/library/collections.html#collections.deque
class IterDataset_HuggingFace_PreTrain(IterDataset_Base):
    def __init__(self, tokenizer, hugging_face_uri, name, text_key, split, records_to_process, max_length, stride):
        super().__init__()
        self.tokenizer_ = tokenizer
        self.hugging_face_uri_ = hugging_face_uri
        if 'hf:' in self.hugging_face_uri_:
            self.hugging_face_uri_ = self.hugging_face_uri_.replace("hf:", "")

        self.records_to_process_ = records_to_process
        self.name_ = name
        self.text_key_ = text_key
        self.split_ = split
        self.max_length_ = max_length
        self.stride_ = stride
        self.read_chunk_size_ = int(max_length/4)
        self.token_queue_capacity_ = 100000
        self.token_queue_ = collections.deque([], maxlen=self.token_queue_capacity_)


    def queue_len(self):
        return len(self.token_queue_)

    def queue_empty(self):
        return len(self.token_queue_) > 0

    def __iter__(self):
        # print(" **** FIXMENM HuggingFace __iter__ create")
        if self.write_text_to_debug_file_:
            with open(self.debug_data_file_name_, 'w') as f:
                f.write("")

        fw = datasets.load_dataset(self.hugging_face_uri_, name=self.name_, split=self.split_, streaming=True)
        for record in fw:
            if self.endPrematurely():
                print("--------- FIXMENM HuggingFace endPrematurely() ----------")
                return self.process_output()
                break

            self.total_records_processed_ += 1
            self.records_processed_since_iter_create_ += 1
            record_dict = dict(record)

            text = record_dict[self.text_key_]

            if text == "":
                print("--------- FIXMENM HuggingFace empty text stop iterating ----------")
                return self.process_output()
                break


            if self.process_callback_ is not None:
                text = self.process_callback_.process(text)

            if self.write_text_to_debug_file_:
                with open(self.debug_data_file_name_, 'a') as f:
                    f.write(text)

            if self.dbg_print_text_:
                print (f"HuggingFace.RECORD[{self.total_records_processed_}] text[0:20]: '{text[0:20]}'")

            tokens = self.tokenizer_.encode(text)

            for token in tokens:
                self.token_queue_.appendleft(token)

            while self.queue_len() > self.max_length_ +1:
                yield self.process_output()


        while not self.queue_empty():
            yield self.process_output()

        print("--------- FIXMENM HuggingFace Iteration done !!!  ----------")
        return self.process_output()

    # # TODO: Seems unused !!!
    # def preprocess(self, text):
    #     # Custom preprocessing logic here (e.g., parse JSON or CSV if necessary)
    #     return text # Simple example, removing newline characters

    def process_output(self):
        input_chunk = []
        target_chunk = []
        last_index =  self.max_length_ -1
        second_last_index =  last_index -1
        n = 0
        # print(f"FIXMENM process_output, self.queue_len(): {self.queue_len()}, n: {n}")
        # print(f"FIXMENM process_output, len(input_chunk): {len(input_chunk)}, self.max_length_: {self.max_length_}")
        while self.queue_len() > 0 and (n < self.max_length_):
            token = self.token_queue_.pop()

            if  idIsNumber(token) and ( (n == 0) or (n >= second_last_index)):
                # print(f"A FIXMENM number token[{n}]: {token}, queue len: {self.queue_len()}")
                input_chunk.append(PADDING_ID)
                self.token_queue_.append(token) # Put NUMBER_ID token back into buffer
                # print(f"B FIXMENM number token[{n}]: {token}, queue len: {self.queue_len()}")
            else:
                input_chunk.append(token)
            n += 1

        while len(input_chunk) < self.max_length_:
            input_chunk.append(PADDING_ID)

        target_chunk = input_chunk[1:]
        target_chunk.append(PADDING_ID)

        input_chunk_tensor = torch.tensor(input_chunk)
        target_chunk_tensor = torch.tensor(target_chunk)

        # return input_chunk_tensor
        # print(f"FIXMENM process_output, input_chunk_tensor: {input_chunk_tensor}")
        return input_chunk_tensor, target_chunk_tensor


def create_iter_loader_HuggingFace_PreTrain(tokenizer, hugging_face_uri, name, text_key, split, records_to_process = -1, batch_size=4, max_length=256,
                                              stride=128, shuffle=False, drop_last=True,
                                              num_workers=0):
    dataset = IterDataset_HuggingFace_PreTrain(tokenizer, hugging_face_uri, name, text_key, split, records_to_process, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader



def create_loader_pretraining(tokenizer, resource_uri, name, text_key, split, records_to_process = -1, batch_size=4, max_length=256, stride=128, shuffle=False, drop_last=True, num_workers=0):
    if 'hf:' in resource_uri:
        return create_iter_loader_HuggingFace_PreTrain(
            tokenizer,
            resource_uri,
            name=name,
            text_key=text_key,
            split=split,
            records_to_process=records_to_process,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers)
    else:
        return create_iter_loader_TextFile_PreTrain(
            tokenizer,
            resource_uri,
            records_to_process,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers)
