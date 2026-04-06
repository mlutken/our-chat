import torch
import os
import traceback
import gc
from sympy import false
import warnings
from sympy.logic.boolalg import Boolean
from torch.utils.data import Dataset, IterableDataset, DataLoader
import collections
import datasets
from tokenizer_utils import *
from dataloader_pre_processors import *
from itertools import islice




class IterDataset_Base(IterableDataset):
    def __init__(self):
        super().__init__()
        self.epoch_number_ = -1
        self.forced_stop_ = False
        self.total_records_processed_ = 0
        self.records_read_this_iteration_ = 0
        self.records_processed_this_iteration_ = 0
        self.debug_data_file_name_ = '/tmp/_our_streaming_records_debug.txt'
        self.write_text_to_debug_file_ = False
        self.dbg_print_text_ = False
        self.process_callback_ = None
        self.records_start_index_ = 0

    def epoch_started(self, epoch_number):
        self.epoch_number_ = epoch_number
        self.do_epoch_started_()

    def processCallbackSet(self, process_callback):
        self.process_callback_ = process_callback

    def forceStop(self):
        self.forced_stop_ = True

    def recordsReadThisIteration(self):
        return self.records_read_this_iteration_

    def recordsProcessedThisIteration(self):
        return self.records_processed_this_iteration_

    def totalRecordsProcessed(self):
        return self.total_records_processed_

    def recordsStartIndex(self):
        return self.records_start_index_

    def debugPrintText(self, do_dbg_print_text):
        self.dbg_print_text_ = int(do_dbg_print_text)

    def writeTextToDebugFile(self, do_write):
        self.write_text_to_debug_file_ = do_write
        self._handleDebugDataFileInit()

    def setDebugDataFileName(self, debug_data_file_name):
        self.debug_data_file_name_ = debug_data_file_name
        self._handleDebugDataFileInit()

    def doCheckEndPrematurely(self):
        if self.forced_stop_:
            # print(f"INFO IterDataset_Base::doCheckEndPrematurely FORCED STOP!")
            return True

        if self.records_to_process_ == -1:  # -1 means process all records
            # print(f"INFO IterDataset_Base All records in dataset processed!")
            return false

        if self.records_processed_this_iteration_ >= self.records_to_process_:
            # print(f"INFO IterDataset_Base All records in this iteration ({self.records_processed_this_iteration_} / {self.records_to_process_}) is processed processed!")
            return True

        return False

    def endPrematurely(self):
        if self.doCheckEndPrematurely():
            return True

        return False

    def do_epoch_started_(self):
        pass

    def _handleDebugDataFileInit(self):
        # Possibly start debug write file
        if self.write_text_to_debug_file_:
            with open(self.debug_data_file_name_, 'w') as f:
                print(f"INFO: Writing debug training data to {self.debug_data_file_name_}")
                f.write("DEBUG FILE START\n")



# ---------------------------------------
# --- IterDataset_TextFile ---
# ---------------------------------------
# https://medium.com/@amit25173/how-to-use-dataloader-with-iterabledataset-in-pytorch-an-advanced-practical-guide-898a49ace81c
# https://docs.python.org/3/library/collections.html#collections.deque
class IterDataset_TextFile(IterDataset_Base):
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


    def queue_len(self):
        return len(self.token_queue_)

    def queue_empty(self):
        return len(self.token_queue_) > 0

    def __iter__(self):
        self.records_read_this_iteration_ = 0
        self.records_processed_this_iteration_ = 0
        # Open file in read mode and yield each line
        file_handle = open(self.text_file_path_, 'r')

        for line in file_handle:
            if self.endPrematurely():
                file_handle.close()
                print("--------- FIXMENM TextFile endPrematurely() ----------")
                return None, None
                break

            self.total_records_processed_ += 1
            self.records_processed_this_iteration_ += 1
            self.records_read_this_iteration_ += 1

            # TODO: No skipping (self.records_start_index_) implemented for text fiels yet. See IterDataset_HuggingFace for inspiration for how to implement!

            if self.dbg_print_text_:
                if self.records_read_this_iteration_ % self.dbg_print_text_ == 0:
                    print (f"TextFile.RECORD[{self.records_read_this_iteration_} / {self.records_processed_this_iteration_}] text[0:20]: '{line[0:20]}'")

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


def create_iter_loader_TextFile(tokenizer, textFilePath, records_to_process = -1, batch_size=4, max_length=256,
                                           stride=128, shuffle=False, drop_last=True,
                                           num_workers=0):
    if not os.path.isfile(textFilePath):
        return None

    dataset = IterDataset_TextFile(tokenizer, textFilePath, records_to_process, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# ------------------------------------------
# --- IterDataset_HuggingFace ---
# ------------------------------------------
# https://medium.com/@amit25173/how-to-use-dataloader-with-iterabledataset-in-pytorch-an-advanced-practical-guide-898a49ace81c
# https://docs.python.org/3/library/collections.html#collections.deque
class IterDataset_HuggingFace(IterDataset_Base):
    def __init__(self, tokenizer, hugging_face_uri, name, text_key, split, records_to_process, records_start_index, max_length, stride):
        super().__init__()
        warnings.filterwarnings("ignore", category=ResourceWarning)
        self.hf_dataset_ = None
        self.hf_iterator_ = None
        self.tokenizer_ = tokenizer
        self.hugging_face_uri_ = hugging_face_uri
        if 'hf:' in self.hugging_face_uri_:
            self.hugging_face_uri_ = self.hugging_face_uri_.replace("hf:", "")

        self.records_to_process_ = records_to_process
        self.records_start_index_ = records_start_index
        self.records_to_read_ = self.records_start_index_ + self.records_to_process_
        self.name_ = name
        self.text_key_ = text_key
        self.split_ = split
        self.max_length_ = max_length
        self.stride_ = stride
        self.read_chunk_size_ = int(max_length/4)
        self.token_queue_capacity_ = 100000
        self.token_queue_ = collections.deque([], maxlen=self.token_queue_capacity_)
        self.hf_dataset_ = datasets.load_dataset(self.hugging_face_uri_, name=self.name_, split=self.split_, streaming=True)
        self.handle_iteration_done()


    def do_epoch_started_(self):
        self.handle_iteration_done()

    def iteration_done(self):
        if (self.hf_dataset_ is None) or (self.hf_iterator_ is None):
            return True

        if self.records_processed_this_iteration_ >= self.records_to_process_:
            print(f"!!! ITERATION DONE: IterDataset_Base All records in this iteration ({self.records_processed_this_iteration_} / {self.records_to_process_}) is processed processed !!!!")
            return True
        return False

    def handle_iteration_done(self):
        print(f"INFO: [{self.records_start_index_}:{self.records_to_process_}] handle_iteration_done [{self.records_read_this_iteration_} / {self.records_processed_this_iteration_}]")
        self._handleDebugDataFileInit()

        if self.hf_dataset_ is None:
            self.hf_dataset_ = datasets.load_dataset(self.hugging_face_uri_, name=self.name_, split=self.split_, streaming=True)

        self.records_read_this_iteration_ = 0
        self.records_processed_this_iteration_ = 0
        if self.records_start_index_ > 0:
            self.hf_iterator_ = self.hf_dataset_.skip(self.records_start_index_)
            self.records_read_this_iteration_ = self.records_start_index_
        else:
            self.hf_iterator_ = self.hf_dataset_.take(self.records_to_read_)

    def queue_len(self):
        return len(self.token_queue_)

    def queue_empty(self):
        return len(self.token_queue_) > 0

    def __iter__(self):
        # print(f"INFO: IterDataset_HuggingFace iterator create: self.process_callback_: {self.process_callback_}")

        if self.iteration_done():
            self.handle_iteration_done()

        sliced_dataset = islice(self.hf_dataset_, self.records_start_index_, None)
        sliced_dataset = islice(sliced_dataset, self.records_to_process_)

        try:
            for record in sliced_dataset:
                if self.endPrematurely():
                    print(f"INFO: IterDataset_HuggingFace::endPrematurely processed recs in iteration: {self.records_processed_this_iteration_}")
                    return self.process_output()

                self.records_read_this_iteration_ += 1
                self.records_processed_this_iteration_ += 1
                self.total_records_processed_ += 1
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
                    if self.records_read_this_iteration_ % self.dbg_print_text_ == 0:
                        print (f"HuggingFace.RECORD ({self.total_records_processed_}) this iteration [rec index / processed]: [{self.records_read_this_iteration_} / {self.records_processed_this_iteration_}] text[0:50]: '{text[0:50]}'")

                tokens = self.tokenizer_.encode(text)

                for token in tokens:
                    self.token_queue_.appendleft(token)

                while self.queue_len() > self.max_length_ +1:
                    yield self.process_output()
        except Exception as e:
            print(f"**** ERROR IterDataset_HuggingFace iteration Error: {e} ****")
        finally:
            # print("INFO Iteration terminated!")
            pass

        while not self.queue_empty():
            yield self.process_output()

        print("--------- FIXMENM HuggingFace Iteration done !!!  ----------")
        return self.process_output()


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


def create_iter_loader_HuggingFace(tokenizer, hugging_face_uri, name, text_key, split, records_to_process = -1, records_start_index = 0,
                                    batch_size=4, max_length=256, stride=128, shuffle=False, drop_last=True, num_workers=0):
    dataset = IterDataset_HuggingFace(tokenizer, hugging_face_uri, name, text_key, split, records_to_process, records_start_index, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader



def create_data_loader(tokenizer, resource_uri, name, text_key, split, records_to_process = -1, records_start_index = 0,
                       batch_size=4, max_length=256, stride=128, shuffle=False, drop_last=True, num_workers=0):
    if 'hf:' in resource_uri:
        return create_iter_loader_HuggingFace(
            tokenizer,
            resource_uri,
            name=name,
            text_key=text_key,
            split=split,
            records_to_process=records_to_process,
            records_start_index=records_start_index,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers)
    else:
        return create_iter_loader_TextFile(
            tokenizer,
            resource_uri,
            records_to_process,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers)
