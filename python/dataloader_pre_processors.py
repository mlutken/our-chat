import torch

from string_utils import *


class ProcessCallbackBase():
    def __init__(self):
        self.text_ = ""
        self.end_of_data_reached_ = False

    def set_reached_end_of_data(self):
        self.end_of_data_reached_ = True

    def process(self, text):
        if text == "":
            self.end_of_data_reached_ = True
        self._add_text(text)
        return self._get_text()

    def _add_text(self, text):
        self.text_ += text

    # This is the one you need to override!
    def _get_text(self):
        text = self.text_
        self.text_ = ""
        return text


# Used for 'hf:Fredithefish/Instruction-Tuning-with-GPT-4-RedPajama-Chat'
class ProcessHumanBot1(ProcessCallbackBase):
    # This is the one you need to override!
    def _get_text(self):
        hb_pair = self._get_next_human_bot_pair()
        if hb_pair == "":
            return ""

        pr_pair = hb_pair.split("<bot>:")
        if len(pr_pair) != 2:
            return ""

        prompt_resp = format_prompt_response(pr_pair[0], pr_pair[1])
        return prompt_resp

    def _get_next_human_bot_pair(self):
        if self.end_of_data_reached_:
            p_start, p_end = parse_between(self.text_, "<human>:", "", False)
        else:
            p_start, p_end = parse_between(self.text_, "<human>:", "<human>:", True)

        if p_start == -1:
            return ""

        hb_pair = self.text_[p_start:p_end]
        if len(hb_pair) >= len(self.text_):
            self.text_ = ""
        else:
            self.text_ = self.text_[p_end:]

        return hb_pair




g_dataloaders = {
    'hf:HuggingFaceFW/finewiki'                                     : { 'dataset_name': 'en'                , 'dataset_key': 'text', 'pre_process': None    },
    'hf:HuggingFaceFW/fineweb'                                      : { 'dataset_name': 'CC-MAIN-2025-26'   , 'dataset_key': 'text', 'pre_process': None    },
    'hf:Fredithefish/Instruction-Tuning-with-GPT-4-RedPajama-Chat'  : { 'dataset_name': 'default'           , 'dataset_key': 'text', 'pre_process': ProcessHumanBot1()    },
}



def dataloader_lookup(uri):
    if uri in g_dataloaders:
        return g_dataloaders[uri]

    return None