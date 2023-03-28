# modified from ef8d4b9f0b16468b5d3ed51ba227a88275323b51 commit by gentaiscool
"""
XNLI is an evaluation corpus for language transfer and cross-lingual sentence classification in 15 languages.
https://arxiv.org/abs/1809.05053
Homepage: None, Repo: https://github.com/facebookresearch/XNLI
"""
import typing

from lm_eval.api.task import PromptSourceTask


_CITATION = """
@inproceedings{roemmele2011choice,
  title={Choice of plausible alternatives: An evaluation of commonsense causal reasoning},
  author={Roemmele, Melissa and Bejan, Cosmin Adrian and Gordon, Andrew S},
  booktitle={2011 AAAI Spring Symposium Series},
  year={2011},
  url={https://people.ict.usc.edu/~gordon/publications/AAAI-SPRING11A.PDF},
}
}"""


class XCOPA(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "xcopa"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]
    
    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]


class XCOPAEt(XCOPA):
    DATASET_NAME = "et"

class XCOPAId(XCOPA):
    DATASET_NAME = "id"

class XCOPAHt(XCOPA):
    DATASET_NAME = "ht"

class XCOPAIt(XCOPA):
    DATASET_NAME = "it"

class XCOPAQu(XCOPA):
    DATASET_NAME = "qu"

class XCOPASw(XCOPA):
    DATASET_NAME = "sw"

class XCOPATa(XCOPA):
    DATASET_NAME = "ta"

class XCOPATh(XCOPA):
    DATASET_NAME = "th"

class XCOPATr(XCOPA):
    DATASET_NAME = "tr"

class XCOPAVi(XCOPA):
    DATASET_NAME = "vi"

class XCOPAZh(XCOPA):
    DATASET_NAME = "zh"

XCOPA_TASKS = [
    XCOPAEt,
    XCOPAId,
    XCOPAHt,
    XCOPAIt,
    XCOPAQu,
    XCOPASw,
    XCOPATa,
    XCOPATh,
    XCOPATr,
    XCOPAVi,
    XCOPAZh,
]


def construct_tasks() -> typing.Dict[str, XCOPA]:
    """
    Returns a dictionary of tasks keyed by task name, for example:
        "xcopa_it" will dispatch to the xcopa Italian.
    """
    tasks = {}
    for task_class in XCOPA_TASKS:
        benchmark = task_class.DATASET_PATH
        lang = task_class.DATASET_NAME
        tasks[f"{benchmark}_{lang}"] = task_class
    return tasks