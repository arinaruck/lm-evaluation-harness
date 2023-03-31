"""
PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification in 7 languages
https://arxiv.org/abs/1908.11828
"""
import typing

from lm_eval.api.task import PromptSourceTask


_CITATION = """
@InProceedings{pawsx2019emnlp,
  title = {{PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification}},
  author = {Yang, Yinfei and Zhang, Yuan and Tar, Chris and Baldridge, Jason},
  booktitle = {Proc. of EMNLP},
  year = {2019}
}
"""


class PAWSX(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "paws-x"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

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


class PAWSXEn(PAWSX):
    DATASET_NAME = "en"

class PAWSXFr(PAWSX):
    DATASET_NAME = "fr"

class PAWSXEs(PAWSX):
    DATASET_NAME = "es"

class PAWSXDe(PAWSX):
    DATASET_NAME = "de"

class PAWSXJa(PAWSX):
    DATASET_NAME = "ja"

class PAWSXKo(PAWSX):
    DATASET_NAME = "ko"

class PAWSXZh(PAWSX):
    DATASET_NAME = "zh"


PAWSX_TASKS = [
    PAWSXEn,
    PAWSXFr,
    PAWSXEs,
    PAWSXDe,
    PAWSXJa,
    PAWSXKo,
    PAWSXZh,
]


def construct_tasks() -> typing.Dict[str, PAWSX]:
    """
    Returns a dictionary of tasks keyed by task name, for example:
        "paws-x_en" will dispatch to the paws-x English.
    """
    tasks = {}
    for task_class in PAWSX_TASKS:
        benchmark = task_class.DATASET_PATH
        lang = task_class.DATASET_NAME
        tasks[f"{benchmark}_{lang}"] = task_class
    return tasks