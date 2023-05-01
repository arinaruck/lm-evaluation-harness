# modified from ef8d4b9f0b16468b5d3ed51ba227a88275323b51 commit by gentaiscool
"""
XNLI is an evaluation corpus for language transfer and cross-lingual sentence classification in 15 languages.
https://arxiv.org/abs/1809.05053
Homepage: None, Repo: https://github.com/facebookresearch/XNLI
"""
import typing

from lm_eval.api.task import PromptSourceTask


_CITATION = """\
@misc{clark2020transformers,
      title={Transformers as Soft Reasoners over Language}, 
      author={Peter Clark and Oyvind Tafjord and Kyle Richardson},
      year={2020},
      eprint={2002.05867},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


class RuleTaker(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "rule_taker"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]
    
    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]


class RuleTakerDepth0(RuleTaker):
    DATASET_NAME = "depth-0"

class RuleTakerDepth1(RuleTaker):
    DATASET_NAME = "depth-1"

class RuleTakerDepth2(RuleTaker):
    DATASET_NAME = "depth-2"

class RuleTakerDepth3(RuleTaker):
    DATASET_NAME = "depth-3"

class RuleTakerDepth5(RuleTaker):
    DATASET_NAME = "depth-5"


RuleTaker_TASKS = [
    RuleTakerDepth0,
    RuleTakerDepth1,
    RuleTakerDepth2,
    RuleTakerDepth3,
    RuleTakerDepth5,
]


def construct_tasks() -> typing.Dict[str, RuleTaker]:
    """
    Returns a dictionary of tasks keyed by task name, for example:
        "rule_taker_depth-0" will dispatch to the 0 hop reasoning task
    """
    tasks = {}
    for task_class in RuleTaker_TASKS:
        benchmark = task_class.DATASET_PATH
        hops = task_class.DATASET_NAME
        tasks[f"{benchmark}_{hops}"] = task_class
    return tasks