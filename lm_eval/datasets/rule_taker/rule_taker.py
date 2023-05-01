# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
from pathlib import Path

import datasets

_FULL_VERSION = "V2020.2.5.0"
_VERSION = _FULL_VERSION.split(".", 1)[1]

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

_DESCRIPTION = """\
Allen AI ruletaker dataset (original)
"""

_HOMEPAGE = "https://allenai.org/data/ruletaker"

_LICENSE = "Apache 2.0"

_URL = "https://aristo-data-public.s3-us-west-2.amazonaws.com/ruletaker/rule-reasoning-dataset-V2020.2.5.zip"


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class RuleTaker(datasets.GeneratorBasedBuilder):
    """Allen AI ruletaker dataset"""

    VERSION = datasets.Version(_VERSION)
    SUBSET_PATH = f"rule-reasoning-dataset-{_FULL_VERSION}/original/"
    DEPTHS = [0, 1, 2, 3, 5]
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f"depth-{n}", version=datasets.Version(_VERSION), description=f"The required depth of reasoning is {n}") \
        for n in DEPTHS
    ]


    def _info(self):

        features = datasets.Features(
            {
                "context_id": datasets.Value("string"),
                "context": datasets.Value("string"),
                "question_id": datasets.Value("string"),
                "question": datasets.Value("string"),
                "label": datasets.Value("int32"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        url = _URL
        data_dir = Path(dl_manager.download_and_extract(url)) / self.SUBSET_PATH / self.config.name 
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir / "train.jsonl",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir / "dev.jsonl",
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir / "test.jsonl",
                    "split": "test"
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        key = 0
        with open(filepath, encoding="utf-8") as f:
            for row in f:
                data = json.loads(row)
                for q in data["questions"]:
                    yield key, {
                        "context_id": data["id"],
                        "context": data["context"],
                        "question_id": q["id"],
                        "question": q["text"],
                        "label": 1 if q["label"] else 0,
                    }
                    key += 1
