import logging
import os
import time
import transformers
from typing import Iterable, List, Optional, Tuple, Union
from tqdm import tqdm

from lm_eval.api import utils
from lm_eval.api.model import TokenLM, TokenSequence


logging.getLogger("openai").setLevel(logging.WARNING)


def get_result(response: dict, ctxlen: int) -> Tuple[float, bool]:
    """Process results from OpenAI API response.

    :param response: dict
        OpenAI API Response
    :param ctxlen: int
        Length of context (so we can slice them away and only keep the predictions)
    :return:
        continuation_logprobs: float
            Log probabilities of continuation tokens
        is_greedy: bool
            whether argmax matches given continuation exactly
    """
    is_greedy = True
    logprobs = response["logprobs"]["token_logprobs"]
    continuation_logprobs = sum(logprobs[ctxlen:])

    for i in range(ctxlen, len(response["logprobs"]["tokens"])):
        token = response["logprobs"]["tokens"][i]
        top_tokens = response["logprobs"]["top_logprobs"][i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


def oa_completion(**kwargs):
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    import openai

    backoff_time = 3
    while True:
        try:
            return openai.Completion.create(**kwargs)
        except openai.error.OpenAIError:
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5


class OpenAICompletionsLM(TokenLM):
    """Implements a language model interface for OpenAI's Completions API.
    See: https://beta.openai.com/docs/api-reference/completions
    """

    def __init__(
        self,
        engine: str,
        device: Optional[str] = None,
        batch_size: Optional[int] = 20,
        user_defined_max_generation_length: Optional[int] = 256,
    ):
        """
        :param engine: str
            OpenAI API engine (e.g. `davinci`)
        """
        super().__init__()
        assert device is None, "Can't specify `device` in the OpenAI API."

        import openai

        self.engine = engine
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        # To make the annoying "Using pad_token, but it is not set yet." error go away
        self.tokenizer.pad_token = "<|endoftext|>"
        self.vocab_size = self.tokenizer.vocab_size

        self._user_defined_max_generation_length = user_defined_max_generation_length
        self._batch_size = batch_size  # TODO: adaptive batch size

        openai.api_key = os.environ["OPENAI_API_SECRET_KEY"]

    @property
    def eot_token(self) -> int:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 2048

    @property
    def user_defined_max_generation_length(self) -> int:
        return self._user_defined_max_generation_length

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: Iterable[int]) -> List[str]:
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(
        self,
        requests: List[Union[Tuple[str, str], TokenSequence, TokenSequence]],
    ) -> List[Tuple[float, bool]]:
        def _collate(x):
            # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
            # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
            # we care about and so we need some kind of backup for when it isn't
            tokens = x[1] + x[2]
            return -len(tokens), tuple(tokens)

        results = []
        reorder = utils.Reorderer(requests, _collate)
        for chunk in tqdm(
            list(utils.chunks(reorder.get_reordered(), self.batch_size)),
        ):
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                # user_defined_max_generation_length+1 because the API takes up to 2049 tokens, including the first context token
                input = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                # TODO: the logic is much simpler if we just look at the length of continuation tokens
                ctxlen = len(context_enc) - max(
                    0,
                    len(context_enc) + len(continuation_enc) - (self.max_length + 1),
                )
                inputs.append(input)
                ctxlens.append(ctxlen)

            responses = self._model_call(inputs)

            for response, ctxlen, (cache_key, context_enc, continuation_enc) in zip(
                responses.choices, ctxlens, chunk
            ):
                answer = get_result(response, ctxlen)
                results.append(answer)
                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return reorder.get_original(results)

    def greedy_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        def sameuntil_chunks(xs, size):
            ret = []
            last_until = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != last_until:
                    yield ret, last_until
                    ret = []
                    last_until = x[1]
                ret.append(x)

            if ret:
                yield ret, last_until

        results = []
        reorder = utils.Reorderer(requests, _collate)
        # TODO: more intelligent batching for heterogeneous `until`
        for chunk, request_args in tqdm(
            list(sameuntil_chunks(reorder.get_reordered(), self.batch_size))
        ):
            stop_sequences = request_args["stop_sequences"]
            max_generation_length = request_args["max_generation_length"]
            num_fewshot = request_args["num_fewshot"]

            assert isinstance(stop_sequences, list) or stop_sequences is None
            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(num_fewshot, int) or num_fewshot is None

            # TODO(jon-tow): This is most likely useless b/c `stop_sequences` is
            # never `None`; see `PromptSourceTask.construct_requests`.
            if stop_sequences is None or num_fewshot == 0:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.user_defined_max_generation_length
            else:
                max_tokens = max_generation_length

            inputs = []
            for context, _ in chunk:
                context_enc = self.tok_encode(context)
                input = context_enc[-(self.max_length - self.user_defined_max_generation_length) :]
                inputs.append(input)

            responses = self._model_generate(
                inputs=inputs,
                max_tokens=max_tokens,
                stop=until,
            )

            # Iterate thru the per-request responses.
            for response, (context, _request_args) in zip(responses.choices, chunk):
                sentence = response["text"]
                _stop_sequences = _request_args["stop_sequences"]
                _until = (
                    [self.eot_token] if _stop_sequences is None else _stop_sequences
                )
                for term in _until:
                    sentence = sentence.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, _until), sentence)
                results.append(sentence)
        return reorder.get_original(results)

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return oa_completion(
            engine=self.engine,
            prompt=inputs,
            echo=True,
            max_tokens=0,
            temperature=0.0,
            logprobs=5,
        )

    def _model_generate(
        self, inputs: TokenSequence, max_tokens: int, stop: Optional[List[str]] = None
    ) -> Union[TokenSequence, List[str]]:
        # NOTE: We don't need to add context size b/c OpenAI completion only
        # expects the max generation count portion.
        generations = oa_completion(
            engine=self.engine,
            prompt=inputs,
            max_tokens=max_tokens,
            temperature=0.0,
            logprobs=5,
            stop=stop,
        )
        return generations
