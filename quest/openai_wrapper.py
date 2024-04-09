import copy
import logging
import time
import os
from dataclasses import dataclass
from typing import Optional, Sequence
from multiprocessing.pool import ThreadPool

from tiktoken import encoding_for_model
from openai import OpenAI, APIError


@dataclass
class OpenAIDecodingArguments:
    max_tokens: int = 2560
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


DEFAULT_ARGS = OpenAIDecodingArguments()
DEFAULT_MODEL = "gpt-3.5-turbo-16k"
DEFAULT_SLEEP_TIME = 20


def openai_completion(
    messages,
    decoding_args: OpenAIDecodingArguments = DEFAULT_ARGS,
    model_name: str = DEFAULT_MODEL,
    sleep_time: int = DEFAULT_SLEEP_TIME,
    api_key: Optional[str] = None,
):
    decoding_args = copy.deepcopy(decoding_args)
    assert decoding_args.n == 1
    while True:
        try:
            client = OpenAI(api_key=api_key) if api_key else OpenAI()
            completions = client.chat.completions.create(
                messages=messages, model=model_name, **decoding_args.__dict__
            )
            break
        except APIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce" in str(e):
                decoding_args.max_tokens = int(decoding_args.max_tokens * 0.8)
                logging.warning(
                    f"Reducing target length to {decoding_args.max_tokens}, Retrying..."
                )
            else:
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)
    return completions.choices[0].message.content


def openai_batch_completion(
    batch,
    decoding_args: OpenAIDecodingArguments = DEFAULT_ARGS,
    model_name: str = DEFAULT_MODEL,
    sleep_time: int = DEFAULT_SLEEP_TIME,
    api_key: Optional[str] = None,
):
    completions = []
    with ThreadPool(len(batch)) as pool:
        results = pool.starmap(
            openai_completion,
            [
                (messages, decoding_args, model_name, sleep_time, api_key)
                for messages in batch
            ],
        )
        for result in results:
            completions.append(result)
    return completions


def openai_tokenize(
    text: str,
    model_name: str,
):
    encoding = encoding_for_model(model_name)
    return encoding.encode(text)


def openai_list_models(
    api_key: Optional[str] = None,
):
    if not api_key:
        return tuple()
    client = OpenAI(api_key=api_key)
    return tuple((m.id for m in client.models.list().data))


def openai_get_key(model_settings):
    env_key = os.getenv("OPENAI_API_KEY", None)
    local_key = model_settings.openai_api_key
    if local_key:
        return local_key
    return env_key
