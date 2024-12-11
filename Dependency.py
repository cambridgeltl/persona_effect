import signal
import openai
openai.api_key = "YOUR API KEY"
import numpy as np
import torch
import os
import random


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


from tenacity import retry, wait_random_exponential, stop_after_attempt


def handler(signum, frame):
    raise TimeoutError


def call_model(prompt, model, timeout_seconds=10):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_seconds)
    logit_bias = {32: 20, 33: 20, 34: 20, 35: 20, 36: 20}

    try:
        chat_completion = openai.ChatCompletion.create(
            model=model,
            n=1,
            seed=42,
            max_tokens=1,
            temperature=1,
            logprobs=True,
            top_logprobs=5,
            messages=prompt,
            logit_bias=logit_bias

        )
        """

            chat_completion = openai.ChatCompletion.create(
            model=model,
            n=1,
            seed=42,
            max_tokens=1,
            temperature=0.1,
            messages=prompt
        )

        """
        return chat_completion
    except TimeoutError:
        return None  # Return a sentinel value indicating timeout
    except Exception as e:
        return None  # Return None for other exceptions to trigger a retry
    finally:
        signal.alarm(0)  # Always disable the alarm


# Decorator to handle retry logic for any exception
@retry(wait=wait_random_exponential(min=0.1, max=1), stop=stop_after_attempt(1000000))
def call_model_with_retry(prompt, model):
    result = call_model(prompt, model)
    if result is None:
        raise Exception("API call failed or timed out")  # Raise a general exception to trigger a retry
    return result
