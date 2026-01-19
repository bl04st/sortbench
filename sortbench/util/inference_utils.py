import os
import time
import traceback
import random
import warnings
import string
import statistics
import math

from google import genai
from util.result_utils import check_if_result_available
from nltk.corpus import wordnet

from openai import OpenAI, InternalServerError
import anthropic

_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "o3-mini", "gpt-4.1-mini", "gpt-5.1_reasoning_none", "gpt-5.1_reasoning_low", "gpt-5-mini_reasoning_minimal"]
_INNCUBE_MODELS = ["llama3.1", "gemma2", "qwen2.5", "deepseekr1"]
_ANTROPIC_MODELS = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001_reasoning_disabled", "claude-sonnet-4-5-20250929_reasoning_disabled", "claude-haiku-4-5-20251001_reasoning_enabled", "claude-sonnet-4-5-20250929_reasoning_enabled"]
_GOOGLE_GEMINI_MODELS = ["gemini-2.5-flash"]

_THINKING_MODELS_WITH_OUTPUT = ["claude-haiku-4-5-20251001_reasoning_enabled", "claude-sonnet-4-5-20250929_reasoning_enabled"]
_THINKING_MODELS_WITH_TOKEN_OUTPUT = ["gpt-5.1_reasoning_low"]

TRANSFORM_STRUCTURE_LIST_BENCHMARK_TYPES = ["sort", "sort-descending", "reverse", "insert", "pop", "filter-lower", "filter-higher"]
TRANSFORM_VALUES_LIST_BENCHMARK_TYPES = ["uppercase", "square"]
SINGLE_RESULT_BENCHMARK_TYPES = ["count", "index", "sum", "min", "max", "product", "any", "all"]

def is_thinking_model_with_thinking_output(model):
    return model in _THINKING_MODELS_WITH_OUTPUT

def is_thinking_model_with_thinking_summary(model):
    return model in _THINKING_MODELS_WITH_TOKEN_OUTPUT

def is_model_supported(model):
    """
    Check if a model is supported by sortbench.
    
    Parameters:
    - model (str): the model name
    """
    return model in _OPENAI_MODELS+_INNCUBE_MODELS+_ANTROPIC_MODELS+_GOOGLE_GEMINI_MODELS

def is_benchmark_type_supported(benchmark_type):
    """
    Check if a benchmark type is supported by sortbench.

    Parameters:
    - benchmark_type (str): the benchmark type
    """
    return benchmark_type in TRANSFORM_STRUCTURE_LIST_BENCHMARK_TYPES + TRANSFORM_VALUES_LIST_BENCHMARK_TYPES + SINGLE_RESULT_BENCHMARK_TYPES

def get_single_result_benchmark_types():
    """
    Get the list of supported single result benchmark types.
    """
    return SINGLE_RESULT_BENCHMARK_TYPES

def get_transform_structure_list_benchmark_types():
    """
    Get the list of supported transform structure benchmark types.
    """
    return TRANSFORM_STRUCTURE_LIST_BENCHMARK_TYPES

def get_transform_values_list_benchmark_types():
    """
    Get the list of supported transform values benchmark types.
    """
    return TRANSFORM_VALUES_LIST_BENCHMARK_TYPES

def sort_list_with_google_gemini_api(unsorted_list, model, system_prompt=None, prompt=None):
    """
    Calls the Google Gemini API to sort a list.

    Parameters:
    - unsorted_list (list): the list to be sorted
    - api_key (str): the Google Gemini API key
    - model (str): the model to use for inference
    """

    if system_prompt is None:
        system_prompt = "Your task is to sort a list according to the common sorting of the used data type in Python. The output must only contain the sorted list and nothing else. The format of the list must stay the same."
    if prompt is None:
        prompt = f"Sort the following list: {unsorted_list}"

    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()

    response = client.models.generate_content(
        model=model,
        contents=f"{system_prompt}\n{prompt}"
    )
    sorted_list = response.text
    return sorted_list

def sort_list_with_antropic_api(unsorted_list, api_key, model, reasoning_effort_param='disabled', system_prompt=None, prompt=None, max_tokens=2000, max_reasoning_tokens=2000):
    """
    Calls the Antropic API to sort a list.

    Parameters:
    - unsorted_list (list): the list to be sorted
    - api_key (str): the Antropic API key
    - model (str): the model to use for inference
    - system_prompt (str): the system prompt to use
    - prompt (str): the prompt to use
    """
    
    if system_prompt is None:
        system_prompt = "Your task is to sort a list according to the common sorting of the used data type in Python. The output must only contain the sorted list and nothing else. The format of the list must stay the same."
    if prompt is None:
        prompt = f"Sort the following list: {unsorted_list}"
    if reasoning_effort_param is not None:
        reasoning_effort_list = ['disabled', 'enabled']
        reasoning_effort = reasoning_effort_param if reasoning_effort_param in reasoning_effort_list else 'disabled'
        if reasoning_effort == 'disabled':
            thinking = {"type": reasoning_effort}
        else:
            thinking = {"type": reasoning_effort, "budget_tokens": max_reasoning_tokens}
            max_tokens = max_reasoning_tokens*2
    else:
        reasoning_effort = 'disabled'
        thinking = {"type": reasoning_effort}

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=1,
        system=system_prompt,
        thinking=thinking,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    if reasoning_effort == 'enabled':
        thinking_text = ''
        final_text = ''
        for block in message.content:
            if block.type == "thinking":
                thinking_text += block.thinking + "\n"
            elif block.type == "text":
                final_text += block.text
        sorted_list = final_text + "<thinking_block_begin>" + thinking_text
    elif reasoning_effort == 'disabled':
        sorted_list = message.content[0].text
    else:
        raise ValueError("Unexpected message content format")
    return sorted_list


def sort_list_with_openai_api(unsorted_list, api_key, model, reasoning_effort_param='none', max_tokens=4000, url=None, use_streaming=False, system_prompt=None, prompt=None, max_attempts=1):
    """
    Calls the OpenAI API to sort a list.

    Parameters:
    - unsorted_list (list): the list to be sorted
    - api_key (str): the OpenAI API key
    - model (str): the model to use for inference
    - url (str): the URL of the OpenAI API endpoint
    - system_prompt (str): the system prompt to use
    - prompt (str): the prompt to use
    - max_attempts (int): the maximum number of attempts to make
    """

    # setup system prompt and prompt
    if system_prompt is None:
        system_prompt = "Your task is to sort a list according to the common sorting of the used data type in Python. The output must only contain the sorted list and nothing else. The format of the list must stay the same."
    if prompt is None:
        prompt = f"Sort the following list: {unsorted_list}"
    if reasoning_effort_param is not None:
        reasoning_effort_list = ['none', 'minimal', 'low', 'medium', 'high']
        reasoning_effort = reasoning_effort_param if reasoning_effort_param in reasoning_effort_list else 'none'
        if reasoning_effort in ['low', 'medium', 'high']:
            max_tokens = max_tokens*2
    else:
        reasoning_effort = 'none'

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        if url is None:
            client = OpenAI(api_key=api_key)
        else:
            client = OpenAI(api_key=api_key, base_url=url)
        try:
            kwargs = {
                "model": model,
                "instructions": system_prompt,
                "max_output_tokens": max_tokens,
                "stream": use_streaming,
                "input": prompt,
                "reasoning": { "effort": reasoning_effort }
            }
            response = client.responses.create(**kwargs)
            if use_streaming:
                collected_tokens = []
                for event in response:
                    if event.type == "response.output_text.delta":
                        collected_tokens.append(event.delta)
                    elif event.type == "response.error":
                        raise RuntimeError(event.error)
                sorted_list = ''.join(collected_tokens)
            else:
                if reasoning_effort != 'none' and reasoning_effort != 'minimal':
                    reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
                    return response.output_text.strip() + "<thinking_token_number>" + str(reasoning_tokens)
                else:
                    return response.output_text.strip()
        except Exception as e:
            print(f"Exception running inference: {e}")
            print()
            print(unsorted_list)
            if attempts == max_attempts:
                print("Waiting 60 seconds before next sequence...")
                time.sleep(60)
                raise RuntimeError()
            else:
                print("Waiting 60 seconds before next attempt...")
                time.sleep(60)
    
    return sorted_list

def call_llm_model_api(model, unsorted_list, system_prompt=None, prompt=None):
    """
    API call to the specified LLM model.
    """

    if model in _OPENAI_MODELS:
        api_key = os.getenv("OPENAI_API_KEY")
        model_split = model.split('_')
        model_name = model_split[0]
        if len(model_split) > 2:
            model_reasoning_effort = model_split[2]
        else:
            model_reasoning_effort = 'none'
        return sort_list_with_openai_api(unsorted_list, api_key, reasoning_effort_param=model_reasoning_effort, model=model_name, max_attempts=2, system_prompt=system_prompt, prompt=prompt, max_tokens=3000)
    elif model in _INNCUBE_MODELS:
        api_key = os.getenv("INNCUBE_API_KEY")
        endpoint_url = "https://llms.innkube.fim.uni-passau.de"
        return sort_list_with_openai_api(unsorted_list, api_key, model=model, url=endpoint_url, use_streaming=True, max_attempts=2, system_prompt=system_prompt, prompt=prompt, max_tokens=3000)
    elif model in _ANTROPIC_MODELS:
        api_key = os.getenv("ANTROPIC_API_KEY")
        model_split = model.split('_')
        model_name = model_split[0]
        if len(model_split) > 2:
            model_reasoning_effort = model_split[2]
        else:
            model_reasoning_effort = 'disabled'
        return sort_list_with_antropic_api(unsorted_list, api_key, model=model_name, reasoning_effort_param=model_reasoning_effort, system_prompt=system_prompt, prompt=prompt, max_tokens=3000, max_reasoning_tokens=3000)
    elif model in _GOOGLE_GEMINI_MODELS:
        return sort_list_with_google_gemini_api(unsorted_list, model=model, system_prompt=system_prompt, prompt=prompt)
    else:
        raise ValueError(f"Model {model} not supported")
    
def extract_thinking_from_model(sorted_list_str, model):
    model_split = model.split('_')
    sorted_list = None
    thinking_text = None
    thinking_tokens = None
    try:
        if model in _THINKING_MODELS_WITH_OUTPUT:
            thinking_split = sorted_list_str.split('<thinking_block_begin>')
            sorted_list, thinking_text = thinking_split[0], thinking_split[1]
        elif model in _THINKING_MODELS_WITH_TOKEN_OUTPUT:
            thinking_split = sorted_list_str.split('<thinking_token_number>')
            sorted_list, thinking_tokens = thinking_split[0], thinking_split[1]
        else:
            sorted_list = sorted_list_str
    except:
        return sorted_list, None, None
    return sorted_list, thinking_text, thinking_tokens



def sort_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose=False, descending=False):
    """
    Sort all unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    - descending (bool): whether to sort the lists in descending order
    """

    try:
        for unsorted_list_name, unsorted_list in lists.items():
            if descending:
                system_prompt = "Your task is to sort a list in descending order according to the common sorting of the used data type in Python. The output must only contain the sorted list and nothing else. The format of the list must stay the same."
                prompt = f"Sort the following list in descending order: {unsorted_list}"
            else:
                system_prompt = "Your task is to sort a list according to the common sorting of the used data type in Python. The output must only contain the sorted list and nothing else. The format of the list must stay the same."
                prompt = f"Sort the following list: {unsorted_list}"

            if verbose and not descending:
                print(f"Sorting list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")
            elif verbose and descending:
                print(f"Sorting list '{unsorted_list_name}' in descending order using model '{model}' for config '{config_name}'")

            sorted_list = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)

            sorted_list, thinking_string, thinking_tokens = extract_thinking_from_model(sorted_list, model)
            if thinking_string != None:
                cur_results['thinking'][unsorted_list_name] = thinking_string
            if thinking_tokens != None:
                cur_results['thinking_tokens'][unsorted_list_name] = int(thinking_tokens)

            if descending:
                cur_results['sorted_lists_descending'][unsorted_list_name] = sorted_list
            else:
                cur_results['sorted_lists'][unsorted_list_name] = sorted_list

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def reverse_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Reverse all unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            system_prompt = "Your task is to reverse a list according to the common list.reverse() operation in Python. The output must only contain the reversed list and nothing else. The format of the list must stay the same."
            prompt = f"Reverse the following list: {unsorted_list}"

            if verbose:
                print(f"Reversing list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")
            
            sorted_list = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)
            cur_results['reversed_lists'][unsorted_list_name] = sorted_list

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def filter_unsorted_lists_in_config(model, config_name, lists, cur_results, results, filter_type, verbose=False):
    """
    Filter all unsorted lists according to a condition in a configuration using the specified model.
    The condition will be either lower or higher than the median, or median string length if non-numeric, of the list

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - filter_type (str): filter type for now either "lower" or "higher"
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            system_prompt = "Your task is to filter a python list according to a condition. The output must only contain the filtered elements of the list as a list and nothing else. The format of the list must stay the same."
            pivot = None

            if (filter_type == "lower"):
                if (all(type(item) in [int, float] for item in unsorted_list)):
                    min_value = min(unsorted_list)
                    max_value = max(unsorted_list)
                    pivot = random.uniform(min_value, max_value)
                    prompt = f"Filter the following list to only include elements x with x < {pivot}: {unsorted_list}"
                else:
                    pivot = ''.join(random.choices(string.ascii_letters, k=3))
                    prompt = f"Filter the following list to only include elements x with x < '{pivot}': {unsorted_list}"
            elif (filter_type == "higher"):
                if (all(type(item) in [int, float] for item in unsorted_list)):
                    min_value = min(unsorted_list)
                    max_value = max(unsorted_list)
                    pivot = random.uniform(min_value, max_value)
                    prompt = f"Filter the following list to only include elements x with x > {pivot}: {unsorted_list}"
                else:
                    pivot = ''.join(random.choices(string.ascii_letters, k=3))
                    prompt = f"Filter the following list to only include elements x with x > '{pivot}': {unsorted_list}"

            if verbose:
                print(f"Filtering elements in list '{unsorted_list_name}' with filter type '{filter_type}' and pivot '{pivot}' using model '{model}' for config '{config_name}'")

            filtered_list = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)

            filtered_list, thinking_string, thinking_tokens = extract_thinking_from_model(filtered_list, model)
            if thinking_string != None:
                cur_results['thinking'][unsorted_list_name] = thinking_string
            if thinking_tokens != None:
                cur_results['thinking_tokens'][unsorted_list_name] = int(thinking_tokens)
            
            if filter_type == "lower":
                cur_results['filter_lower_lists'][unsorted_list_name] = filtered_list
            elif filter_type == "higher":
                cur_results['filter_higher_lists'][unsorted_list_name] = filtered_list
            cur_results['pivot'][unsorted_list_name] = pivot

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def get_outlier_string(min_str, max_str, make_smaller=True, out_len=7):
    """
    Generate a random 5-character ASCII string (A-Z, a-z), that is lexicographically
    smaller than min_str or higher than max_str depending on make_smaller.

    Args:
        min_str (str): Reference string for make_smaller=True
        max_str (str): Reference string for make_smaller=False
        make_smaller (bool): True -> String < min_str, False -> String > max_str

    Returns:
        str: 5-character string lexicographically smaller/larger than reference

    Raises:
        ValueError: If no valid outlier can be generated for given direction and reference.
    """
    ascii_chars = list(string.ascii_letters)

    candidate = ''.join(random.choices(ascii_chars, k=out_len))

    ref_str = min_str if make_smaller else max_str
    if (make_smaller and candidate < ref_str) or (not make_smaller and candidate > ref_str):
        return candidate
    
    replace_char = None
    replace_char_index = 0

    for ref_char in ref_str:
        replace_char_index = replace_char_index + 1
        if replace_char_index == out_len:
            break
        if make_smaller:
            possible_chars = [c for c in ascii_chars if c < ref_char]
            if len(possible_chars) > 0:
                replace_char = random.choice(possible_chars)
                break
            else:
                candidate = candidate[:replace_char_index-1] + ref_char + candidate[replace_char_index:]
        else:
            possible_chars = [c for c in ascii_chars if c > ref_char]
            if len(possible_chars) > 0:
                replace_char = random.choice(possible_chars)
                break
            else:
                candidate = candidate[:replace_char_index-1] + ref_char + candidate[replace_char_index:]

    if not make_smaller and len(ref_str) < out_len:
        replace_char_index = len(ref_str) + 1
        replace_char = random.choice(ascii_chars)

    if not replace_char:
        return ' '
    elif replace_char_index > 1:
        outlier = candidate[:replace_char_index-1] + replace_char + candidate[replace_char_index:]
    else:
        outlier = replace_char + candidate[1:]
    return outlier

def any_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Return whether any element in the unsorted lists in config satisfies a condition using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            system_prompt = "Your task is to determine whether any element in a python list satisfies a condition. The output must only consist of either 'True' or 'False' and no extra text."

            pivot_index = -1
            pivot_value = None
            condition = ""
            if all(isinstance(x, (int, float)) for x in unsorted_list):
                min_value = min(unsorted_list)
                max_value = max(unsorted_list)
                condition = f"element x < {min_value} or x > {max_value}"
                if random.random() < 0.5:
                    pivot_index = random.randint(0, len(unsorted_list)-1)
                    range_value = max_value - min_value
                    delta = random.uniform(0.1, 0.5) * range_value
                    if random.random() < 0.5:
                        pivot_value = max_value + delta
                        if all(isinstance(x, int) for x in unsorted_list):
                            pivot_value = math.ceil(pivot_value)
                    else:
                        pivot_value = min_value - delta
                        if all(isinstance(x, int) for x in unsorted_list):
                            pivot_value = math.floor(pivot_value)
                    unsorted_list[pivot_index] = pivot_value
            else:
                min_value = min(unsorted_list)
                max_value = max(unsorted_list)
                condition = f"element x < '{min_value}' or x > '{max_value}'"
                if random.random() < 0.5:
                    pivot_index = random.randint(0, len(unsorted_list)-1)
                    pivot_value = "<random_string>"
                    make_smaller = random.choice([True, False])
                    pivot_value = get_outlier_string(min_value, max_value, make_smaller=make_smaller)
                    if pivot_value == "":
                        make_smaller = not make_smaller
                        pivot_value = get_outlier_string(min_value, max_value, make_smaller=make_smaller)
                    unsorted_list[pivot_index] = pivot_value
            
            prompt = "Determine whether any element in the following list satisfies this condition: " + condition + f"\n{unsorted_list}"

            if verbose:
                print(f"Getting any condition in list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")

            any_bool = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)

            any_bool, thinking_string, thinking_tokens = extract_thinking_from_model(any_bool, model)
            if thinking_string != None:
                cur_results['thinking'][unsorted_list_name] = thinking_string
            if thinking_tokens != None:
                cur_results['thinking_tokens'][unsorted_list_name] = int(thinking_tokens)
            
            cur_results['any_values'][unsorted_list_name] = any_bool
            cur_results['pivot_index'][unsorted_list_name] = pivot_index
            cur_results['pivot_value'][unsorted_list_name] = pivot_value

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def all_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Return whether all elements in the unsorted lists in config satisfy a condition using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            system_prompt = "Your task is to determine whether all elements in a python list satisfy a condition. The output must only consist of either 'True' or 'False' and no extra text."

            pivot_index = -1
            pivot_value = None
            condition = ""
            if all(isinstance(x, (int, float)) for x in unsorted_list):
                min_value = min(unsorted_list)
                max_value = max(unsorted_list)
                condition = f"element x >= {min_value} and x <= {max_value}"
                if random.random() < 0.5:
                    pivot_index = random.randint(0, len(unsorted_list)-1)
                    range_value = max_value - min_value
                    delta = random.uniform(0.1, 0.5) * range_value
                    if random.random() < 0.5:
                        pivot_value = max_value + delta
                        if all(isinstance(x, int) for x in unsorted_list):
                            pivot_value = math.ceil(pivot_value)
                    else:
                        pivot_value = min_value - delta
                        if all(isinstance(x, int) for x in unsorted_list):
                            pivot_value = math.floor(pivot_value)
                    unsorted_list[pivot_index] = pivot_value
            else:
                min_value = min(unsorted_list)
                max_value = max(unsorted_list)
                condition = f"element x >= '{min_value}' and x <= '{max_value}'"
                if random.random() < 0.5:
                    pivot_index = random.randint(0, len(unsorted_list)-1)
                    pivot_value = "<random_string>"
                    make_smaller = random.choice([True, False])
                    pivot_value = get_outlier_string(min_value, max_value, make_smaller=make_smaller)
                    if pivot_value == "":
                        make_smaller = not make_smaller
                        pivot_value = get_outlier_string(min_value, max_value, make_smaller=make_smaller)
                    unsorted_list[pivot_index] = pivot_value

            prompt = "Determine whether all elements in the following list satisfy this condition: " + condition + f"\n{unsorted_list}"

            if verbose:
                print(f"Getting all condition in list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")

            all_bool = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)

            all_bool, thinking_string, thinking_tokens = extract_thinking_from_model(all_bool, model)
            if thinking_string != None:
                cur_results['thinking'][unsorted_list_name] = thinking_string
            if thinking_tokens != None:
                cur_results['thinking_tokens'][unsorted_list_name] = int(thinking_tokens)
            
            cur_results['all_values'][unsorted_list_name] = all_bool
            cur_results['pivot_index'][unsorted_list_name] = pivot_index
            cur_results['pivot_value'][unsorted_list_name] = pivot_value

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def uppercase_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Uppercase all string elements in unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():

            system_prompt = "Your task is to convert all strings in the list to uppercase. The output must only contain the list and nothing else. The format of the list must stay the same."
            prompt = f"Uppercase this list: {unsorted_list}"

            if verbose:
                print(f"Uppercasing list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")

            mapped_list = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)
            cur_results['uppercase_lists'][unsorted_list_name] = mapped_list

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def square_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Square all numeric elements in unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():

            if not all(isinstance(x, (int, float)) for x in unsorted_list):
                raise ValueError("List contains non-numeric values, cannot square")
            system_prompt = "Your task is to square all numbers in the list. The output must only contain the list and nothing else. The format of the list must stay the same."
            prompt = f"Square this list: {unsorted_list}"

            if verbose:
                print(f"Squaring list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")

            mapped_list = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)
            cur_results['squared_lists'][unsorted_list_name] = mapped_list

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def count_unsorted_list_items_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Count the number of items in all unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            system_prompt = "Your task is to count the number of items in a list. The output must only contain the count and nothing else."
            prompt = f"Count the number of items in this list: {unsorted_list}."

            if verbose:
                print(f"Counting items in list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")

            count = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)
            cur_results['count_values'][unsorted_list_name] = count

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def get_index_values_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Get the index values of all unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            index = random.randint(0, len(unsorted_list)-1)
            system_prompt = f"Your task is to get the item at a specific index, starting at index 0, from the list. The output must only contain the item and nothing else."
            prompt = f"Get the item at index {index} from this list: {unsorted_list}."

            if verbose:
                print(f"Getting item from list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")
            
            value = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)
            cur_results['index_values'][unsorted_list_name] = value
            cur_results['index_used'][unsorted_list_name] = index

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def get_random_unique_string(unsorted_list, length=5):
    while True:
        s = ''.join(random.choices(string.ascii_lowercase, k=length))
        if s not in unsorted_list:
            return s

def get_random_unique_integer(unsorted_list, min_value=0, max_value=100):
    while True:
        n = random.randint(min_value, max_value)
        if n not in unsorted_list:
            return n
        
def get_random_unique_float(unsorted_list, min_value=0.0, max_value=100.0):
    while True:
        n = random.uniform(min_value, max_value)
        if n not in unsorted_list:
            return n
        
def get_random_unique_word(unsorted_list):
    words = list(set(wordnet.words()))
    words = [word for word in words if "'" not in word]
    while True:
        w = random.choice(words)
        if w not in unsorted_list:
            return w

def insert_values_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Insert a random value at a random index in all unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            index = random.randint(0, len(unsorted_list)) # allow insertion at end of list
            item = None
            if all(isinstance(x, int) for x in unsorted_list):
                item = get_random_unique_integer(unsorted_list)
            elif all(isinstance(x, float) for x in unsorted_list):
                item = get_random_unique_float(unsorted_list)
            elif all(isinstance(x, str) for x in unsorted_list):
                item = get_random_unique_word(unsorted_list)
            else:
                warnings.warn(f"List {unsorted_list_name} contains mixed types, cannot insert item")
                continue
            system_prompt = f"Your task is to insert an item at a specific index, starting at index 0, into a list. The output must only contain the complete list with the inserted element and nothing else."
            prompt = f"Insert the item {item} at index {index} into this list: {unsorted_list}."

            if verbose:
                print(f"Inserting item '{item}' at index '{index}' into list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")
            
            insert_list = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)
            cur_results['insert_lists'][unsorted_list_name] = insert_list
            cur_results['index_used'][unsorted_list_name] = index
            cur_results['item_used'][unsorted_list_name] = item

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def pop_values_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Pop a value at a random index in all unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            index = random.randint(0, len(unsorted_list)-1)
            system_prompt = f"Your task is to remove an item at a specific index, starting at index 0, from a list. The output must only contain the complete list without the removed element and nothing else."
            prompt = f"Remove the item at index {index} from this list: {unsorted_list}."

            if verbose:
                print(f"Popping item at index '{index}' from list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")

            pop_list = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)
            cur_results['pop_lists'][unsorted_list_name] = pop_list
            cur_results['index_used'][unsorted_list_name] = index

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def sum_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Get the sum of all values of all unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():

            system_prompt = f"Your task is to get the sum of all values in a list. The output must only contain the value of the sum and nothing else."
            prompt = f"Get the sum from this list: {unsorted_list}."

            if verbose:
                print(f"Getting sum from list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")

            sum = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)
            cur_results['sum_values'][unsorted_list_name] = sum

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def product_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Get the product of all values of all unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            system_prompt = f"Your task is to get the product of all values in a list. The output must only contain the value of the product and nothing else."
            prompt = f"Get the product from this list: {unsorted_list}."

            if verbose:
                print(f"Getting product from list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")
            
            product = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)
            cur_results['product_values'][unsorted_list_name] = product

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def min_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Get the minimum of all values of all unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            system_prompt = f"Your task is to get the minimum of all values in a list. The output must only contain the value of the minimum and nothing else."
            prompt = f"Get the minimum from this list: {unsorted_list}."

            if verbose:
                print(f"Getting minimum from list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")

            min = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)
            cur_results['min_values'][unsorted_list_name] = min

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def max_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Get the maximum of all values of all unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            system_prompt = f"Your task is to get the maximum of all values in a list. The output must only contain the value of the maximum and nothing else."
            prompt = f"Get the maximum from this list: {unsorted_list}."

            if verbose:
                print(f"Getting maximum from list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")
            
            max = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)
            cur_results['max_values'][unsorted_list_name] = max

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def median_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Get the median of all values of all unsorted lists in a configuration using the specified model.
    If the list contains non-numeric values and is even in length, get the lower median.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            system_prompt = f"Your task is to get the median of all values in a list. The output must only contain the value of the median and nothing else. If the list has an even number of elements and is non numeric, return the lower median."
            prompt = f"Get the median from this list: {unsorted_list}."

            if verbose:
                print(f"Getting median from list '{unsorted_list_name}' using model '{model}' for config '{config_name}'")
            
            median = call_llm_model_api(model, unsorted_list, system_prompt=system_prompt, prompt=prompt)
            cur_results['median_values'][unsorted_list_name] = median

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config '{config_name}' and model '{model}': {e}")
        print(traceback.format_exc())

    return results

def run_single_config_for_model(config_name, lists, model="gpt-4o-mini", verbose=True, results=None, benchmark_type="sort"):
    """
    Run inference on all configs for a single model.

    Parameters:
    - configs (dict): the dictionary of configs
    - api_key (str): the OpenAI API key
    - model (str): the model to use for inference
    - verbose (bool): whether to print verbose output
    - results (dict): the dictionary of results that already exist to avoid re-running inference
    - benchmark_type (str): the benchmark type
    """

    if results is None:
        results = {}
        
    cur_results = {}
    cur_results['model'] = model
    cur_results['benchmark_type'] = benchmark_type
    if is_thinking_model_with_thinking_output(model):
        cur_results['thinking'] = {}
    if is_thinking_model_with_thinking_summary(model):
        cur_results['thinking_tokens'] = {}

    match (benchmark_type):
        case "sort":
            cur_results['sorted_lists'] = {}
            results = sort_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case "sort-descending":
            cur_results['sorted_lists_descending'] = {}
            results = sort_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose, descending=True)
        case "reverse":
            cur_results['reversed_lists'] = {}
            results = reverse_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case "count":
            cur_results['count_values'] = {}
            results = count_unsorted_list_items_in_config(model, config_name, lists, cur_results, results, verbose)
        case "index":
            cur_results['index_values'] = {}
            cur_results['index_used'] = {}
            results = get_index_values_in_config(model, config_name, lists, cur_results, results, verbose)
        case "insert":
            cur_results['insert_lists'] = {}
            cur_results['index_used'] = {}
            cur_results['item_used'] = {}
            results = insert_values_in_config(model, config_name, lists, cur_results, results, verbose)
        case "pop":
            cur_results['pop_lists'] = {}
            cur_results['index_used'] = {}
            results = pop_values_in_config(model, config_name, lists, cur_results, results, verbose)
        case "sum":
            for unsorted_list_name, unsorted_list in lists.items():
                if not all(isinstance(x, (int, float)) for x in unsorted_list):
                    warnings.warn(f"List {unsorted_list_name} contains non-numeric values. Skipping sum benchmark.")
                    return results
            cur_results['sum_values'] = {}
            results = sum_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case "product":
            for unsorted_list_name, unsorted_list in lists.items():
                if not all(isinstance(x, (int, float)) for x in unsorted_list):
                    warnings.warn(f"List {unsorted_list_name} contains non-numeric values. Skipping product benchmark.")
                    return results
            cur_results['product_values'] = {}
            results = product_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case "min":
            cur_results['min_values'] = {}
            results = min_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case "max":
            cur_results['max_values'] = {}
            results = max_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case "median":
            cur_results['median_values'] = {}
            results = median_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case "uppercase":
            for unsorted_list_name, unsorted_list in lists.items():
                if not all(isinstance(x, (str)) for x in unsorted_list):
                    warnings.warn(f"List {unsorted_list_name} contains non-string values. Skipping uppercase benchmark.")
                    return results
            cur_results['uppercase_lists'] = {}
            results = uppercase_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case "square":
            for unsorted_list_name, unsorted_list in lists.items():
                if not all(isinstance(x, (int, float)) for x in unsorted_list):
                    warnings.warn(f"List {unsorted_list_name} contains non-numeric values. Skipping square benchmark.")
                    return results
            cur_results['squared_lists'] = {}
            results = square_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case "filter-lower":
            cur_results['filter_lower_lists'] = {}
            cur_results['pivot'] = {}
            results = filter_unsorted_lists_in_config(model, config_name, lists, cur_results, results, "lower", verbose)
        case "filter-higher":
            cur_results['filter_higher_lists'] = {}
            cur_results['pivot'] = {}
            results = filter_unsorted_lists_in_config(model, config_name, lists, cur_results, results, "higher", verbose)
        case "any":
            cur_results['any_values'] = {}
            cur_results['pivot_index'] = {}
            cur_results['pivot_value'] = {}
            results = any_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case "all":
            cur_results['all_values'] = {}
            cur_results['pivot_index'] = {}
            cur_results['pivot_value'] = {}
            results = all_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case _:
            raise ValueError(f"Benchmark type {benchmark_type} not supported")

    return results


def run_configs_for_single_model(configs, model="gpt-4o-mini", verbose=True, results=None):
    """
    Run inference on all configs for a single model.

    Parameters:
    - configs (dict): the dictionary of configs
    - api_key (str): the OpenAI API key
    - model (str): the model to use for inference
    - verbose (bool): whether to print verbose output
    - results (dict): the dictionary of results that already exist to avoid re-running inference
    """
    if not is_model_supported(model):
        raise ValueError(f"Model {model} not supported")

    if results is None:
        results = {}    
    for config_name, lists in configs.items():
        if check_if_result_available(results, config_name, model):
            if verbose:
                print(f"Results for config {config_name} and model {model} already available. Skipping.")
            continue
        cur_results = {}
        cur_results['model'] = model
        cur_results['sorted_lists'] = {}
        
        try:
            for unsorted_list_name, unsorted_list in lists.items():
                if verbose:
                    print(f"Sorting list {unsorted_list_name} using model {model} for config {config_name}")
                if model in _OPENAI_MODELS:
                    api_key = os.getenv("OPENAI_API_KEY")
                    sorted_list = sort_list_with_openai_api(unsorted_list, api_key, model=model)
                elif model in _INNCUBE_MODELS:
                    api_key = os.getenv("INNCUBE_API_KEY")
                    endpoint_url = "https://llms-inference.innkube.fim.uni-passau.de"
                    sorted_list = sort_list_with_openai_api(unsorted_list, api_key, model=model, url=endpoint_url, use_streaming=True, max_attempts=2)
                elif model in _ANTROPIC_MODELS:
                    api_key = os.getenv("ANTROPIC_API_KEY")
                    sorted_list = sort_list_with_antropic_api(unsorted_list, api_key, model=model)
                else:
                    raise ValueError(f"Model {model} not supported")
                cur_results['sorted_lists'][unsorted_list_name] = sorted_list
        
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
        except Exception as e:
            print(f"Error while running inference for config {config_name} and model {model}: {e}")
            print(traceback.format_exc())

    return results
