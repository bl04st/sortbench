import os
import time
import traceback
import random
import warnings

from util.result_utils import check_if_result_available

from openai import OpenAI, InternalServerError
import anthropic

_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "o3-mini"]
_INNCUBE_MODELS = ["llama3.1", "gemma2", "qwen2.5", "deepseekr1"]
_ANTROPIC_MODELS = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"]
_GOOGLE_GEMINI_MODELS = ["gemini-2.5-flash"]

_BENCHMARK_TYPES = ["sort", "sort-descending", "reverse", "count", "index", "map", "sum", "min", "product"]

def is_model_supported(model):
    """
    Check if a model is supported by sortbench.
    
    Parameters:
    - model (str): the model name
    """
    return model in _OPENAI_MODELS+_INNCUBE_MODELS+_ANTROPIC_MODELS+_GOOGLE_GEMINI_MODELS

def is_benchmark_type_supported(bench_type):
    """
    Check if a benchmark type is supported by sortbench.

    Parameters:
    - bench_type (str): the benchmark type
    """
    return bench_type in _BENCHMARK_TYPES

def sort_list_with_google_gemini_api(unsorted_list, model, system_prompt=None, prompt=None):
    """
    Calls the Google Gemini API to sort a list.

    Parameters:
    - unsorted_list (list): the list to be sorted
    - api_key (str): the Google Gemini API key
    - model (str): the model to use for inference
    """
    from google import genai

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

def sort_list_with_antropic_api(unsorted_list, api_key, model, system_prompt=None, prompt=None):
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
    
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=1,
        system=system_prompt,
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
    sorted_list = message.content[0].text
    return sorted_list


def sort_list_with_openai_api(unsorted_list, api_key, model, url=None, use_streaming=False, system_prompt=None, prompt=None, max_attempts=1):
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
    
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        if url is None:
            client = OpenAI(api_key=api_key)
        else:
            client = OpenAI(api_key=api_key, base_url=url)
        try:
            if model=='o1-mini':
                # the reasoning models from OpenAI do not have a system prompt
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n{prompt}"}
                    ],
                    stream=use_streaming
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    stream=use_streaming
                )
            if use_streaming:
                # uncomment collected chunks for debugging
                # collected_chunks = []
                collected_messages = []
                for chunk in response:
                    # collected_chunks.append(chunk)
                    chunk_message = chunk.choices[0].delta.content
                    collected_messages.append(chunk_message)
                #finish_reason = response.choices[0].finish_reason
                #if finish_reason != 'stop':
                #    raise RuntimeError(f"Stream did not finish properly: {finish_reason}")
                sorted_list = ''.join([m for m in collected_messages if m is not None])
            else:
                sorted_list = response.choices[0].message.content.strip()
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
        finally:
            client.close()
    
    return sorted_list

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
                system_prompt = None
                prompt = None

            if verbose and not descending:
                print(f"Sorting list {unsorted_list_name} using model {model} for config {config_name}")
            elif verbose and descending:
                print(f"Sorting list {unsorted_list_name} in descending order using model {model} for config {config_name}")
            if model in _OPENAI_MODELS:
                api_key = os.getenv("OPENAI_API_KEY")
                sorted_list = sort_list_with_openai_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _INNCUBE_MODELS:
                api_key = os.getenv("INNCUBE_API_KEY")
                endpoint_url = "https://llms-inference.innkube.fim.uni-passau.de"
                sorted_list = sort_list_with_openai_api(unsorted_list, api_key, model=model, url=endpoint_url, use_streaming=True, max_attempts=2, system_prompt=system_prompt, prompt=prompt)
            elif model in _ANTROPIC_MODELS:
                api_key = os.getenv("ANTROPIC_API_KEY")
                sorted_list = sort_list_with_antropic_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _GOOGLE_GEMINI_MODELS:
                sorted_list = sort_list_with_google_gemini_api(unsorted_list, model=model, system_prompt=system_prompt, prompt=prompt)
            else:
                raise ValueError(f"Model {model} not supported")
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
        print(f"Error while running inference for config {config_name} and model {model}: {e}")
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
                print(f"Reversing list {unsorted_list_name} using model {model} for config {config_name}")
            if model in _OPENAI_MODELS:
                api_key = os.getenv("OPENAI_API_KEY")
                sorted_list = sort_list_with_openai_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _INNCUBE_MODELS:
                api_key = os.getenv("INNCUBE_API_KEY")
                endpoint_url = "https://llms-inference.innkube.fim.uni-passau.de"
                sorted_list = sort_list_with_openai_api(unsorted_list, api_key, model=model, url=endpoint_url, use_streaming=True, max_attempts=2, system_prompt=system_prompt, prompt=prompt)
            elif model in _ANTROPIC_MODELS:
                api_key = os.getenv("ANTROPIC_API_KEY")
                sorted_list = sort_list_with_antropic_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _GOOGLE_GEMINI_MODELS:
                sorted_list = sort_list_with_google_gemini_api(unsorted_list, model=model, system_prompt=system_prompt, prompt=prompt)
            else:
                raise ValueError(f"Model {model} not supported")
            cur_results['reversed_lists'][unsorted_list_name] = sorted_list

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config {config_name} and model {model}: {e}")
        print(traceback.format_exc())

    return results

def map_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose=False):
    """
    Apply a mapping function to all unsorted lists in a configuration using the specified model.

    Parameters:
    - model (str): the model to use for inference
    - config_name (str): the name of the configuration
    - lists (dict): the dictionary of lists
    - cur_results (dict): the current results dictionary
    - results (dict): the overall results dictionary
    - verbose (bool): whether to print verbose output
    """
    map_func = None
    try:
        for unsorted_list_name, unsorted_list in lists.items():
            if all(isinstance(x, (int, float)) for x in unsorted_list):
                map_func = "square each number"
            elif all(isinstance(x, str) for x in unsorted_list):
                map_func = "convert each string to uppercase"
            system_prompt = "Your task is to apply a function to all elements in a list. The output must only contain the mapped list and nothing else. The format of the list must stay the same."
            prompt = f"Apply this function to all list elements: {map_func}.\nThe list is: {unsorted_list}"
            if verbose:
                print(f"Mapping list {unsorted_list_name} using mapping function \"{map_func}\" using model {model} for config {config_name}")
            if model in _OPENAI_MODELS:
                api_key = os.getenv("OPENAI_API_KEY")
                sorted_list = sort_list_with_openai_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _INNCUBE_MODELS:
                api_key = os.getenv("INNCUBE_API_KEY")
                endpoint_url = "https://llms-inference.innkube.fim.uni-passau.de"
                sorted_list = sort_list_with_openai_api(unsorted_list, api_key, model=model, url=endpoint_url, use_streaming=True, max_attempts=2, system_prompt=system_prompt, prompt=prompt)
            elif model in _ANTROPIC_MODELS:
                api_key = os.getenv("ANTROPIC_API_KEY")
                sorted_list = sort_list_with_antropic_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _GOOGLE_GEMINI_MODELS:
                sorted_list = sort_list_with_google_gemini_api(unsorted_list, model=model, system_prompt=system_prompt, prompt=prompt)
            else:
                raise ValueError(f"Model {model} not supported")
            cur_results['mapped_lists'][unsorted_list_name] = sorted_list

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config {config_name} and model {model}: {e}")
        print(traceback.format_exc())
    cur_results['mapping_function'] = map_func

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
                print(f"Counting items in list {unsorted_list_name} using model {model} for config {config_name}")
            if model in _OPENAI_MODELS:
                api_key = os.getenv("OPENAI_API_KEY")
                count = sort_list_with_openai_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _INNCUBE_MODELS:
                api_key = os.getenv("INNCUBE_API_KEY")
                endpoint_url = "https://llms-inference.innkube.fim.uni-passau.de"
                count = sort_list_with_openai_api(unsorted_list, api_key, model=model, url=endpoint_url, use_streaming=True, max_attempts=2, system_prompt=system_prompt, prompt=prompt)
            elif model in _ANTROPIC_MODELS:
                api_key = os.getenv("ANTROPIC_API_KEY")
                count = sort_list_with_antropic_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _GOOGLE_GEMINI_MODELS:
                count = sort_list_with_google_gemini_api(unsorted_list, model=model, system_prompt=system_prompt, prompt=prompt)
            else:
                raise ValueError(f"Model {model} not supported")
            cur_results['list_counts'][unsorted_list_name] = count

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config {config_name} and model {model}: {e}")
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
                print(f"Getting item from list {unsorted_list_name} using model {model} for config {config_name}")
            if model in _OPENAI_MODELS:
                api_key = os.getenv("OPENAI_API_KEY")
                value = sort_list_with_openai_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _INNCUBE_MODELS:
                api_key = os.getenv("INNCUBE_API_KEY")
                endpoint_url = "https://llms-inference.innkube.fim.uni-passau.de"
                value = sort_list_with_openai_api(unsorted_list, api_key, model=model, url=endpoint_url, use_streaming=True, max_attempts=2, system_prompt=system_prompt, prompt=prompt)
            elif model in _ANTROPIC_MODELS:
                api_key = os.getenv("ANTROPIC_API_KEY")
                value = sort_list_with_antropic_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _GOOGLE_GEMINI_MODELS:
                value = sort_list_with_google_gemini_api(unsorted_list, model=model, system_prompt=system_prompt, prompt=prompt)
            else:
                raise ValueError(f"Model {model} not supported")
            cur_results['index_values'][unsorted_list_name] = value
            cur_results['index_used'][unsorted_list_name] = index

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config {config_name} and model {model}: {e}")
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
                print(f"Getting sum from list {unsorted_list_name} using model {model} for config {config_name}")
            if model in _OPENAI_MODELS:
                api_key = os.getenv("OPENAI_API_KEY")
                value = sort_list_with_openai_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _INNCUBE_MODELS:
                api_key = os.getenv("INNCUBE_API_KEY")
                endpoint_url = "https://llms-inference.innkube.fim.uni-passau.de"
                value = sort_list_with_openai_api(unsorted_list, api_key, model=model, url=endpoint_url, use_streaming=True, max_attempts=2, system_prompt=system_prompt, prompt=prompt)
            elif model in _ANTROPIC_MODELS:
                api_key = os.getenv("ANTROPIC_API_KEY")
                value = sort_list_with_antropic_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _GOOGLE_GEMINI_MODELS:
                value = sort_list_with_google_gemini_api(unsorted_list, model=model, system_prompt=system_prompt, prompt=prompt)
            else:
                raise ValueError(f"Model {model} not supported")
            cur_results['sum_values'][unsorted_list_name] = value

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config {config_name} and model {model}: {e}")
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
                print(f"Getting product from list {unsorted_list_name} using model {model} for config {config_name}")
            if model in _OPENAI_MODELS:
                api_key = os.getenv("OPENAI_API_KEY")
                value = sort_list_with_openai_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _INNCUBE_MODELS:
                api_key = os.getenv("INNCUBE_API_KEY")
                endpoint_url = "https://llms-inference.innkube.fim.uni-passau.de"
                value = sort_list_with_openai_api(unsorted_list, api_key, model=model, url=endpoint_url, use_streaming=True, max_attempts=2, system_prompt=system_prompt, prompt=prompt)
            elif model in _ANTROPIC_MODELS:
                api_key = os.getenv("ANTROPIC_API_KEY")
                value = sort_list_with_antropic_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _GOOGLE_GEMINI_MODELS:
                value = sort_list_with_google_gemini_api(unsorted_list, model=model, system_prompt=system_prompt, prompt=prompt)
            else:
                raise ValueError(f"Model {model} not supported")
            cur_results['product_values'][unsorted_list_name] = value

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config {config_name} and model {model}: {e}")
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
                print(f"Getting minimum from list {unsorted_list_name} using model {model} for config {config_name}")
            if model in _OPENAI_MODELS:
                api_key = os.getenv("OPENAI_API_KEY")
                value = sort_list_with_openai_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _INNCUBE_MODELS:
                api_key = os.getenv("INNCUBE_API_KEY")
                endpoint_url = "https://llms-inference.innkube.fim.uni-passau.de"
                value = sort_list_with_openai_api(unsorted_list, api_key, model=model, url=endpoint_url, use_streaming=True, max_attempts=2, system_prompt=system_prompt, prompt=prompt)
            elif model in _ANTROPIC_MODELS:
                api_key = os.getenv("ANTROPIC_API_KEY")
                value = sort_list_with_antropic_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _GOOGLE_GEMINI_MODELS:
                value = sort_list_with_google_gemini_api(unsorted_list, model=model, system_prompt=system_prompt, prompt=prompt)
            else:
                raise ValueError(f"Model {model} not supported")
            cur_results['min_values'][unsorted_list_name] = value

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config {config_name} and model {model}: {e}")
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
                print(f"Getting maximum from list {unsorted_list_name} using model {model} for config {config_name}")
            if model in _OPENAI_MODELS:
                api_key = os.getenv("OPENAI_API_KEY")
                value = sort_list_with_openai_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _INNCUBE_MODELS:
                api_key = os.getenv("INNCUBE_API_KEY")
                endpoint_url = "https://llms-inference.innkube.fim.uni-passau.de"
                value = sort_list_with_openai_api(unsorted_list, api_key, model=model, url=endpoint_url, use_streaming=True, max_attempts=2, system_prompt=system_prompt, prompt=prompt)
            elif model in _ANTROPIC_MODELS:
                api_key = os.getenv("ANTROPIC_API_KEY")
                value = sort_list_with_antropic_api(unsorted_list, api_key, model=model, system_prompt=system_prompt, prompt=prompt)
            elif model in _GOOGLE_GEMINI_MODELS:
                value = sort_list_with_google_gemini_api(unsorted_list, model=model, system_prompt=system_prompt, prompt=prompt)
            else:
                raise ValueError(f"Model {model} not supported")
            cur_results['max_values'][unsorted_list_name] = value

        if config_name in results:
            results[config_name]['results'].append(cur_results)
        else:
            results[config_name] = {'unsorted_lists': lists,
                                    'results': [cur_results]}
    except Exception as e:
        print(f"Error while running inference for config {config_name} and model {model}: {e}")
        print(traceback.format_exc())

    return results


def run_single_config_for_model(config_name, lists, model="gpt-4o-mini", verbose=True, results=None, bench_type="sort"):
    """
    Run inference on all configs for a single model.

    Parameters:
    - configs (dict): the dictionary of configs
    - api_key (str): the OpenAI API key
    - model (str): the model to use for inference
    - verbose (bool): whether to print verbose output
    - results (dict): the dictionary of results that already exist to avoid re-running inference
    - bench_type (str): the benchmark type
    """

    if results is None:
        results = {}
        
    cur_results = {}
    cur_results['model'] = model
    cur_results['bench_type'] = bench_type
    # cur_results['sorted_lists'] = {}

    match (bench_type):
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
            cur_results['list_counts'] = {}
            results = count_unsorted_list_items_in_config(model, config_name, lists, cur_results, results, verbose)
        case "index":
            cur_results['index_values'] = {}
            cur_results['index_used'] = {}
            results = get_index_values_in_config(model, config_name, lists, cur_results, results, verbose)
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
            for unsorted_list_name, unsorted_list in lists.items():
                if not all(isinstance(x, (int, float)) for x in unsorted_list):
                    warnings.warn(f"List {unsorted_list_name} contains non-numeric values. Skipping min benchmark.")
                    return results
            cur_results['min_values'] = {}
            results = min_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case "max":
            for unsorted_list_name, unsorted_list in lists.items():
                if not all(isinstance(x, (int, float)) for x in unsorted_list):
                    warnings.warn(f"List {unsorted_list_name} contains non-numeric values. Skipping max benchmark.")
                    return results
            cur_results['max_values'] = {}
            results = max_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case "map":
            cur_results['mapped_lists'] = {}
            results = map_unsorted_lists_in_config(model, config_name, lists, cur_results, results, verbose)
        case _:
            raise ValueError(f"Benchmark type {bench_type} not supported")

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
