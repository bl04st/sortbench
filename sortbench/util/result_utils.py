import gzip
import json
import os

def load_data_local(file_path='benchmark_data', name='sortbench', mode='basic', version='v1.0'):
    """
    Load all data from a local directory into a dict of dicts.

    Parameters:
    file_path (str): path to directory containing data files (default: 'benchmark_data')
    name (str): name of the benchmark data (default: 'sortbench')
    mode (str): mode of the benchmark data (default: 'basic')
    version (str): version of the benchmark data (default: 'v1.0')
    """
    configs = {}

    # fetch all filenames from file_path and filter by name
    filenames = os.listdir(file_path)
    filenames = sorted([filename for filename in filenames if filename.startswith(f'{name}_{mode}_{version}_')])

    # load all data into configs dict
    for filename in filenames:
        with gzip.open(f'{file_path}/{filename}', 'rt', encoding="UTF-8") as f:
            data = json.load(f)
            configs[filename] = data

    return configs

def fetch_configs_from_results(file_path='benchmark_results', name='sortbench', mode='basic', version='v1.0'):
    """
    Load all data from a local directory into a dict of dicts.

    Parameters:
    file_path (str): path to directory containing data files (default: 'benchmark_data')
    name (str): name of the benchmark data (default: 'sortbench')
    mode (str): mode of the benchmark data (default: 'basic')
    version (str): version of the benchmark data (default: 'v1.0')
    """
    # fetch all filenames from file_path and filter by name
    filenames = os.listdir(file_path)
    filenames = sorted([filename for filename in filenames if filename.startswith(f'{name}_{mode}_{version}_')])
    return filenames

def load_results_from_disk(file_path='benchmark_results', name='sortbench', mode='basic', version='v1.0'):
    """
    Load all results from a local directory into a dict of dicts. Will return an empty dict if no results are found.
    Results are stored as gzipped JSON files.

    Parameters:
    file_path (str): path to directory containing results files (default: 'benchmark_results')
    """
    results = {}

    # fetch all filenames from file_path and filter by name
    filenames = os.listdir(file_path)
    filenames = sorted([filename for filename in filenames if filename.startswith(f'{name}_{mode}_{version}_')])

    for filename in filenames:
        with gzip.open(os.path.join(file_path, filename), 'rt', encoding="UTF-8") as f:
            try:
                data = json.load(f)
                results[filename] = data
            except Exception as e:
                print(f"Error while loading results from {filename}: {e}")
    return results

def load_single_result_from_disk(config_name, file_path='benchmark_results'):
    """
    Load all results from a local directory into a dict of dicts. Will return an empty dict if no results are found.
    Results are stored as gzipped JSON files.

    Parameters:
    config_name (str): name of the config
    file_path (str): path to directory containing results files (default: 'benchmark_results')
    """
    
    filename = os.path.join(file_path, config_name)
    
    if not os.path.exists(filename):
        return None
    
    results = {}
    with gzip.open(filename, 'rt', encoding="UTF-8") as f:
        try:
            data = json.load(f)
            results[config_name] = data
        except Exception as e:
            print(f"Error while loading results from {filename}: {e}")
    return results

def check_if_result_available_on_disk(results_path, config_name, model_name, benchmark_type):
    """
    Check if results for a specific config and model are already available.

    Parameters:
    results (dict): dict containing all results
    config_name (str): name of the config
    model_name (str): name of the model
    """
    #create results file path
    results = load_single_result_from_disk(config_name, results_path)
    if results is None:
        return False
    else:
        for result in results[config_name]['results']:
            if result['model'] == model_name and result['benchmark_type'] == benchmark_type:
                return True
    return False



def check_if_result_available(results, config_name, model_name):
    """
    Check if results for a specific config and model are already available.

    Parameters:
    results (dict): dict containing all results
    config_name (str): name of the config
    model_name (str): name of the model
    """
    if config_name in results:
        model_names = [result['model'] for result in results[config_name]['results']]
        if model_name in model_names:
            return True
    return False

def write_results_to_disk(results, file_path='benchmark_results', overwrite=False):
    """
    Write results to disk. Results are stored as gzipped JSON files.

    Parameters:
    results (dict): dict containing all results
    file_path (str): path to directory to write results files to (default: 'benchmark_results')
    overwrite (bool): whether to overwrite existing files or whether to append new results (default: False)
    """
    for config_name, config_results in results.items():
        file = os.path.join(file_path, config_name)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if not overwrite and os.path.exists(file):
            # read existing results and update
            with gzip.open(file, 'rt', encoding="UTF-8") as f:
                try:
                    existing_results = json.load(f)
                except Exception as e:
                    print(f"Error while loading results from {file}: {e}")
                    print("Overwriting file.")
                    existing_results = {}
                existing_results['results'] = existing_results['results'] + config_results['results']
                dict_to_write = existing_results
        else:
            dict_to_write = config_results

        with gzip.open(file, 'wt', encoding="UTF-8") as f:
            json.dump(dict_to_write, f)
