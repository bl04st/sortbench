import inflect
import gzip
import json
import os
import random
import string

from nltk.corpus import wordnet

def generate_unsorted_list(n=10, lst_type='integer', **kwargs):
    """
    Generate a list of n random values of the given type. The following types are supported:
    - integer: random integers between min_value and max_value
    - float: random floats between min_value and max_value
    - string: random strings of 5 ASCII characters (both lower and upper case)
    - string_lower: random strings of 5 lower case ASCII characters
    - string_upper: random strings of 5 upper case ASCII characters
    - word: random words from the NLTK wordnet corpus. Words with apostrophes are excluded.
    - number_string: random number strings between min_value and max_value
    - prefix_string: random strings of 8 characters with a prefix of 3 characters that are equal
    - prefix_word: random words from the NLTK wordnet corpus with a prefix of 3 characters that are equal. Words with apostrophes are excluded.
    - hexadecimal_string: random hexadecimal strings between min_value and max_value (not implemented yet)

    The random seed should be controlled by the caller if reproducibility is desired. Use random.seed() from the random module.

    Parameters:
    - n: the number of values to generate (default: 10)
    - lst_type: the type of the values to generate. Possible values are: 'integer', 'float', 'string', 'word', 'number_string', 'prefix_string', 'prefix_words' (default: 'integer')
    - min_value: the minimum value for the generated values (optional, only for numeric types, default: 0)
    - max_value: the maximum value for the generated values (optional, only for numeric types, default: 100)
    - duplicates: if set to true, each item appears twice in the list. Only possible with even n. (optional, default: False)
    - sorted: if set to true, the list is sorted in ascending order (optional, default: False)

    Returns:
    - a list of n values of the given type
    """
    if n<=0:
        raise ValueError("n must be a positive integer")
    
    supported_types = ['integer', 'float', 'string', 'string_lower', 'string_upper', 'word', 'number_string', 'prefix_string', 'prefix_word', 'hexadecimal_string']
    if lst_type not in supported_types:
        raise ValueError(f"Type must be in {supported_types}, got: {lst_type}")
    min_value = kwargs.get('min_value', 0)
    max_value = kwargs.get('max_value', 100)
    if not (isinstance(min_value, int) or isinstance(min_value, float)):
        raise ValueError("min_value must be an integer or float")
    if not (isinstance(max_value, int) or isinstance(max_value, float)):
        raise ValueError(f"max_value must be an integer or float")
    if min_value >= max_value:
        raise ValueError("min_value must be less than max_value")
    duplicates = kwargs.get('duplicates', False)
    if duplicates:
        if n % 2 != 0:
            raise ValueError("n must be an even number for duplicates")
    sorted = kwargs.get('sorted', False)

    result = []
    if lst_type == 'integer':
        result = random.sample(range(min_value, max_value), n)
    elif lst_type == 'float':
        flt_list = []
        while len(flt_list) < n:
            new_vals = [random.uniform(min_value, max_value) for _ in range(n-len(flt_list))]
            flt_list.extend(new_vals)
            flt_list = list(set(flt_list))
        random.shuffle(flt_list)
        result = flt_list
    elif lst_type == 'string' or lst_type == 'string_lower' or lst_type == 'string_upper':
        return generate_string_list(n, lst_type)
    elif lst_type == 'word':
        words = list(set(wordnet.words()))
        words = [word for word in words if "'" not in word]
        result = random.sample(words, n)
    elif lst_type == 'number_string':
        p = inflect.engine()
        numbers = random.sample(range(min_value, max_value), n)
        result = [p.number_to_words(number) for number in numbers]
    elif lst_type == 'prefix_string':
        prefix_letter = random.choice(string.ascii_letters)
        str_list = generate_string_list(n, 'string')
        result = [f"{prefix_letter*3}{string}" for string in str_list]
    elif lst_type == 'prefix_word':
        prefix_letter = random.choice(string.ascii_letters)
        words = list(set(wordnet.words()))
        words = [word for word in words if "'" not in word]
        word_smpl = random.sample(words, n)
        result = [f"{prefix_letter*3}{word}" for word in word_smpl]
    elif lst_type == 'hexadecimal_string':
        numbers = random.sample(range(min_value, max_value), n)
        result = [hex(number) for number in numbers]
    else:
        raise ValueError("Unknown type")
    
    if duplicates:
        # use only first half of list and duplicate it
        result = [item for sublist in zip(result[:int(n/2)], result[:int(n/2)]) for item in sublist]

    if sorted:
        result.sort()
        
    return result
    
def generate_string_list(n, lst_type):
    """
    Generate a list of n random strings of 5 ASCII characters.

    Parameters:
    - n: the number of strings to generate
    - lst_type: the type of the strings to generate. Possible values are: 'string', 'string_lower', 'string_upper'

    Returns:
    - a list of n strings of 5 ASCII characters
    """
    if lst_type == 'string':
        letters = string.ascii_letters
    elif lst_type == 'string_lower':
        letters = string.ascii_lowercase
    elif lst_type == 'string_upper':
        letters = string.ascii_uppercase

    str_list = []
    while len(str_list) < n:
        new_vals = [''.join(random.choices(letters, k=5)) for _ in range(n-len(str_list))]
        str_list.extend(new_vals)
        str_list = list(set(str_list))
    random.shuffle(str_list)
    return str_list

def generate_json_file(file, num_lists, generator):
    """
    Generate a JSON file with num_lists lists of random data generated by the generator function.
    Files are stored as gzipped JSON files, to save space.

    Parameters:
    - file: the name of the file to write
    - num_lists: the number of lists to generate
    - generator: a function that generates a list of random data
    """
    data = {}
    for i in range(num_lists):
        data[f'list_{i+1}'] = generator()
    with gzip.open(file, 'wt', encoding="UTF-8") as f:
        f.write(json.dumps(data))

def generate_benchmark_data(path, name, version, num_lists, sizes, types, type_names=None, gen_kwargs=None):
    """
    Generate benchmark data for a given name, sizes, and types.
    The data is written to the folder 'benchmark_data', with one JSON file per size and type.

    Parameters:
    - path: the path to the folder where the data files will be written
    - name: the name of the benchmark data. The name must include a single underscore.
    - version: the version of the benchmark data. The version must not include underscores.
    - num_lists: the number of lists to generate
    - sizes: a list of sizes for the lists
    - types: a list of types for the lists
    - type_names: a list of names for the types. Uses types if None. Names must not include underscores. (optional, default: None)
    - gen_kwargs: a list of dictionaries with keyword arguments for the generator function. (optional, default: None)
    """
    max_digits = len(str(max(sizes)))
    if type_names is None:
        type_names = types
    if len(types) != len(type_names):
        raise ValueError("Types and type names must have the same length")
    if len(name.split('_')) != 2:
        raise ValueError("Name must include a single underscore (between name of benchmark and mode)")
    if '_' in version:
        raise ValueError("Version must not include underscores")
    if any('_' in name for name in type_names):
        raise ValueError("Type names must not include underscores")
    if gen_kwargs is None:
        gen_kwargs = [{} for _ in range(len(types))]
    elif len(gen_kwargs) != len(types):
        raise ValueError("gen_kwargs must have the same length as types")

    for size in sizes:
        for type, type_name, cur_gen_kwargs in zip(types, type_names, gen_kwargs):
            # use pathlib to create the file path
            size_str = str(size).zfill(max_digits)
            file = os.path.join(path, f'{name}_{version}_{type_name}_{size_str}.json.gz')
            os.makedirs(os.path.dirname(file), exist_ok=True)
            generate_json_file(file, num_lists, lambda: generate_unsorted_list(size, type, **cur_gen_kwargs))

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