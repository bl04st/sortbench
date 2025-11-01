import os
import re
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from collections import Counter

import util.inference_utils as inf_utils

def compute_insert_score(unsorted_list, insert_list, index_used, item_used):
    """
    Compute the insert score between two lists of items. Insert score just checks if the item was inserted at the correct index.
    It does not consider the order of other items or the correctness of all other items in the list.

    The Method expects that item_used is not in unsorted_list.

    Parameters:
    - unsorted_list (list): First list of items.
    - insert_list (list): Second list of items. List should be the unsorted list with one item inserted.

    Returns:
    - float: Insert score between the two lists.
    """
    if index_used is None or item_used is None:
        return None
    if index_used < 0 or index_used > len(unsorted_list):
        return None
    
    try:
        # if the item is at the correct position and the list is 1 item more the score is 1.0 (perfect insertion, not necessarily correct rest of list)
        if (item_used == insert_list[index_used]) and (len(unsorted_list) == len(insert_list) - 1):
            return 1.0
        # if the item is in the list, but at the wrong index the score is 0.5
        elif (item_used in insert_list):
            return 0.5
        # if one item is inserted, but it's the wrong item the score is 0.25
        elif (len(unsorted_list) == len(insert_list) - 1) and (item_used not in insert_list):
            return 0.25
        # if the item is not in the list, the score is 0.0
        return 0.0
    except IndexError:
        # assume index error means wrongful insertion
        return 0.0

def compute_pop_score(unsorted_list, pop_list, index_used):
    """
    Compute the pop score between two lists of items. Pop score just checks if the item was popped at the correct index.
    It does not consider the order of other items or the correctness of all other items in the list.

    The Method expects that item_used is not in unsorted_list.

    Parameters:
    - unsorted_list (list): First list of items.
    - insert_list (list): Second list of items. List should be the unsorted list with one item inserted.

    Returns:
    - float: Insert score between the two lists.
    """
    if index_used is None:
        return None
    if index_used < 0 or index_used > len(unsorted_list) - 1:
        return None
    
    try:
        deleted_item = unsorted_list[index_used]
        # if the item is not in the pop list and the lengths match the score is 1.0 (perfect deletion, not necessarily correct rest of list)
        if (deleted_item not in pop_list) and (len(unsorted_list) == len(pop_list) + 1):
            return 1.0
        # if the item is in the pop list, but the lengths match (meaning 1 item was popped, just not the correct one) the score is 0.5
        if (len(unsorted_list) == len(pop_list) + 1) and (deleted_item in pop_list):
            return 0.5
        # if the correct item was popped, but the lengths don't match (meaning more than 1 item was removed) the score is 0.25
        if (len(unsorted_list) != len(pop_list) + 1) and (deleted_item not in pop_list):
            return 0.25
        return 0.0
    except IndexError:
        # assume index error means wrongful insertion
        return 0.0

def compute_numeric_similarity_reversed(unsorted_list, reversed_list):
    """
    Compute the numeric similarity between two lists of items.

    Parameters:
    - unsorted_list (list): First list of items.
    - reversed_list (list): Second list of items. List should be the reversed version of the first list.

    Returns:
    - float: Numeric similarity between the two lists.
    """
    unsorted_list_reverse = list(reversed(unsorted_list))
    try:
        original = np.array(unsorted_list_reverse, dtype=float)
        predicted = np.array(reversed_list[:len(original)], dtype=float)
        max_val = max(original.max(), predicted.max(), 1e-9)
        original = original / max_val
        predicted = predicted / max_val
        diff = np.abs(original - predicted).mean()
        return round(1 - diff, 6)
    except Exception:
        # Fallback: fuzzy string similarity
        n = min(len(unsorted_list_reverse), len(reversed_list))
        scores = [
            SequenceMatcher(None, str(unsorted_list_reverse[i]), str(reversed_list[i])).ratio()
            for i in range(n)
        ]
        return round(sum(scores) / len(scores), 6) if scores else 0.0
    

def count_non_matching_items_reversed(unsorted_list, reversed_list):
    """
    Count the number of items that are not in the correct position in the reversed list compared to the unsorted list.

    Parameters:
    - unsorted_list (list): A list of items.
    - reversed_list (list): A reversed version of the unsorted list

    Returns:
    - int: The number of items that are not in the correct position.
    """
    count = 0
    unsorted_list_reverse = list(reversed(unsorted_list))
    min_length = min(len(unsorted_list_reverse), len(reversed_list))
    for i in range(min_length):
        if unsorted_list_reverse[i] != reversed_list[i]:
            count += 1
    # count remaining items in the longer list as incorrect
    count += abs(len(unsorted_list_reverse) - len(reversed_list))
    return count

def count_unordered_pairs(lst):
    """
    Count the number of unordered pairs in a list.
    
    Parameters:
    - lst (list): A list of integers.

    Returns:
    - int: The number of unordered pairs in the list.
    """
    count = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                count += 1
    return count

def count_unordered_pairs_descending(lst):
    """
    Count the number of unordered pairs in a list.
    
    Parameters:
    - lst (list): A list of integers.

    Returns:
    - int: The number of unordered pairs in the list.
    """
    count = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] < lst[j]:
                count += 1
    return count

def count_unordered_neighbors(lst):
    """
    Count the number of unordered neighbors in a list.

    Parameters:
    - lst (list): A list of integers.

    Returns:
    - int: The number of unordered neighbors in the list.
    """
    count = 0
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            count += 1
    return count

def count_unordered_neighbors_descending(lst):
    """
    Count the number of unordered neighbors in a list.

    Parameters:
    - lst (list): A list of integers.

    Returns:
    - int: The number of unordered neighbors in the list.
    """
    count = 0
    for i in range(len(lst) - 1):
        if lst[i] < lst[i + 1]:
            count += 1
    return count

def count_missing_items(unsorted_list, sorted_list):
    """
    Count the number of missing items in the sorted list compared to the unsorted list.

    Parameters:
    - unsorted_list (list): A list of items.
    - sorted_list (list): A sorted version of the unsorted list

    Returns:
    - int: The number of missing items in the sorted list compared to the unsorted list.
    """
    unsorted_counter = Counter(unsorted_list)
    sorted_counter = Counter(sorted_list)
    missing_items = unsorted_counter - sorted_counter
    return sum(missing_items.values())

def count_missing_items_pop(unsorted_list, pop_list):
    """
    Count the number of missing items in the pop list compared to the unsorted list.

    Parameters:
    - unsorted_list (list): A list of items.
    - pop_list (list): A pop version of the unsorted list

    Returns:
    - int: The number of missing items in the pop list compared to the unsorted list.
    """
    unsorted_counter = Counter(unsorted_list)
    pop_counter = Counter(pop_list)
    missing_items = unsorted_counter - pop_counter
    missing_count = sum(missing_items.values())
    if missing_count > 0:
        missing_count = missing_count - 1  # subtract the removed item
    return missing_count

def count_additional_items(unsorted_list, sorted_list):
    """
    Count the number of additional items in the sorted list compared to the unsorted list.

    Parameters:
    - unsorted_list (list): A list of items.
    - sorted_list (list): A sorted version of the unsorted list

    Returns:
    - int: The number of additional items in the sorted list compared to the unsorted list.
    """
    unsorted_counter = Counter(unsorted_list)
    sorted_counter = Counter(sorted_list)
    additional_items = sorted_counter - unsorted_counter
    return sum(additional_items.values())

def count_additional_items_insert(unsorted_list, insert_list):
    """
    Count the number of additional items in the insert list compared to the unsorted list.

    Parameters:
    - unsorted_list (list): A list of items.
    - insert_list (list): The insert list to compare against the unsorted list.

    Returns:
    - int: The number of additional items in the insert list compared to the unsorted list.
    """
    unsorted_counter = Counter(unsorted_list)
    insert_counter = Counter(insert_list)
    additional_items = insert_counter - unsorted_counter
    additional_count = sum(additional_items.values())
    if additional_count > 0:
        additional_count = additional_count - 1  # subtract the inserted item
    return additional_count

#########################################
# Functions for handling parsing errors #
#########################################

# corner cases for errors with brakets in output

def missing_closing_broken_quotes(str_list):
    """
    Try to fix a string that is missing a closing bracket and has broken quotes.
    """
    last_comma_index = str_list.rfind(',')
    if last_comma_index==-1:
        return None
    str_list = str_list[:last_comma_index] + ']'
    # drop all colons
    str_list = str_list.replace("'", '')
    str_list = str_list.replace(", ", "', '")
    str_list = str_list.replace("[", "['")
    str_list = str_list.replace("]", "']")
    sorted_list = None
    try:
        sorted_list = eval(str_list)
    except:
        sorted_list = None
    return sorted_list

def missing_closing_bracket(str_list):
    """
    Try to fix a string that is missing a closing bracket.
    """
    last_comma_index = str_list.rfind(',')
    if last_comma_index==-1:
        return None
    str_list = str_list[:last_comma_index] + ']'
    sorted_list = None
    try:
        sorted_list = eval(str_list)
    except:
        sorted_list = None
    return sorted_list

def strings_without_quotes(str_list):
    """
    Try to fix a string that has items that are strings, but with the quotes missing.
    """
    str_list = str_list.replace(", ", "', '")
    str_list = str_list.replace("[", "['")
    str_list = str_list.replace("]", "']")
    sorted_list = None
    try:
        sorted_list = eval(str_list)
    except:
        pass
    return sorted_list

def drop_newlines(str_list):
    """
    Try to fix a string by removing all newlines.
    """
    str_list = str_list.replace('\n', ' ')
    sorted_list = None
    try:
        sorted_list = eval(str_list)
    except:
        pass
    return sorted_list

def drop_and_fix_quotes(str_list):
    """
    Try to fix a string by dropping all quotes and adding them back in the right places.
    """
    # first drop all quotes
    str_list = str_list.replace("'", "")
    # then add expected quotes
    str_list = str_list.replace(", ", "', '")
    str_list = str_list.replace("[", "['")
    str_list = str_list.replace("]", "']")
    sorted_list = None
    try:
        sorted_list = eval(str_list)
    except:
        pass
    return sorted_list

def drop_quotes_and_newlines(str_list):
    """
    Try to fix a string by dropping all quotes and newlines and adding them back in the right places.
    """
    str_list = str_list.replace('\n', ' ')
    # first drop all quotes
    str_list = str_list.replace("'", "")
    # then add expected quotes
    str_list = str_list.replace(", ", "', '")
    str_list = str_list.replace("[", "['")
    str_list = str_list.replace("]", "']")
    sorted_list = None
    try:
        sorted_list = eval(str_list)
        # sanity check for length of list items
        if len(sorted_list)==1:
            sorted_list = None
    except:
        pass
    return sorted_list

def drop_after_last_closing_bracket(str_list):
    """
    Try to fix a string by dropping all content after the last closing bracket.
    """
    cropped_sorted_list = str_list[:(str_list.rfind(']')+1)]
    cropped_sorted_list = cropped_sorted_list[cropped_sorted_list.rfind('['):]
    sorted_list = None
    try:
        sorted_list = eval(cropped_sorted_list)
    except:
        pass 
        # cropped_sorted_list = str_list[:str_list.rfind(',')] + ']'
    return sorted_list

def drop_after_first_closing_bracket(str_list):
    """
    Try to fix a string by dropping all content after the first closing bracket.
    """
    str_list = str_list[:str_list.find(']')+1]
    sorted_list = None
    try:
        sorted_list = eval(str_list)
        if len(sorted_list)<=1:
            sorted_list = None
        for item in sorted_list:
            if type(item)==str and len(item)==0:
                sorted_list = None
                break
    except:
        pass
    return sorted_list

def latex_matcher(str_list):
    """
    Try to parse a list that is given in latex format.
    """
    if not str_list.endswith(r'\]'):
        return None
    str_list = str_list.split('\n')[1] # assume first row starts latex, second row ends it
    # drop all instances of \boxed{ and }
    if str_list.startswith(r'\boxed{'):
        str_list = str_list.replace(r'\boxed{', '')
        str_list = str_list.replace('}', '')
    if not str_list.startswith(r'['):
        str_list = '[' + str_list
    if not str_list.endswith(r']'):
        str_list = str_list + ']'
    sorted_list = None
    try:
        sorted_list = eval(str_list)
        if len(sorted_list)<=1:
            sorted_list = None
    except:
        sorted_list = None
    return sorted_list

def first_lines_latex_matcher(str_list):
    """
    Try to parse a list that is given in latex format which has additional content after the list
    """
    lines = str_list.split('\n')
    if len(lines)<=2:
        return None
    if lines[0].startswith(r'[') and lines[2].endswith(r'\]'):
        str_list = lines[1]
        # drop all instances of \boxed{ and }
        if str_list.startswith(r'\boxed{'):
            str_list = str_list.replace(r'\boxed{', '')
            str_list = str_list.replace('}', '')
        if not str_list.startswith(r'['):
            str_list = '[' + str_list
        if not str_list.endswith(r']'):
            str_list = str_list + ']'
        sorted_list = None
        try:
            sorted_list = eval(str_list)
        except:
            sorted_list = None
        return sorted_list
    return None

def last_lines_latex_matcher(str_list):
    """
    Try to parse a list that is given in latex format which has additional content before the list
    """
    lines = str_list.split('\n')
    if len(lines)<=2:
        return None
    if lines[-3].startswith(r'\[') and lines[-1].endswith(r'\]'):
        str_list = lines[-2]
        # drop all instances of \boxed{ and }
        if str_list.startswith(r'\boxed{'):
            str_list = str_list.replace(r'\boxed{', '')
            str_list = str_list.replace('}', '')
        if not str_list.startswith(r'['):
            str_list = '[' + str_list
        if not str_list.endswith(r']'):
            str_list = str_list + ']'
        sorted_list = None
        try:
            sorted_list = eval(str_list)
        except:
            sorted_list = None
        return sorted_list
    return None

def last_line_latex_matcher(str_list):
    """
    Try to parse a list that is given in latex format which has additional content before the list
    """
    lines = str_list.split('\n')
    if len(lines)<=2:
        return None
    if lines[-1].startswith(r'\[') and lines[-1].endswith(r'\]'):
        str_list = lines[-1]
        # drop all instances of \boxed{ and }
        str_list = str_list.replace(r'\[', '[')
        str_list = str_list.replace(r'\]', ']')
        sorted_list = None
        try:
            sorted_list = eval(str_list)
        except:
            sorted_list = None
        return sorted_list
    return None

def last_line_list(str_list):
    """
    Multi-line output that contains the list in the last line
    """
    lines = str_list.split('\n')
    if len(lines)==0:
        return None
    last_line = lines[-1]
    sorted_list = None
    if last_line.startswith('[') and last_line.endswith(']'):
        try:
            sorted_list = eval(last_line)
        except:
            pass
    return sorted_list

def last_line_list_no_close(str_list):
    """
    Multi-line output that contains the list in the last line without closing
    """
    lines = str_list.split('\n')
    if len(lines)==0:
        return None
    last_line = lines[-1]
    sorted_list = None
    if last_line.startswith('['):
        last_line = last_line[:last_line.rfind(',')] + ']'
        try:
            sorted_list = eval(last_line)
        except:
            pass
    return sorted_list

def last_items_incomplete(str_list):
    sorted_list = None
    if str_list.startswith('[') and str_list.endswith(']'):
        # drop all ... from string
        str_list = str_list.replace('...', '')
        try:
            sorted_list = eval(str_list)
        except:
            pass
    return sorted_list

# corner cases for errors without brakets in output

def numbered_list_matcher(str_list):
    """
    Try to parse a list that is given as a numbered list.
    """
    lines = str_list.split('\n')
    if len(lines)==0:
        return None
    # look for start in last three lines
    lst = []
    attempts = 0
    while True:
        has_enum = re.match(r'\d+\. ', lines[-1].strip())
        if has_enum:
            lst.insert(0, lines[-1].split(' ')[1].strip())
        lines = lines[:-1]
        attempts += 1
        if len(lines)==0 or (not has_enum and len(lst)>0) or (not has_enum and attempts>3):
            break
    if len(lst)>0:
        return lst
    return None

def backtick_matcher(str_list):
    """
    Try to parse a list that is given in a code block marked by three backticks.
    """
    backtick_matches = re.findall(r"```plaintext(.*)```", str_list, re.DOTALL)
    if len(backtick_matches)>0:
        matched_content = backtick_matches[-1].strip()
        sorted_list = [s.strip() for s in matched_content.split('\n')]
        return sorted_list
    backtick_matches = re.findall(r"```(.*)```", str_list, re.DOTALL)
    if len(backtick_matches)>0:
        matched_content = backtick_matches[-1].strip()
        sorted_list = [s.strip() for s in matched_content.split('\n')]
        return sorted_list
    return None
    
def curly_braces_matcher(str_list):
    """
    Try to parse a list in curly braces that may have additional latex code.
    """
    # drop potential boxed statements
    str_list = str_list.replace(r'\boxed{', '')
    str_list = str_list.replace('}', '')
    # the assumption is that this is now latex code and underscores are escaped
    curly_braces_matches = re.findall(r"\{(.*)\}", str_list, re.DOTALL)
    if len(curly_braces_matches)>0:
        matched_content = curly_braces_matches[-1].strip()
        matched_content = matched_content.replace('\\_', '_')
        sorted_list = [s.strip() for s in matched_content.split(', ')]
        return sorted_list
    return None

def linewise_matcher(str_list):
    """
    Try to parse a list that is given line-wise with a starting string.
    """
    lines = str_list.strip().split('\n')
    if len(lines)==0:
        return None
    # check if first line indicates sorted list
    if not lines[0].startswith('The sorted list'):
        return None
    
    # assuming all non-empty lines after first line are list items
    lst = []
    for line in lines[1:]:
        stripped_line = line.strip()
        if stripped_line.endswith(','):
            stripped_line = stripped_line[:-1]
        if len(stripped_line)>0:
            lst.append(stripped_line)
    if len(lst)>1:
        return lst
    return None

def last_line_is_list_matcher(str_list):
    """
    Comma-separated list in last line
    """
    lines = str_list.split('\n')
    if len(lines)==0:
        return None
    last_line = lines[-1]
    sorted_list = None
    lst = []
    if last_line.count(',')>4:
        csv_items = last_line.split(',')
        for item in csv_items:
            item = item.replace(r'\boxed{', '')
            item = item.replace('}', '')
            lst.append(item.strip())
        if len(lst)>0:
            sorted_list = lst
    return sorted_list
    
def eval_str_list(str_list, expected_type, debug=True, config_name='config', model_name='model', list_name='lst'):
    """
    Tries to parse a list given as a string. If the string is not valid python, we try to things:
    1. We check if the string ends in a closing bracket. If not, we look for the last comma, crop the string there and add a new closing bracket.
    2. We remove all single quotes that are not part of a string (i.e. not followed by a space or a comma) and try to evaluate the string again.

    Parameters:
    - str_list (str): A string representation of a list.

    Returns:
    - sorted_list (list): The evaluated list. None if not possible.
    - is_cropped (bool): True if the string was cropped.
    - is_cleaned (bool): True if the string was cleaned.
    """
    error_type = None
    sorted_list = None
    
    # strip deepseek reasoning
    if str_list.startswith('<think>'):
        str_list_org = str_list # keep original for debugging
        str_list = str_list[str_list.find('</think>')+8:]
        # TODO: check if this can be dropped. seems like there is some corner case handling here. 
        contains_brackets = str_list.count('[')>0
        if contains_brackets:
            str_list = str_list[str_list.find('['):].strip()
        # remove ** from string
        str_list = str_list.replace('**', '')

    # strip "input()" from all strings as this can trip up eval
    str_list = str_list.replace('input()', '')
    
    try:
        sorted_list = eval(str_list)
    except:
        # find last closing bracket and crop after and then first open bracket before
        last_closing_bracket_index = str_list.rfind(']')
        last_opening_bracket_index = str_list.rfind('[')
        if last_closing_bracket_index==-1 and last_opening_bracket_index==-1:
            # no brackets at all
            if sorted_list is None:
                sorted_list = numbered_list_matcher(str_list)
                if sorted_list is not None:
                    error_type = 'Numbered, line-wise list'
            if sorted_list is None:
                sorted_list = backtick_matcher(str_list)
                if sorted_list is not None:
                    error_type = 'List in backticks'
            if sorted_list is None:
                sorted_list = curly_braces_matcher(str_list)
                if sorted_list is not None:
                    error_type = 'List in latex'
            if sorted_list is None:
                sorted_list = linewise_matcher(str_list)
                if sorted_list is not None:
                    error_type = 'Line-wise list with content berfore list'
            if sorted_list is None:
                sorted_list = last_line_is_list_matcher(str_list)
                if sorted_list is not None:
                    error_type = 'Multi-line output with comma-separated list in last line'
        else:
            # has some brackets
            if sorted_list is None:
                sorted_list = missing_closing_bracket(str_list)
                if sorted_list is not None:
                    error_type = 'Missing closing bracket'
            if sorted_list is None:
                sorted_list = last_items_incomplete(str_list)
                if sorted_list is not None:
                    error_type = 'Incomplete list items at the end'
            if sorted_list is None:
                sorted_list = missing_closing_broken_quotes(str_list)
                if sorted_list is not None:
                    error_type = 'Missing closing bracket and broken quotes'
            if sorted_list is None:
                sorted_list = strings_without_quotes(str_list)
                if sorted_list is not None:
                    error_type = 'String list without quotes'
            if sorted_list is None:
                sorted_list = drop_newlines(str_list)
                if sorted_list is not None:
                    error_type = 'Invalid newlines in list'
            if sorted_list is None:
                sorted_list = drop_and_fix_quotes(str_list)
                if sorted_list is not None:
                    error_type = 'Broken quotes'
            if sorted_list is None:
                sorted_list = drop_quotes_and_newlines(str_list)
                if sorted_list is not None:
                    error_type = 'Broken quotes and invalid newlines'
            if sorted_list is None:
                sorted_list = drop_after_last_closing_bracket(str_list)
                if sorted_list is not None:
                    error_type = 'Content after last closing bracket'
            if sorted_list is None:
                sorted_list = drop_after_first_closing_bracket(str_list)
                if sorted_list is not None:
                    error_type = 'Multiple closing brackets, valid list before first closing bracket'
            if sorted_list is None:
                sorted_list = latex_matcher(str_list)
                if sorted_list is not None:
                    error_type = 'List as latex'
            if sorted_list is None:
                sorted_list = first_lines_latex_matcher(str_list)
                if sorted_list is not None:
                    error_type = 'List as latex with additional content after list'
            if sorted_list is None:
                sorted_list = last_lines_latex_matcher(str_list)
                if sorted_list is not None:
                    error_type = 'List as latex with additional content before list'
            if sorted_list is None:
                sorted_list = last_line_latex_matcher(str_list)
                if sorted_list is not None:
                    error_type = 'List as latex with additional content before list'
            if sorted_list is None:
                sorted_list = last_line_list(str_list)
                if sorted_list is not None:
                    error_type = 'Content before list'
            if sorted_list is None:
                sorted_list = last_line_list_no_close(str_list)
                if sorted_list is not None:
                    error_type = 'Content before list without closing bracket'
            
    if sorted_list is None and debug:
        file_name = f'not_parsed_{config_name}_{model_name}_{list_name}.txt'
        if not os.path.exists(f'known_parsing_errors/{file_name}'):
            with open(f'debug/{file_name}', 'w') as f:
                f.write(str_list)

    is_list = False
    has_ellipsis = False
    required_type_parsing = False
    if type(sorted_list)==list:
        is_list = True
    else:
        if type(sorted_list)==tuple:
            if str_list.startswith('(\''):
                sorted_list = list(sorted_list)
            elif str_list.startswith('([') or str_list.endswith('],'):
                sorted_list = sorted_list[0]
            is_list = False
    if type(sorted_list)!=list:
        sorted_list = None
    if sorted_list is not None and ... in sorted_list:
        # drop all ellipses from list
        sorted_list = [s for s in sorted_list if s!=...]
        has_ellipsis = True
    if sorted_list is not None:
        for i in range(len(sorted_list)):
            if type(sorted_list[i])!=expected_type:
                try:
                    required_type_parsing = True
                    sorted_list[i] = expected_type(sorted_list[i])
                except:
                    print(f'Could not convert "{sorted_list[i]}" to {expected_type} - setting list to None')
                    print('error_code:', error_type)
                    sorted_list = None
                    is_list = False
                    has_ellipsis = False
                    error_type = None
                    file_name = f'not_parsed_{config_name}_{model_name}_{list_name}.txt'
                    if not os.path.exists(f'known_parsing_errors/{file_name}'):
                        with open(f'debug/{file_name}', 'w') as f:
                            f.write(str_list)
                    break
            
    return (sorted_list, error_type, is_list, has_ellipsis, required_type_parsing)

def get_result_dict(benchmark_name, bench_type, benchmark_mode, benchmark_version, model, data_type, list_length,
list_name, unordered_pairs_before=None, unordered_pairs_after=None, unordered_neighbors_before=None, unordered_neighbors_after=None,
count_missing=None, count_additional=None, numeric_similarity=None, incorrect_items=None, out_list_len=None,
num_chars=None, thinking_length=None, is_parsed=None, error_type=None, is_list=None, has_ellipsis=None, required_type_parsing=None, benchmark_score=None):

    return {
        'Benchmark': benchmark_name,
        'Benchmark Type': bench_type,
        'Mode': benchmark_mode,
        'Version': benchmark_version,
        'Model': model,
        'Type': data_type,
        'Size': list_length,
        'List Name': list_name,
        'Unordered Pairs Before': unordered_pairs_before,
        'Unordered Pairs After': unordered_pairs_after,
        'Unordered Neighbors Before': unordered_neighbors_before,
        'Unordered Neighbors After': unordered_neighbors_after,
        'Missing Items': count_missing,
        'Additional Items': count_additional,
        'Numeric Similarity': numeric_similarity,
        'Incorrect Items': incorrect_items,
        'Output List Length': out_list_len,
        'Output Length': num_chars,
        'Thinking Length': thinking_length,
        'Parsed': is_parsed,
        'HasError': error_type is not None,
        'ErrorType': error_type,    
        'IsList': is_list,
        'HasEllipsis': has_ellipsis,
        'RequiredTypeParsing': required_type_parsing,
        'Unordered Pairs (%)': None,
        'Unordered Neighbors (%)': None,
        'Missing Items (%)': None,
        'Additional Items (%)': None,
        'Incorrect Items (%)': None,
        'Validity Score': None,
        'Faithfulness Score': None,
        'Benchmark Score': benchmark_score,
        'Overall Score': None
    }


def eval_sort_benchmark(results, config_name, cur_result, unsorted_lists, benchmark_name, bench_type, model, benchmark_mode, benchmark_version, data_type, list_length):

    results_with_eval = []

    for list_name, sorted_list in cur_result['sorted_lists'].items():
        unsorted_list = unsorted_lists[list_name]
        expected_type = type(unsorted_list[0])
        num_chars = len(sorted_list)
        thinking_length = 0
        thinking_pos = sorted_list.rfind('</think>')
        if thinking_pos!=-1:
            thinking_length = deepseek_numtokens(sorted_list[:thinking_pos+8])
        sorted_list, error_type, is_list, has_ellipsis, required_type_parsing = eval_str_list(sorted_list, expected_type, debug=True, config_name=config_name, model_name=model, list_name=list_name)
        if sorted_list is None:
            unordered_pairs_before = None
            unordered_pairs_after = None
            unordered_neighbors_before = None
            unordered_neighbors_after = None
            count_missing = None
            count_additional = None
            out_list_len = None
            is_parsed = False
        else:
            unordered_pairs_before = count_unordered_pairs(unsorted_list)
            unordered_pairs_after = count_unordered_pairs(sorted_list)
            unordered_neighbors_before = count_unordered_neighbors(unsorted_list)
            unordered_neighbors_after = count_unordered_neighbors(sorted_list)
            count_missing = count_missing_items(unsorted_list, sorted_list)
            count_additional = count_additional_items(unsorted_list, sorted_list)
            out_list_len = len(sorted_list)
            is_parsed = True

        result_dict = get_result_dict(
            benchmark_name, bench_type, benchmark_mode, benchmark_version, model, data_type, list_length, list_name,
            unordered_pairs_before=unordered_pairs_before, unordered_pairs_after=unordered_pairs_after, unordered_neighbors_before=unordered_neighbors_before,
            unordered_neighbors_after=unordered_neighbors_after, count_missing=count_missing, count_additional=count_additional, out_list_len=out_list_len, num_chars=num_chars,
            thinking_length=thinking_length, is_parsed=is_parsed, error_type=error_type, is_list=is_list, has_ellipsis=has_ellipsis, required_type_parsing=required_type_parsing
        )

        results_with_eval.append(result_dict)
    return results_with_eval

def eval_sort_descending_benchmark(results, config_name, cur_result, unsorted_lists, benchmark_name, bench_type, model, benchmark_mode, benchmark_version, data_type, list_length):

    results_with_eval = []

    for list_name, sorted_list in cur_result['sorted_lists_descending'].items():
        unsorted_list = unsorted_lists[list_name]
        expected_type = type(unsorted_list[0])
        num_chars = len(sorted_list)
        thinking_length = 0
        thinking_pos = sorted_list.rfind('</think>')
        if thinking_pos!=-1:
            thinking_length = deepseek_numtokens(sorted_list[:thinking_pos+8])
        sorted_list, error_type, is_list, has_ellipsis, required_type_parsing = eval_str_list(sorted_list, expected_type, debug=True, config_name=config_name, model_name=model, list_name=list_name)
        if sorted_list is None:
            unordered_pairs_before = None
            unordered_pairs_after = None
            unordered_neighbors_before = None
            unordered_neighbors_after = None
            count_missing = None
            count_additional = None
            out_list_len = None
            is_parsed = False
        else:
            unordered_pairs_before = count_unordered_pairs_descending(unsorted_list)
            unordered_pairs_after = count_unordered_pairs_descending(sorted_list)
            unordered_neighbors_before = count_unordered_neighbors_descending(unsorted_list)
            unordered_neighbors_after = count_unordered_neighbors_descending(sorted_list)
            count_missing = count_missing_items(unsorted_list, sorted_list)
            count_additional = count_additional_items(unsorted_list, sorted_list)
            out_list_len = len(sorted_list)
            is_parsed = True

        result_dict = get_result_dict(
            benchmark_name, bench_type, benchmark_mode, benchmark_version, model, data_type, list_length, list_name,
            unordered_pairs_before=unordered_pairs_before, unordered_pairs_after=unordered_pairs_after, unordered_neighbors_before=unordered_neighbors_before,
            unordered_neighbors_after=unordered_neighbors_after, count_missing=count_missing, count_additional=count_additional, out_list_len=out_list_len, num_chars=num_chars,
            thinking_length=thinking_length, is_parsed=is_parsed, error_type=error_type, is_list=is_list, has_ellipsis=has_ellipsis, required_type_parsing=required_type_parsing
        )

        results_with_eval.append(result_dict)
    return results_with_eval

def eval_reverse_benchmark(results, config_name, cur_result, unsorted_lists, benchmark_name, bench_type, model, benchmark_mode, benchmark_version, data_type, list_length):

    results_with_eval = []

    for list_name, reversed_list in cur_result['reversed_lists'].items():
        unsorted_list = unsorted_lists[list_name]
        expected_type = type(unsorted_list[0])
        num_chars = len(reversed_list)
        thinking_length = 0
        thinking_pos = reversed_list.rfind('</think>')
        if thinking_pos!=-1:
            thinking_length = deepseek_numtokens(reversed_list[:thinking_pos+8])
        reversed_list, error_type, is_list, has_ellipsis, required_type_parsing = eval_str_list(reversed_list, expected_type, debug=True, config_name=config_name, model_name=model, list_name=list_name)
        if reversed_list is None:
            incorrect_items = None
            numeric_similarity = None
            count_missing = None
            count_additional = None
            out_list_len = None
            is_parsed = False
        else:
            incorrect_items = count_non_matching_items_reversed(unsorted_list, reversed_list)
            numeric_similarity = compute_numeric_similarity_reversed(unsorted_list, reversed_list)
            count_missing = count_missing_items(unsorted_list, reversed_list)
            count_additional = count_additional_items(unsorted_list, reversed_list)
            out_list_len = len(reversed_list)
            is_parsed = True

        result_dict = get_result_dict(
            benchmark_name, bench_type, benchmark_mode, benchmark_version, model, data_type, list_length, list_name,
            incorrect_items=incorrect_items, numeric_similarity=numeric_similarity, count_missing=count_missing,
            count_additional=count_additional, out_list_len=out_list_len, num_chars=num_chars,
            thinking_length=thinking_length, is_parsed=is_parsed, error_type=error_type, is_list=is_list,
            has_ellipsis=has_ellipsis, required_type_parsing=required_type_parsing
        )

        results_with_eval.append(result_dict)
    return results_with_eval

def eval_insert_benchmark(results, config_name, cur_result, unsorted_lists, benchmark_name, bench_type, model, benchmark_mode, benchmark_version, data_type, list_length):

    results_with_eval = []

    for list_name, insert_list in cur_result['insert_lists'].items():
        index_used = cur_result['index_used'].get(list_name, None)
        item_used = cur_result['item_used'].get(list_name, None)
        unsorted_list = unsorted_lists[list_name]
        expected_type = type(unsorted_list[0])
        num_chars = len(insert_list)
        thinking_length = 0
        thinking_pos = insert_list.rfind('</think>')
        if thinking_pos!=-1:
            thinking_length = deepseek_numtokens(insert_list[:thinking_pos+8])
        insert_list, error_type, is_list, has_ellipsis, required_type_parsing = eval_str_list(insert_list, expected_type, debug=True, config_name=config_name, model_name=model, list_name=list_name)
        if insert_list is None:
            count_missing = None
            count_additional = None
            out_list_len = None
            is_parsed = False
        else:
            benchmark_score = compute_insert_score(unsorted_list, insert_list, index_used, item_used)
            count_missing = count_missing_items(unsorted_list, insert_list)
            count_additional = count_additional_items_insert(unsorted_list, insert_list)
            out_list_len = len(insert_list)
            is_parsed = True

        result_dict = get_result_dict(
            benchmark_name, bench_type, benchmark_mode, benchmark_version, model, data_type, list_length, list_name, count_missing=count_missing,
            count_additional=count_additional, out_list_len=out_list_len, num_chars=num_chars,
            thinking_length=thinking_length, is_parsed=is_parsed, error_type=error_type, is_list=is_list,
            has_ellipsis=has_ellipsis, required_type_parsing=required_type_parsing, benchmark_score=benchmark_score
        )

        results_with_eval.append(result_dict)
    return results_with_eval

def eval_pop_benchmark(results, config_name, cur_result, unsorted_lists, benchmark_name, bench_type, model, benchmark_mode, benchmark_version, data_type, list_length):

    results_with_eval = []

    for list_name, pop_list in cur_result['pop_lists'].items():
        index_used = cur_result['index_used'].get(list_name, None)
        unsorted_list = unsorted_lists[list_name]
        expected_type = type(unsorted_list[0])
        num_chars = len(pop_list)
        thinking_length = 0
        thinking_pos = pop_list.rfind('</think>')
        if thinking_pos!=-1:
            thinking_length = deepseek_numtokens(pop_list[:thinking_pos+8])
        pop_list, error_type, is_list, has_ellipsis, required_type_parsing = eval_str_list(pop_list, expected_type, debug=True, config_name=config_name, model_name=model, list_name=list_name)
        if pop_list is None:
            pop_score = None
            count_missing = None
            count_additional = None
            out_list_len = None
            is_parsed = False
        else:
            pop_score = compute_pop_score(unsorted_list, pop_list, index_used)
            count_missing = count_missing_items_pop(unsorted_list, pop_list)
            count_additional = count_additional_items(unsorted_list, pop_list)
            out_list_len = len(pop_list)
            is_parsed = True

        result_dict = get_result_dict(
            benchmark_name, bench_type, benchmark_mode, benchmark_version, model, data_type, list_length, list_name,
            benchmark_score=pop_score, count_missing=count_missing,
            count_additional=count_additional, out_list_len=out_list_len, num_chars=num_chars,
            thinking_length=thinking_length, is_parsed=is_parsed, error_type=error_type, is_list=is_list,
            has_ellipsis=has_ellipsis, required_type_parsing=required_type_parsing
        )

        results_with_eval.append(result_dict)
    return results_with_eval

def eval_count_output(count_str, config_name, model_name, list_name, debug=False):

    count = None
    error_type = None
    count_str = str(count_str).strip()
    try:
        count = int(float(count_str))
        error_type = None
    except ValueError:
        match = re.search(r"-?\d+", count_str)
        if match:
            count = int(match.group(0))
            if match.start() > 0:
                error_type = "Extra characters before number"
            elif match.end() < len(count_str):
                error_type = "Extra characters after number"
            else:
                error_type = None
        else:
            error_type = "No numeric value found"
        
    if debug and count is None:
        file_name = f'not_parsed_{config_name}_{model_name}_{list_name}.txt'
        if not os.path.exists(f'known_parsing_errors/{file_name}'):
            with open(f'debug/{file_name}', 'w') as f:
                f.write(count_str)

    return count, error_type

def eval_count_benchmark(results, config_name, cur_result, unsorted_lists, benchmark_name, bench_type, model, benchmark_mode, benchmark_version, data_type, list_length):

    results_with_eval = []

    for list_name, count in cur_result['list_counts'].items():
        unsorted_list = unsorted_lists[list_name]
        num_chars = len(unsorted_list)
        thinking_length = 0
        thinking_pos = count.rfind('</think>')
        if thinking_pos!=-1:
            thinking_length = deepseek_numtokens(count[:thinking_pos+8])

        count, error_type = eval_count_output(count, debug=True, config_name=config_name, model_name=model, list_name=list_name)

        # count output is not a list therefore set these to default values
        is_list = False
        has_ellipsis = False
        required_type_parsing = False

        if count is None:
            count_score = None
            count_missing = None
            count_additional = None
            out_list_len = None
            is_parsed = False
        else:
            count_score = 1.0 if count==len(unsorted_list) else max(0.0, 1 - float(abs(count - len(unsorted_list)) / len(unsorted_list)))
            count_missing = None
            count_additional = None
            out_list_len = None
            is_parsed = True

        result_dict = get_result_dict(
            benchmark_name, bench_type, benchmark_mode, benchmark_version, model, data_type, list_length, list_name,
            benchmark_score=count_score, count_missing=count_missing,
            count_additional=count_additional, out_list_len=out_list_len, num_chars=num_chars,
            thinking_length=thinking_length, is_parsed=is_parsed, error_type=error_type, is_list=is_list,
            has_ellipsis=has_ellipsis, required_type_parsing=required_type_parsing
        )

        results_with_eval.append(result_dict)
    return results_with_eval
    
def evaluate_results(results):
    """
    Evaluate the results of the sorting benchmarks.

    Parameters:
    - results (dict): The results of the sorting benchmarks.

    Returns:
    - df_results (pd.DataFrame): A DataFrame with the evaluated results.
    """
    results_with_eval = []
    for config_name, config_data in results.items():
        benchmark_name = config_name.split('_')[0]
        benchmark_mode = config_name.split('_')[1]
        benchmark_version = config_name.split('_')[2]
        data_type = config_name.split('_')[3]
        list_length = int(config_name.split('_')[4].split('.')[0])
        
        unsorted_lists = config_data['unsorted_lists']
        
        for cur_result in config_data['results']:
            model = cur_result['model']
            bench_type = cur_result['bench_type']
            match (bench_type):
                case "sort":
                    temp_results = eval_sort_benchmark(results, config_name, cur_result, unsorted_lists, benchmark_name, bench_type, model, benchmark_mode, benchmark_version, data_type, list_length)
                    for res in temp_results:
                        results_with_eval.append(res)
                case "sort-descending":
                    temp_results = eval_sort_descending_benchmark(results, config_name, cur_result, unsorted_lists, benchmark_name, bench_type, model, benchmark_mode, benchmark_version, data_type, list_length)
                    for res in temp_results:
                        results_with_eval.append(res)
                case "reverse":
                    temp_results = eval_reverse_benchmark(results, config_name, cur_result, unsorted_lists, benchmark_name, bench_type, model, benchmark_mode, benchmark_version, data_type, list_length)
                    for res in temp_results:
                        results_with_eval.append(res)
                case "insert":
                    temp_results = eval_insert_benchmark(results, config_name, cur_result, unsorted_lists, benchmark_name, bench_type, model, benchmark_mode, benchmark_version, data_type, list_length)
                    for res in temp_results:
                        results_with_eval.append(res)
                case "pop":
                    temp_results = eval_pop_benchmark(results, config_name, cur_result, unsorted_lists, benchmark_name, bench_type, model, benchmark_mode, benchmark_version, data_type, list_length)
                    for res in temp_results:
                        results_with_eval.append(res)
                case "count":
                    temp_results = eval_count_benchmark(results, config_name, cur_result, unsorted_lists, benchmark_name, bench_type, model, benchmark_mode, benchmark_version, data_type, list_length)
                    for res in temp_results:
                        results_with_eval.append(res)
                case _:
                    print(f'Unknown benchmark type: {bench_type}')

    df_results = pd.DataFrame(results_with_eval)
    df_results = normalize_metrics(df_results)
    df_results = compute_total_score(df_results)
    return df_results

def score_single_results(sorted_list, unsorted_list, benchmark_name, benchmark_mode, benchmark_version, config_name, model, data_type, list_length, list_name):
    expected_type = type(unsorted_list[0])
    num_chars = len(sorted_list)
    thinking_length = 0
    thinking_pos = sorted_list.rfind('</think>')
    if thinking_pos!=-1:
        thinking_length = deepseek_numtokens(sorted_list[:thinking_pos+8])
    sorted_list, error_type, is_list, has_ellipsis, required_type_parsing = eval_str_list(sorted_list, expected_type, debug=True, config_name=config_name, model_name=model, list_name=list_name)
    if sorted_list is None:
        unordered_pairs_before = None
        unordered_pairs_after = None
        unordered_neighbors_before = None
        unordered_neighbors_after = None
        count_missing = None
        count_additional = None
        len_diff = None
        is_parsed = False
    else:
        unordered_pairs_before = count_unordered_pairs(unsorted_list)
        unordered_pairs_after = count_unordered_pairs(sorted_list)
        unordered_neighbors_before = count_unordered_neighbors(unsorted_list)
        unordered_neighbors_after = count_unordered_neighbors(sorted_list)
        count_missing = count_missing_items(unsorted_list, sorted_list)
        count_additional = count_additional_items(unsorted_list, sorted_list)
        len_diff = len(unsorted_list)-len(sorted_list)
        is_parsed = True

    result_dict = {
        'Benchmark': benchmark_name,
        'Mode': benchmark_mode,
        'Version': benchmark_version,
        'Model': model,
        'Type': data_type,
        'Size': list_length,
        'List Name': list_name,
        'Unordered Pairs Before': unordered_pairs_before,
        'Unordered Pairs After': unordered_pairs_after,
        'Unordered Neighbors Before': unordered_neighbors_before,
        'Unordered Neighbors After': unordered_neighbors_after,
        'Missing Items': count_missing,
        'Additional Items': count_additional,
        'Length Difference': len_diff,
        'Output Length': num_chars,
        'Thinking Length': thinking_length,
        'Parsed': is_parsed,
        'HasError': error_type is not None,
        'ErrorType': error_type,
        'IsList': is_list,
        'HasEllipsis': has_ellipsis,
        'RequiredTypeParsing': required_type_parsing
    }
    return result_dict

def normalize_metrics(df_results):
    """
    Normalize the metrics to be percentages of the size of the list. In case of pairs, we normalize by the number of pairs in the list.

    Parameters:
    - df_results: DataFrame with the results of the benchmark

    Returns:
    - df_results: DataFrame with the normalized metrics
    """
    print(df_results.columns.tolist())

    df_results['Unordered Pairs (%)'] = df_results['Unordered Pairs After']/(df_results['Output List Length']*(df_results['Output List Length']-1)/2)
    df_results['Unordered Neighbors (%)'] = df_results['Unordered Neighbors After']/df_results['Output List Length'] # TODO: size is only for the original list, sorted list might have different length. this is buggy
    df_results['Incorrect Items (%)'] = (df_results['Incorrect Items']/df_results['Size']).clip(upper=1)
    df_results['Missing Items (%)'] = (df_results['Missing Items']/df_results['Size']).clip(upper=1)
    df_results['Additional Items (%)'] = (df_results['Additional Items']/df_results['Size']).clip(upper=1)
    return df_results

def calc_score(row):
    if row['Benchmark Type'] == 'sort' or row['Benchmark Type'] == 'sort-descending':
        return 1 - (row['Unordered Pairs (%)'] + row['Unordered Neighbors (%)'])/2
    elif row['Benchmark Type'] == 'reverse':
        return 1 - (row['Incorrect Items (%)'] + (1 - row['Numeric Similarity']))/2
    elif row['Benchmark Type'] == 'insert':
        return row['Benchmark Score'] # use precomputed insert score
    elif row['Benchmark Type'] == 'pop':
        return row['Benchmark Score'] # use precomputed pop score
    elif row['Benchmark Type'] == 'count':
        return row['Benchmark Score'] # use precomputed count score
    else:
        return np.nan

def compute_total_score(df_results):
    """
    Compute the total score for each benchmark result.

    Parameters:
    - df_results: DataFrame with the results of the benchmark

    Returns:
    - df_results: DataFrame with the total score for each benchmark result
    """

    bench_types = df_results['Benchmark Type']

    # Validity Score for List Type Results:
    isListType = bench_types.isin(inf_utils.get_list_benchmark_types())
    df_results['Validity Score'] = 0.0
    df_results.loc[isListType & (df_results['Parsed']==True) & (df_results['HasError']==True), 'Validity Score'] = 0.5
    df_results.loc[isListType & (df_results['Parsed']==True) & (df_results['HasError']==True) & (df_results['ErrorType']=='Missing closing bracket'), 'Validity Score'] = 0.75
    df_results.loc[isListType & (df_results['Parsed']==True) & (df_results['HasError']==False) & (df_results['IsList']==False), 'Validity Score'] = 0.75
    df_results.loc[isListType & (df_results['Parsed']==True) & (df_results['HasError']==False) & (df_results['HasEllipsis']==True), 'Validity Score'] = 0.75
    df_results.loc[isListType & (df_results['Parsed']==True) & (df_results['HasError']==False) & (df_results['RequiredTypeParsing']==True), 'Validity Score'] = 0.75
    df_results.loc[isListType & (df_results['Parsed']==True) & (df_results['HasError']==False) & (df_results['IsList']==True) & (df_results['HasEllipsis']==False) & (df_results['RequiredTypeParsing']==False), 'Validity Score'] = 1.0

    # Validity Score for Single Result Type Results:
    isSingleResultType = bench_types.isin(inf_utils.get_single_result_benchmark_types())
    df_results.loc[isSingleResultType & (df_results['Parsed']==True) & (df_results['HasError']==True), 'Validity Score'] = 0.5
    df_results.loc[isSingleResultType & (df_results['Parsed']==True) & (df_results['HasError']==False), 'Validity Score'] = 1.0

    df_results['Benchmark Score'] = df_results.apply(calc_score, axis=1)
    df_results['Faithfulness Score'] = 1-(df_results['Missing Items (%)'] + df_results['Additional Items (%)'])/2
    df_results.loc[isListType & df_results['Validity Score']>0, 'Overall Score'] = df_results['Validity Score']*(df_results['Benchmark Score'] + df_results['Faithfulness Score'])/2
    df_results.loc[isSingleResultType & df_results['Validity Score']>0, 'Overall Score'] = df_results['Validity Score']*df_results['Benchmark Score']
    df_results.loc[df_results['Validity Score']==0, 'Overall Score'] = 0
    return df_results

from transformers import AutoTokenizer
import os

# get token from env
hf_access_token = os.getenv("HF_ACCESS_TOKEN")
deepseekr_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", token=hf_access_token)

def deepseek_numtokens(input):
    return deepseekr_tokenizer(input, return_tensors="pt").input_ids.shape[1]