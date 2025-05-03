import random
import time
import pylcs
import argparse
import sys

NEIGHBOR_FINDING_TIMEOUT_SECONDS = 3
MAX_NEIGHBOR_TRYING = 5
LOCAL_SEARCH_TIMEOUT_SECONDS = 5
ILS_TIMEOUT_SECONDS = 15
ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT = 10
LOCAL_SEARCH_MAXIMUM_NEIGHBORS = 5
RANDOM_SEED = 3022024

"""
    Given a list, returns `n` sorted random indexes from it
"""
def get_sorted_random_indexes(original_list, n):
    # Get n random indices
    random_indices = random.sample(range(len(original_list)), n)

    # Sort the indices in ascending order
    sorted_indices = sorted(random_indices)

    return sorted_indices

"""
    Given two source code versions, returns a random candidate,
    respecting the partial order of the versions
    Uses the candidate data structure: [('v1', 0), ('v1', 1), ('v1', 2), ('v2', 1)]
"""
def partial_order_random_candidate(v1, v2):
    v1_lines = v1.splitlines()
    v2_lines = v2.splitlines()
    number_lines_v1 = random.randrange(len(v1_lines)+1)
    number_lines_v2 = random.randrange(len(v2_lines)+1)
    candidate_size = number_lines_v1 + number_lines_v2
    
    indexes_v1 = get_sorted_random_indexes(v1_lines, number_lines_v1)
    indexes_v2 = get_sorted_random_indexes(v2_lines, number_lines_v2)
    
    candidate = []
    while(len(candidate) < candidate_size):
        line_source = ''
        if(len(indexes_v1)>0 and len(indexes_v2)>0):
            line_source = random.choice(['v1', 'v2'])
        elif len(indexes_v2)==0:
            line_source='v1'
        else:
            line_source='v2'
        if line_source == 'v1':
            candidate.append(('v1', indexes_v1.pop(0)))
        else:
            candidate.append(('v2', indexes_v2.pop(0)))
            
    candidate_text = get_candidate_text(candidate, v1, v2)
    # we dont allow resolutions equal to v1 or v2 (only combination)
    if candidate_text == v1 or candidate_text == v2:
        return partial_order_random_candidate(v1, v2)
    return candidate

def get_lcs(text1,text2):
    res = pylcs.lcs_sequence_idx(text1, text2)
    lcs = ''.join([text2[i] for i in res if i != -1])
    return lcs

def get_gestalt(text1, text2):
    try:
        if text1 == text2:
            return 1
        return 2*len(get_lcs(text1, text2)) / (len(text1) + len(text2))
    except Exception as e:
        print(e)
        return 0

def evaluate(candidate, v1, v2):
    try:
        v1_similarity = get_gestalt(candidate, v1)
        v2_similarity = get_gestalt(candidate, v2)
        return (v1_similarity + v2_similarity) / 2
    except Exception as e:
        print(e)
        return -1

"""
    Given a candidate and a side (v1 or v2),
    returns the index of the last element from the side in the candidate
"""
def get_last_side_index(candidate, side):
    last_side_index = -1
    for candidate_element in candidate[-1::-1]:
        element_side = candidate_element[0]
        if element_side == side:
            last_side_index = candidate_element[1]
            break
    return last_side_index

'''
    A resolution violates the partial order when there is no way to arrange the 
        chunk lines to compose the resolution without breaking their original order
    Uses candidate data structure, thus the analysis is based on matching lines indexes
'''
def candidate_has_partial_order(v1,v2, candidate):
    
    last_v1_index = -1
    last_v2_index = -1
    for candidate_element in candidate:
        line_side = candidate_element[0]
        line_index = candidate_element[1]
        if line_side == 'v1':
            if line_index < last_v1_index:
                return False
            last_v1_index = line_index
        else:
            if line_index < last_v2_index:
                return False
            last_v2_index = line_index
    return True

"""
    Given a candidate representation, returns a string with the candidate text
"""
def get_candidate_text(indexes, v1, v2):
    candidate = []
    for index in indexes:
        candidate.append(eval(f"{index[0]}.splitlines()[{index[1]}]"))
    return '\n'.join(candidate) + "\n"

'''
    Get a random neighbor and 
        check if the neighbor satisfies the partial order,
        Otherwise, generate another, until one that satisfies is found
    Timeouts if no partial order complaint neighbor is found
'''
def get_next_neighbor(candidate, v1, v2, neighbors):
    random_neighbor = get_random_neighbor(candidate, v1, v2)
    random_neighbor_text = get_candidate_text(random_neighbor, v1, v2)
    start_time = time.time()
    count_tries = 0
    has_partial_order = candidate_has_partial_order(v1, v2, random_neighbor)
    # we dont allow resolutions equal to v1 or v2 (only combination)
    while not has_partial_order or hash(random_neighbor_text) in neighbors or random_neighbor_text == v1 or random_neighbor_text == v2:
        if time.time() - start_time > NEIGHBOR_FINDING_TIMEOUT_SECONDS or count_tries >= MAX_NEIGHBOR_TRYING:
            return None
        random_neighbor = get_random_neighbor(candidate, v1, v2)
        random_neighbor_text = get_candidate_text(random_neighbor, v1, v2)
        count_tries+=1
        has_partial_order = candidate_has_partial_order(v1, v2, random_neighbor)
        if not has_partial_order:
            print('Violates partial order', flush=True)
    return random_neighbor

"""
    Given a candidate, a position and which side to look for (v1 or v2), 
    returns the first index before the position that is from the side
"""
def get_first_index_before_position(candidate, position, side):
    first_index_before = -1
    if position > 0:
        for element in candidate[position-1::-1]:
            element_side = element[0]
            element_index = element[1]
            if element_side == side:
                first_index_before = element_index
                break
    return first_index_before

"""
    Given a candidate, a position and which side to look for (v1 or v2),
    returns the first index after the position that is from the side
"""
def get_first_index_after_position(candidate, position, side):
    first_index_after = float('inf')
    if position < len(candidate):
        for element in candidate[position:]:
            element_side = element[0]
            element_index = element[1]
            if element_side == side:
                first_index_after = element_index
                break
    return first_index_after

'''
    Given a candidate, a position and the two versions,
    returns the lines that can be added to the candidate at the position,
    respecting the partial order
        Candidate example: [('v1', 0), ('v1', 1), ('v1', 2), ('v2', 1)]
'''
def find_feasible_lines_to_add(candidate, v1, v2, position):
    feasible_lines = []
    # Get the lines that are not in the candidate
    v1_lines_indexes = list(range(len(v1.splitlines())))
    v2_lines_indexes = list(range(len(v2.splitlines())))
    for candidate_element in candidate:
        line_side = candidate_element[0]
        line_index = candidate_element[1]
        v1_lines_indexes.remove(line_index) if line_side == 'v1' else v2_lines_indexes.remove(line_index)
    
    # Get the lines that can be added to the candidate
    v1_index_before = get_first_index_before_position(candidate, position, 'v1')
    v2_index_before = get_first_index_before_position(candidate, position, 'v2')
    v1_index_after = get_first_index_after_position(candidate, position, 'v1')
    v2_index_after = get_first_index_after_position(candidate, position, 'v2')

    for line_index in v1_lines_indexes:
        if line_index > v1_index_before and line_index < v1_index_after:
            feasible_lines.append(('v1', line_index))

    for line_index in v2_lines_indexes:
        if line_index > v2_index_before and line_index < v2_index_after:
            feasible_lines.append(('v2', line_index))
    
    return feasible_lines

'''
    Given a candidate, returns a random neighbor
        A neighbor is defined as a candidate that differs
        at most in one line (either by addition, removal or swapping)
        Candidate example: [('v1', 0), ('v1', 1), ('v1', 2), ('v2', 1)]
'''
def get_random_neighbor(candidate, v1, v2):
    neighbor = candidate.copy()
    v1_lines = v1.splitlines()
    v2_lines = v2.splitlines()
    feasible_actions = []
    if len(candidate) > 0:
        random_position = random.randint(0, len(candidate)-1)
        feasible_actions.append('remove')
        if random_position > 0:
                # we can only swap when the other element is from the opposite side
                if candidate[random_position-1][0] != candidate[random_position][0]:
                    feasible_actions.append('swap_before')
        if random_position < len(candidate)-1:
            # we can only swap when the other element is from the opposite side
            if candidate[random_position+1][0] != candidate[random_position][0]:
                feasible_actions.append('swap_after')
        if len(candidate) < len(v1_lines) + len(v2_lines):
            feasible_lines = find_feasible_lines_to_add(candidate, v1, v2, random_position)
            if len(feasible_lines) > 0:
                feasible_actions.append('add')
    else: # candidate is empty, only add is possible
        random_position = random.randint(0, len(candidate))
        feasible_actions = ['add']
        feasible_lines = find_feasible_lines_to_add(candidate, v1, v2, random_position)
    raffled_action = random.choice(feasible_actions)
    
    if raffled_action == 'remove':
        neighbor.pop(random_position)
    elif raffled_action == 'swap_before':
        neighbor[random_position], neighbor[random_position-1] = neighbor[random_position-1], neighbor[random_position]
    elif raffled_action == 'swap_after':
        neighbor[random_position], neighbor[random_position+1] = neighbor[random_position+1], neighbor[random_position]
    else: # add
        if random_position == len(candidate)-1: # a chance to add at the end
            if random.choice([0,1]) == 1:
                # we can only add at the end when either the last v1 index or last v2 index 
                # are smaller than the length of v1 and v2, respectively
                last_v1_index = get_last_side_index(candidate, 'v1')
                last_v2_index = get_last_side_index(candidate, 'v2')
                if last_v1_index < len(v1.splitlines())-1 or last_v2_index < len(v2.splitlines())-1:
                    random_position += 1
                    # need to update the feasible lines beause the position has changed
                    feasible_lines = find_feasible_lines_to_add(candidate, v1, v2, random_position)
        if len(feasible_lines) > 0:
            random_line = random.choice(feasible_lines)
            neighbor.insert(random_position, random_line)
        else:
            print('No feasible lines to add')
            print('candidate', candidate, 'position:', random_position)
    # print('raffled_action', raffled_action, 'random_position', random_position)
    return neighbor

def get_fittest(neighbors):
    fittest_neighbor = None
    fittest_value = -9999999
    for neighbor in neighbors.items():
        if neighbor[1][1] > fittest_value:
            fittest_neighbor = neighbor[1][0]
            fittest_value = neighbor[1][1]
    return fittest_neighbor, fittest_value

def local_search(starting_candidate, starting_candidate_fitness, n, v1, v2, source, depth):
    neighbors = {}
    start_time = time.time()
    tries_count = 0
    while len(neighbors) < n and time.time() - start_time < LOCAL_SEARCH_TIMEOUT_SECONDS and tries_count < MAX_NEIGHBOR_TRYING:
        s_new = get_next_neighbor(starting_candidate, v1, v2, neighbors)
        if s_new != None:
            f_new = evaluate(get_candidate_text(s_new, v1, v2), v1, v2)
            neighbors[hash(get_candidate_text(s_new, v1, v2))] = [s_new, f_new]
        else:
            tries_count+=1

    s_candidate, f_candidate = get_fittest(neighbors)
    if f_candidate > starting_candidate_fitness:
        return local_search(s_candidate, f_candidate, n, v1, v2, 'ls_in', depth+1)
    else:
        return starting_candidate, starting_candidate_fitness

def pertubate(candidate, v1, v2):
    return partial_order_random_candidate(v1, v2)

def ils_resolution(v1, v2):

    s_star = partial_order_random_candidate(v1, v2)
    f_star = evaluate(get_candidate_text(s_star, v1, v2), v1, v2)

    s_star, f_star = local_search(s_star, f_star, LOCAL_SEARCH_MAXIMUM_NEIGHBORS, v1, v2, 'initial', 0)

    start_time = time.time()
    iteration_number = 1
    iterations_without_improvement = 0
    while (time.time() - start_time < ILS_TIMEOUT_SECONDS) and iterations_without_improvement <= ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT:
        s_new = pertubate(s_star, v1, v2)
        f_new = evaluate(s_new, v1, v2)


        s_star_new, f_star_new = local_search(s_new, f_new, LOCAL_SEARCH_MAXIMUM_NEIGHBORS, v1, v2, f'it-{iteration_number}', 0)

        if f_star_new > f_star:
            s_star = s_star_new
            f_star = f_star_new
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1
        iteration_number+=1
    return s_star, f_star


def read_file_content(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"Error: File not found: '{filename}'", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file '{filename}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading '{filename}': {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='SBCR. Please inform v1 and v2 files.'
    )

    parser.add_argument(
        'v1',
        type=str,
        help='Path to the file containing the content for v1.'
    )

    parser.add_argument(
        'v2',
        type=str,
        help='Path to the file containing the content for v2.'
    )

    args = parser.parse_args()


    v1 = read_file_content(args.file_v1)
    v2 = read_file_content(args.file_v2)

    ils_resolution(v1, v2)

if __name__ == "__main__":
    main()