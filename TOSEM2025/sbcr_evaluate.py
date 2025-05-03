import pandas as pd
import random
import os
import subprocess
import time
import pylcs
from datetime import datetime
import sys

NEIGHBOR_FINDING_TIMEOUT_SECONDS = 3
MAX_NEIGHBOR_TRYING = 200
LOCAL_SEARCH_TIMEOUT_SECONDS = 5
ILS_TIMEOUT_SECONDS = 100
ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT = 5
LOCAL_SEARCH_MAXIMUM_NEIGHBORS = 5
DEBUG_ACTIVE = False
SAVE_ITERATION_INFO = True
RANDOM_SEED = 3022024
DATASET = "dataset1"
SAVE_CANDIDATE = True
EXECUTION_NAME = DATASET


if len(sys.argv) > 1:
    DATASET = sys.argv[1]
    EXECUTION_NAME = DATASET
    if len(sys.argv) > 2:
            # configuration example: "5,5,10"
            # The first value in the tuple is the maximum number of neighbors to be generated in the local search
            # The second value is the maximum number of iterations without improvement in the ILS
            # The third value is the timeout in seconds for the ILS
        CONFIGURATION = sys.argv[2]
        LOCAL_SEARCH_MAXIMUM_NEIGHBORS = int(CONFIGURATION.split(',')[0].replace('"','').strip())
        ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT = int(CONFIGURATION.split(',')[1].strip())
        ILS_TIMEOUT_SECONDS = int(CONFIGURATION.split(',')[2].replace('"','').strip())
        print(f"Configuration: {LOCAL_SEARCH_MAXIMUM_NEIGHBORS}, {ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT}, {ILS_TIMEOUT_SECONDS}")
        if len(sys.argv) > 3:
            EXECUTION_NAME = sys.argv[3]

OUTPUT_FOLDER = f"{EXECUTION_NAME}/OUTPUT"       
DEBUG_FOLDER = f"{EXECUTION_NAME}/DEBUG"
RESULTS_FOLDER = f"{EXECUTION_NAME}/RESULTS"
CANDIDATES_FOLDER = f"{OUTPUT_FOLDER}/candidates_tunning"
DEBUG_REPOSITORY_PATH = f"{DEBUG_FOLDER}/debug_timeout_{DATASET}"
ORIGINAL_DEBUG_REPOSITORY_PATH = DEBUG_REPOSITORY_PATH
TMP_FOLDER = f"{EXECUTION_NAME}/tmp"
        
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)
RESULT_FILE = f'{RESULTS_FOLDER}/results_evaluate_{DATASET}_{EXECUTION_NAME}_seed-{RANDOM_SEED}.xlsx'

random.seed(RANDOM_SEED)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if DEBUG_ACTIVE:
    if not os.path.exists(DEBUG_FOLDER):
        os.makedirs(DEBUG_FOLDER)
if not os.path.exists(TMP_FOLDER):
    os.makedirs(TMP_FOLDER)

"""
    Given a candidate representation, returns a string with the candidate text
"""
def get_candidate_text(indexes, v1, v2):
    candidate = []
    for index in indexes:
        candidate.append(eval(f"{index[0]}.splitlines()[{index[1]}]"))
    return '\n'.join(candidate) + "\n"

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

'''
    Cleans the chunk resolution text by removing potential context lines.
    We assume that 3 context lines are used before and after the conflicting chunk
    If the solution's context is different from the chunk's context, returns None
'''
def get_clean_solution(solution, before_context, after_context):
    solution = normalize_lines(solution.splitlines()).copy()
    before_context = normalize_lines(before_context.splitlines())
    after_context = normalize_lines(after_context.splitlines())
    
    if len(solution) >= 6:
        # what is the last line from the three first solution lines that is present in the before_context?
        solution_before_context_candidate = solution[:3]
        last_line_before_context = 0
        for index, line in enumerate(solution_before_context_candidate):
            if line in before_context:
                last_line_before_context = index
        
        # what is the first line from the last three solution lines that is present in the after_context?
        solution_after_context_candidate = solution[-3:]
        first_line_after_context = len(solution)
        for index, line in enumerate(solution_after_context_candidate):
            if line in after_context:
                first_line_after_context = len(solution) - (3 - index)
                break
        solution_before_context = solution[:last_line_before_context+1]
        solution_after_context = solution[first_line_after_context:]
        if (equivalent_context(solution_before_context, before_context) 
            and equivalent_context(solution_after_context, after_context)):
            return solution[last_line_before_context+1:first_line_after_context]
        else:
            return None
    return solution 

def normalize_line(line):
    return line

def normalize_lines(lines):
    normalized_lines = []
    for line in lines:
            normalized_lines.append(normalize_line(line))
    return normalized_lines

def equivalent_context(context_solution, context_chunk):
    context_solution = remove_empty_lines(context_solution)
    context_chunk = remove_empty_lines(context_chunk)
    if len(context_solution) != len(context_chunk):
        return False
    for i in range(len(context_solution)):
        if context_solution[i] != context_chunk[i]:
            return False
    return True

def remove_empty_lines(lines):
    cleaned_lines = []
    for line in lines:
        if line != '':
            cleaned_lines.append(line)
    return cleaned_lines

def get_lcs(text1,text2):
    res = pylcs.lcs_sequence_idx(text1, text2)
    lcs = ''.join([text2[i] for i in res if i != -1])
    return lcs

def get_gestalt(file1,file2):
    try:
        with open(file1, 'r', encoding='iso-8859-1') as f1:
            file1_text = f1.read()
        with open(file2, 'r', encoding='iso-8859-1') as f2:
            file2_text = f2.read()
            if file1_text == file2_text:
                return 1
            return 2*len(get_lcs(file1_text, file2_text)) / (len(file1_text) + len(file2_text))
    except Exception as e:
        print(e)
        return 0
    
def write_file(content, file_name):
    with open(file_name, 'w') as f:
        f.write(str(content))

def execute_command_timeout(command, timeout, print_output=False):
    try:
        print(f'{time.ctime()} ### Executing command: ', command, flush=True)
        result = subprocess.check_output([command], stderr=subprocess.STDOUT, text=True, shell=True, timeout=timeout)
        return result
    except subprocess.CalledProcessError as e:
        # print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output), flush=True)
        return e.output

def save_candidate(candidate, name):
    if DEBUG_ACTIVE:
        write_file(candidate, f'{DEBUG_REPOSITORY_PATH}/{name}')

def save_iterations_info(iterations_info):
    if SAVE_ITERATION_INFO:
        if not os.path.exists(f'{DEBUG_REPOSITORY_PATH}/iterations_info.csv'):
            with open(f'{DEBUG_REPOSITORY_PATH}/iterations_info.csv', 'w') as f:
                f.write('LOCAL_SEARCH_MAXIMUM_NEIGHBORS,ILS_TIMEOUT_SECONDS,MAX_ITERATIONS_WITHOUT_IMPROVEMENT,iteration_number,fitness,elapsed_time,iterations_without_improvement\n')
        with open(f'{DEBUG_REPOSITORY_PATH}/iterations_info.csv', 'a') as f:
            for iteration_info in iterations_info:
                f.write(f'{LOCAL_SEARCH_MAXIMUM_NEIGHBORS},{ILS_TIMEOUT_SECONDS},{ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT},{iteration_info[0]},{iteration_info[1]},{iteration_info[2]},{iteration_info[3]}\n')

def clear_iteration_info():
    if SAVE_ITERATION_INFO:
        if os.path.exists(f'{DEBUG_REPOSITORY_PATH}'):
            for root, dirs, files in os.walk(f'{DEBUG_REPOSITORY_PATH}', topdown=False):
                for dir in dirs:
                    if os.path.exists(f'{DEBUG_REPOSITORY_PATH}/{dir}/iterations_info.csv'):
                        os.remove(f'{DEBUG_REPOSITORY_PATH}/{dir}/iterations_info.csv')


def get_current_timestamp():
    current_datetime = datetime.now()
    return current_datetime.strftime("%d-%m-%Y_%H-%M-%S-%f")

def evaluate(candidate, v1, v2):
    try:
        write_file(candidate, f'{TMP_FOLDER}/candidate')
        write_file(v1, f'{TMP_FOLDER}/v1')
        write_file(v2, f'{TMP_FOLDER}/v2')

        v1_similarity = get_gestalt(f'{TMP_FOLDER}/candidate', f'{TMP_FOLDER}/v1')
        v2_similarity = get_gestalt(f'{TMP_FOLDER}/candidate', f'{TMP_FOLDER}/v2')
        return (v1_similarity + v2_similarity) / 2
    except Exception as e:
        print(e)
        return -1

def compare(content1, content2):
    try:
        content1 = '\n'.join(remove_empty_lines(content1.splitlines()))
        content2 = '\n'.join(remove_empty_lines(content2.splitlines()))
        write_file(content1, f'{TMP_FOLDER}/content1')
        write_file(content2, f'{TMP_FOLDER}/content2')
        content_similarity = get_gestalt(f'{TMP_FOLDER}/content1', f'{TMP_FOLDER}/content2')
        
        return content_similarity
    except Exception as e:
        print(e)
        return -1

'''
    Gets one random line of available_lines and 
        adds it to the current_lines at the parameter position
'''
def add_line(current_lines, position, available_lines):
    current_lines.insert(position,random.choice(available_lines))

def get_available_lines(current_lines, all_possible_lines):
    available_lines = all_possible_lines.copy()
    for line in current_lines:
        available_lines.remove(line)
    return available_lines

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

            save_candidate(get_candidate_text(s_new, v1, v2), f'neighbor_{source}-{depth}_{get_current_timestamp()}')
            save_candidate(f_new, f'neighbor_{source}-{depth}_{get_current_timestamp()}_f')
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

    save_candidate(v1+'\n'+v2, 'available_lines')

    s_star = partial_order_random_candidate(v1, v2)
    f_star = evaluate(get_candidate_text(s_star, v1, v2), v1, v2)

    save_candidate(get_candidate_text(s_star, v1, v2), 'initial_candidate')
    save_candidate(f_star, 'initial_candidate_f')

    s_star, f_star = local_search(s_star, f_star, LOCAL_SEARCH_MAXIMUM_NEIGHBORS, v1, v2, 'initial', 0)

    save_candidate(get_candidate_text(s_star, v1, v2), 'initial_candidate_from_localsearch')
    save_candidate(f_star, 'initial_candidate_from_localsearch_f')

    start_time = time.time()
    iteration_number = 1
    iterations_without_improvement = 0
    iterations_info = []
    while (time.time() - start_time < ILS_TIMEOUT_SECONDS) and iterations_without_improvement <= ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT:
        s_new = pertubate(s_star, v1, v2)
        f_new = evaluate(s_new, v1, v2)

        save_candidate(get_candidate_text(s_star, v1, v2), f'perturbed_candidate_it-{iteration_number}')
        save_candidate(f_new, f'perturbed_candidate_it-{iteration_number}_f')
        save_candidate(time.time() - start_time, f'perturbed_candidate_it-{iteration_number}_t')

        s_star_new, f_star_new = local_search(s_new, f_new, LOCAL_SEARCH_MAXIMUM_NEIGHBORS, v1, v2, f'it-{iteration_number}', 0)

        save_candidate(get_candidate_text(s_star_new, v1, v2), f'candidate_after_localsearch_it-{iteration_number}')
        save_candidate(f_star_new, f'candidate_after_localsearch_it-{iteration_number}_f')
        save_candidate(time.time() - start_time, f'candidate_after_localsearch_it-{iteration_number}_t')
        original_iterations = iterations_without_improvement
        if f_star_new > f_star:
            s_star = s_star_new
            f_star = f_star_new
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1
        iterations_info.append([iteration_number, f_star, time.time() - start_time, original_iterations])
        iteration_number+=1
    save_iterations_info(iterations_info)
    return s_star, f_star

def adapt_dataset(df):
    if 'chunk_id' not in df.columns:
        df['chunk_id'] = df['merge_id'].astype(str) + '-' + df['chunk_number'].astype(str)
    if 'v1' not in df.columns:
        df.rename(columns={'all_raw_a':'v1', 'all_raw_b':'v2', 'all_raw_res':'solution'}, inplace=True)
      

def execute_experiment():
    global DEBUG_REPOSITORY_PATH, LOCAL_SEARCH_MAXIMUM_NEIGHBORS, ILS_TIMEOUT_SECONDS, ILS_STOP_CRITERIA_ITERATIONS_WITHOUT_IMPROVEMENT
    df_chunks = pd.read_json(f"../data/{DATASET}_testing.json")
    df_chunks = df_chunks.sample(frac=1, random_state=RANDOM_SEED)
    data = []
    adapt_dataset(df_chunks)
    clear_iteration_info()
    count = 1
    chunk_count = 1
    columns = ['chunk_id', 'fitness', 'solution_sim_lcs', 'status', 'time_seconds']
    analyzed_chunk_ids = []
    if os.path.exists(RESULT_FILE):
        df = pd.read_excel(RESULT_FILE, engine='openpyxl')
        data = df.values.tolist()
        analyzed_chunk_ids = df['chunk_id'].unique()
    for index, row in df_chunks.iterrows():
        chunk_id = row['chunk_id']
        if chunk_id not in analyzed_chunk_ids:
            print(f"{time.ctime()} ### Analyzing chunk ({chunk_id}) {chunk_count} of {len(df_chunks)}. Dataset {DATASET}", flush=True)
            if len(row['v1'])<20000 and len(row['v2']) < 20000:
                v1 = row['v1']
                v2 = row['v2']
                if 'before_context' in df_chunks.columns:
                    solution = get_clean_solution(row['solution'], row['before_context'], row['after_context'])
                    solution = '\n'.join(solution)
                else:
                    solution = row['solution']
                if solution != None and len(solution) > 0:
                    try:
                        DEBUG_REPOSITORY_PATH = f"{ORIGINAL_DEBUG_REPOSITORY_PATH}/{chunk_id}"
                        if not os.path.exists(f"{DEBUG_REPOSITORY_PATH}"):
                            os.makedirs(DEBUG_REPOSITORY_PATH, exist_ok=True) 
                        start_time = time.time()
                        candidate, fitness = ils_resolution(v1, v2)
                        candidate = get_candidate_text(candidate, v1, v2)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        if SAVE_CANDIDATE:
                            if not os.path.exists(f"{OUTPUT_FOLDER}"):
                                os.makedirs(OUTPUT_FOLDER)
                            write_file(candidate, f"{OUTPUT_FOLDER}/{chunk_id}")
                        lcs_solution_similarity = compare(solution, candidate)
                        # print(f"Elapsed time: {elapsed_time}, Fitness: {fitness}, LCS Similarity: {lcs_solution_similarity}", flush=True)
                        data.append([row['chunk_id'], fitness, lcs_solution_similarity, 'ok', elapsed_time])
                        if count % 5 == 0:
                            pd.DataFrame(data, columns=columns).to_excel(RESULT_FILE, index=False)
                            count=0
                        count+=1
                    except Exception as e:
                        error = e
                        print(e)
                        data.append([row['chunk_id'], "", "", e, ""])
                else:
                    print('Empty solution', flush=True)
            else:
                print(len(row['v1']))
                print(len(row['v2']))
                data.append([row['chunk_id'], "", "", "OUT_OF_LIMITS", ""])
        else:
            print(f"Chunk ({chunk_id}) {chunk_count} of {len(df_chunks)} was already analyzed before.", flush=True)
        chunk_count+=1
    pd.DataFrame(data, columns=columns).to_excel(RESULT_FILE, index=False)
execute_experiment()