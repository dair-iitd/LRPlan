def normalize_action(action_name):
    """
    Normalize an action name by stripping whitespace and removing a trailing '*' if present.
    """
    return action_name.strip().rstrip('*').strip()

def is_idle_action(task_action):
    """
    Returns True if the task action is considered idle (its name ends with '*').
    """
    return task_action.strip().endswith('*')

def parse_plan(plan):
    """
    Parses a plan (list of strings like "0: pick rice") into a dictionary mapping time steps to lists of actions.
    
    Enforces:
      - Every time step from 0 to the maximum time must be explicitly defined.
      - If a time step contains 'wait', it must be the only entry.
    """
    plan_dict = {}
    for entry in plan:
        try:
            time_str, action = entry.split(":", 1)
        except ValueError:
            raise ValueError(f"Plan entry '{entry}' is not formatted as 'time: action'.")
        t = int(time_str.strip())
        action = action.strip()
        if t not in plan_dict:
            plan_dict[t] = []
        plan_dict[t].append(action)
    if plan_dict:
        max_time = max(plan_dict.keys())
    else:
        max_time = -1
    for t in range(max_time + 1):
        if t not in plan_dict:
            raise ValueError(f"Missing plan entry for time step {t}. Every time step must be defined (use 'wait' if no action is initiated).")
        actions = plan_dict[t]
        if any(normalize_action(a).lower() == "wait" for a in actions) and len(actions) > 1:
            raise ValueError(f"Time step {t} contains 'wait' along with other actions, which is not allowed.")
    return plan_dict

def consume_action(available_actions, act):
    """
    Consumes one occurrence of an action (by normalized name) from the available_actions list.
    Returns True if an occurrence was found and removed, else False.
    """
    norm_act = normalize_action(act)
    for i, a in enumerate(available_actions):
        if normalize_action(a) == norm_act:
            del available_actions[i]
            return True
    return False

def evaluate_plan(tasks, plan):
    """
    Evaluate the plan against a set of tasks with the following rules:
    
      1. Dependency Order: An action is only counted if all its dependency actions (per the task’s dependency graph)
         have been completed *before* it is initiated. If not, it is not scheduled and does not contribute progress.
      
      2. Wait Enforcement for Non-Idle Actions:
         - For a non-idle action with duration n, once it is initiated the next n-1 time steps (for that task) must be "wait"
           (i.e. no new non-idle action may be started).
         - This is implemented by “locking” the task for non-idle actions until the current one’s finish time.
         - **New requirement:** While any non-idle action is underway (global lock), no actions (even idle) may be initiated.
           
      3. Progress Speed:  
         At each simulation time step, for each task, the percentage progress is computed as the sum of durations for completed
         sub‑tasks divided by the total duration.
         
      4. Completion Speed:  
         For each task that is fully completed, the completion time is given as the last subtask's finish time.
         
      5. Unique Consumption:  
         Each plan action occurrence is “consumed” once. That is, if a subtask (e.g. "wash dish") is required by two tasks,
         it must appear twice in the plan (or in two distinct time steps) to progress both tasks.
    
    Parameters:
      tasks: A dict where each key is a task name and each value is a two-element list:
             [ durations_dict, dependencies_dict ]
             - durations_dict maps action names (e.g. "pick rice", "cook rice in pot*") to their duration (in minutes).
             - dependencies_dict maps an action name to a list of dependency action names.
      plan: A list of strings representing time-stamped actions (e.g. "0: pick rice", "1: wait", etc.)
    
    Returns:
      A dict with:
         - For each task: its percentage complete, whether it was fully completed,
           and a list of errors encountered (dependency violations, wait violations, etc.).
         - For each task, a "progress_speed" time series showing percentage complete per time step (pruned up to the last action finish time).
         - For each task, its "completion_speed" (i.e. the time when the task was fully completed, or None).
         - The overall count of fully completed tasks and the total number of tasks.
    """
    # Parse the global plan.
    plan_dict = parse_plan(plan)
    
    # Use a fixed order for tasks.
    tasks_order = list(tasks.keys())
    
    # Initialize a schedule for each task.
    # For each task, we record scheduled actions as:
    #    schedule[task][action] = { start_time, finish_time, completed, duration, idle }
    schedule = { task: {} for task in tasks }
    
    # For collecting errors per task.
    errors = { task: [] for task in tasks }
    
    # For non-idle actions, maintain a lock per task (the finish time until which no new non-idle action may start).
    # (This was previously per-task; now we enforce a global check later.)
    non_idle_lock = { task: None for task in tasks }
    
    # For progress speed: record progress (percentage complete) per time step for each task.
    progress_speed = { task: {} for task in tasks }
    
    # Determine simulation end time.
    max_plan_time = max(plan_dict.keys()) if plan_dict else 0
    max_duration = max(max(durations.values()) for durations, _ in tasks.values()) if tasks else 0
    simulation_end = max_plan_time + max_duration + 1  # +1 to allow completions
    
    # Simulation loop: step through each minute.
    for t in range(simulation_end):
        # First, mark any actions that finish exactly at this time.
        for task in tasks_order:
            for act, record in schedule[task].items():
                if not record.get("completed", False) and record["finish_time"] == t:
                    record["completed"] = True
                    # If this was a non-idle action, release its lock.
                    if not record["idle"]:
                        if non_idle_lock[task] == record["finish_time"]:
                            non_idle_lock[task] = None
        
        # At the end of this time step, record progress for each task.
        for task, (durations, _) in tasks.items():
            total_duration = sum(durations.values())
            completed_duration = sum(durations[act] for act in durations 
                                     if act in schedule[task] and schedule[task][act].get("completed", False))
            percentage_complete = (completed_duration / total_duration) * 100 if total_duration > 0 else 0
            progress_speed[task][t] = percentage_complete
        
        # Process the plan entry for time t (if any).
        if t in plan_dict:
            # Create a copy of the actions available at this time step.
            available_actions = list(plan_dict[t])
            # If the only entry is "wait", then skip processing for this time step.
            if len(available_actions) == 1 and normalize_action(available_actions[0]).lower() == "wait":
                continue
            
            # Global lock check: if any task has a non-idle lock active, no actions may be started.
            if any(non_idle_lock[task] is not None and t < non_idle_lock[task] for task in tasks_order):
                for task in tasks_order:
                    # Build candidate actions for this task.
                    durations = tasks[task][0]
                    candidate_actions = []
                    for act_str in available_actions:
                        norm_act = normalize_action(act_str)
                        if norm_act.lower() == "wait":
                            continue
                        for task_act in durations:
                            if normalize_action(task_act) == norm_act:
                                candidate_actions.append(task_act)
                    candidate_actions = list(set(candidate_actions))
                    for act in candidate_actions:
                        errors[task].append(
                            f"Action '{act}' initiated at time {t} while a non-idle action is underway (global lock)."
                        )
                continue  # Skip processing all tasks at this time step.
            
            # If no global lock is active, iterate over tasks in a fixed order.
            for task in tasks_order:
                durations, deps = tasks[task]
                
                # Build candidate actions for this task from the available actions.
                candidate_actions = []
                for act_str in available_actions:
                    norm_act = normalize_action(act_str)
                    if norm_act.lower() == "wait":
                        continue
                    for task_act in durations:
                        if normalize_action(task_act) == norm_act:
                            candidate_actions.append(task_act)
                candidate_actions = list(set(candidate_actions))
                
                # Process non-idle candidates first.
                non_idle_candidates = [act for act in candidate_actions if not is_idle_action(act)]
                if len(non_idle_candidates) > 1:
                    errors[task].append(f"At time {t}, more than one non-idle action initiated: {non_idle_candidates}.")
                    non_idle_candidates = []
                if non_idle_candidates:
                    act = non_idle_candidates[0]
                    if act in schedule[task]:
                        errors[task].append(f"Non-idle action '{act}' initiated more than once (at time {t}).")
                    else:
                        deps_met = True
                        if act in deps:
                            for dep in deps[act]:
                                dep_candidate = None
                                for candidate in durations:
                                    if normalize_action(candidate) == normalize_action(dep):
                                        dep_candidate = candidate
                                        break
                                if dep_candidate is None:
                                    errors[task].append(f"Dependency '{dep}' for action '{act}' not found in task.")
                                    deps_met = False
                                    break
                                if (dep_candidate not in schedule[task] or 
                                    not schedule[task][dep_candidate].get("completed", False) or 
                                    schedule[task][dep_candidate]["finish_time"] > t):
                                    errors[task].append(
                                        f"Non-idle action '{act}' started at time {t} before dependency '{dep_candidate}' completed."
                                    )
                                    deps_met = False
                                    break
                        if deps_met:
                            duration = durations[act]
                            schedule[task][act] = {
                                "start_time": t,
                                "finish_time": t + duration,
                                "completed": False,
                                "duration": duration,
                                "idle": False
                            }
                            # Lock the task until this non-idle action completes.
                            non_idle_lock[task] = t + duration
                            consume_action(available_actions, act)
                
                # Process idle candidates (multiple idle actions are allowed).
                idle_candidates = [act for act in candidate_actions if is_idle_action(act)]
                for act in idle_candidates:
                    if act in schedule[task]:
                        errors[task].append(f"Idle action '{act}' initiated more than once (at time {t}).")
                        continue
                    deps_met = True
                    if act in deps:
                        for dep in deps[act]:
                            dep_candidate = None
                            for candidate in durations:
                                if normalize_action(candidate) == normalize_action(dep):
                                    dep_candidate = candidate
                                    break
                            if dep_candidate is None:
                                errors[task].append(f"Dependency '{dep}' for idle action '{act}' not found in task.")
                                deps_met = False
                                break
                            if (dep_candidate not in schedule[task] or 
                                not schedule[task][dep_candidate].get("completed", False) or 
                                schedule[task][dep_candidate]["finish_time"] > t):
                                errors[task].append(
                                    f"Idle action '{act}' initiated at time {t} before dependency '{dep_candidate}' completed."
                                )
                                deps_met = False
                                break
                    if deps_met:
                        schedule[task][act] = {
                            "start_time": t,
                            "finish_time": t + durations[act],
                            "completed": False,
                            "duration": durations[act],
                            "idle": True
                        }
                        consume_action(available_actions, act)
    
    # --- Prune progress_speed: for each task, only keep time steps up to the last action finish time.
    for task in progress_speed:
        if schedule[task]:
            max_finish = max(record["finish_time"] for record in schedule[task].values())
            progress_speed[task] = { t: prog for t, prog in progress_speed[task].items() if t <= max_finish }
        else:
            progress_speed[task] = { t: 0 for t in range(simulation_end) }
    
    # After simulation, compute results per task.
    final_results = {}
    completion_speed = {}  # Completion time per task.
    for task, (durations, _) in tasks.items():
        total_duration = sum(durations.values())
        completed_duration = 0
        last_finish_time = 0
        for act in durations:
            if act in schedule[task] and schedule[task][act].get("completed", False):
                completed_duration += durations[act]
                last_finish_time = max(last_finish_time, schedule[task][act]["finish_time"])
        percentage_complete = (completed_duration / total_duration) * 100 if total_duration > 0 else 0
        fully_completed = all(
            act in schedule[task] and schedule[task][act].get("completed", False)
            for act in durations
        )
        final_results[task] = {
            "percentage_complete": percentage_complete,
            "fully_completed": fully_completed,
            "errors": errors[task]
        }
        completion_speed[task] = last_finish_time if fully_completed else None
    
    num_fully_completed = sum(1 for task in final_results if final_results[task]["fully_completed"])
    total_tasks = len(tasks)
    
    overall_result = {
        "final_results": final_results,
        "progress_speed": progress_speed,         # Pruned progress time series for each task.
        "completion_time": completion_speed,       # Completion time per task.
        "num_fully_completed": num_fully_completed,
        "total_tasks": total_tasks
    }
    return overall_result


# return scores, detailed scores, detailed feedback
# cumulative # divided # idx query plan - detailed_evaluation
def timearena_feedback(set_type: str, plans: list, indices: list, llm_config_list: list = None, llm_feedback_flag = False):
    tcost = 0
    prog = 0
    cnt = 0
    pcnt = 0
    fullcnt = 0
    avgctime = 0
    avgpspeed = [0,0]
    lprog = [0]*3
    lcnt = [0]*3
    lpcnt = [0]*3
    lfullcnt = [0]*3
    lavgctime = [0]*3
    lavgpspeed = [[0,0]]*3
    cnt_idle_violated = 0
    lcnt_idle_violated = [0]*3
    cnt_dependency_violated = 0
    lcnt_dependency_violated = [0]*3
    import copy
    data = copy.deepcopy(plans)
    from tqdm import tqdm
    for i in tqdm(range(len(data))):
        if "plan" in data[i]:
            try:
                result = evaluate_plan(data[i]['dependency_graph'], data[i]['plan'])
            except Exception as e:
                print(f"Error evaluating plan at index {i}: {e}")
                result = {
                    "final_results": {},
                    "error": str(e)
                }
                data[i]['eval'] = result
                continue
            data[i]['eval'] = result
            
            try:
                tcost += data[i]['cost']['usage_including_cached_inference']['total_cost']
                if 'cost_agents' in data[i]:
                    tcost += data[i]['cost_agents']['usage_including_cached_inference']['total_cost']
            except:
                pass
            flag = True
            ind = result['total_tasks'] - 1
            for j in result['final_results']:
                flag_idle = False
                flag_dependency = False
                for k in result['final_results'][j]['errors']:
                    if "dependency" in k:
                        flag_dependency = True
                    if "global lock" in k:
                        flag_idle = True
                
                if flag_idle:
                    cnt_idle_violated += 1
                    lcnt_idle_violated[ind] += 1
                if flag_dependency:
                    cnt_dependency_violated += 1
                    lcnt_dependency_violated[ind] += 1
                
                prog += result['final_results'][j]['percentage_complete'] 
                lprog[ind] += result['final_results'][j]['percentage_complete']
                pcnt += 1
                lpcnt[ind] += 1
                if result['final_results'][j]['fully_completed']:
                    cnt += 1
                    lcnt[ind] += 1
                else:
                    flag = False
            if flag:
                fullcnt += 1
                lfullcnt[ind] += 1
            for j in result['progress_speed']:
                try:
                    avgpspeed[0] +=  result['progress_speed'][j][(len(list(result['progress_speed'][j].keys()))-1)]
                    avgpspeed[1] +=  len(list(result['progress_speed'][j].keys()))-1
                    lavgpspeed[ind][0] +=  result['progress_speed'][j][(len(list(result['progress_speed'][j].keys()))-1)]
                    lavgpspeed[ind][1] +=  len(list(result['progress_speed'][j].keys()))-1 
                except KeyError as e:
                    print(f"Error in progress speed for task {j}: {result['progress_speed'][j]}")
                    raise e
                
            for j in result['completion_time']:
                if result['completion_time'][j] is not None:
                    avgctime += result['completion_time'][j]
                    lavgctime[ind] += result['completion_time'][j]
            
            
        else:
            print(f"missing plan at index {i}")
    ret = {}   
    ret["scores"] = {
        "fully_completed_samples": fullcnt,
        "fully_completed_samples_percentage": fullcnt / len(data) if len(data) > 0 else 0,
        "completed_tasks": cnt,
        "completed_tasks_percentage": cnt / pcnt if pcnt > 0 else 0,
        "avg_progress": prog / pcnt if pcnt > 0 else 0,
        "avg_completion_time": avgctime / cnt if cnt > 0 else 0,
        "avg_progress_speed": avgpspeed[0] / avgpspeed[1] if avgpspeed[1] > 0 else 0,
        "tasks_with_idle_violation" : cnt_idle_violated,
        "tasks_with_dependency_violation" : cnt_dependency_violated,
        "total_tasks": pcnt,
        "total_samples": len(data),
        "cost": tcost,   
    }
    ret["detailed_scores"] = []
    for i in range(3):
        a={}
        a["fully_completed_samples"] = lfullcnt[i]
        a["completed_tasks"] = lcnt[i]
        a["avg_progress"] = lprog[i] / lpcnt[i] if lpcnt[i] > 0 else 0
        a["avg_completion_time"] = lavgctime[i] / lcnt[i] if lcnt[i] > 0 else 0
        a["avg_progress_speed"] = lavgpspeed[i][0] / lavgpspeed[i][1] if lavgpspeed[i][1] > 0 else 0
        a["tasks_with_idle_violation"] = lcnt_idle_violated[i]
        a["tasks_with_dependency_violation"] = lcnt_dependency_violated[i]
        a["#tasks"] = i+1
        a["total_tasks"] = lpcnt[i]
        
        ret["detailed_scores"].append(a)
    ret["detailed_feedback"] = []
    for i in range(len(data)):
        if "plan" in data[i]:
            if "eval" in data[i]:
                b = {}
                b["id"] = data[i]["id"]
                b["query"] = data[i]["query"]
                b["plan"] = data[i]["plan"]
                b["detailed_evaluation"] = data[i]["eval"]

                if llm_feedback_flag==True:
                    from agent_utils import get_llm_feedback
                    llm_feedback, feedback_cost = get_llm_feedback(b["query"], b["plan"], b["detailed_evaluation"]["final_results"], llm_config_list)
                    b["llm_feedback"] = llm_feedback
                    b["feedback_cost"] = feedback_cost

                ret["detailed_feedback"].append(b)
    return ret["scores"], ret["detailed_scores"], ret["detailed_feedback"]
        

# Example usage:
if __name__ == "__main__":
    tasks = {
        "cooking1": [
            {
                "wash dish": 3,
                "pick rice": 2,
                "pick beef": 2,
                "cook rice in pot*": 4,
                "add rice to dish": 2,
                "chop beef": 3,
                "fry beef in fryer*": 5,
                "add beef to dish": 2
            },
            {
                "cook rice in pot*": ["pick rice"],
                "add rice to dish": ["cook rice in pot*", "wash dish"],
                "chop beef": ["pick beef"],
                "fry beef in fryer*": ["chop beef"],
                "add beef to dish": ["fry beef in fryer*", "wash dish"]
            }
        ],
        "cooking2": [
            {
                "wash dish": 3,
                "pick noodle": 1,
                "cook noodle in pot*": 5,
                "add noodle to dish": 2,
                "pick mushroom": 2,
                "chop mushroom": 3,
                "fry mushroom in fryer*": 2,
                "add mushroom to dish": 2,
                "pick shrimp": 1,
                "chop shrimp": 2,
                "fry shrimp in fryer*": 4,
                "add shrimp to dish": 2
            },
            {
                "cook noodle in pot*": ["pick noodle"],
                "add noodle to dish": ["cook noodle in pot*", "wash dish"],
                "chop mushroom": ["pick mushroom"],
                "fry mushroom in fryer*": ["chop mushroom"],
                "add mushroom to dish": ["fry mushroom in fryer*", "wash dish"],
                "chop shrimp": ["pick shrimp"],
                "fry shrimp in fryer*": ["chop shrimp"],
                "add shrimp to dish": ["fry shrimp in fryer*", "wash dish"]
            }
        ]
    }
    
    # A sample plan that now must supply separate occurrences for shared subtasks.
    plan = [
        "0: pick rice",
            "1: wait",
            "2: cook rice in pot",
            "3: wash dish",
            "4: wait",
            "5: wait",
            "6: add rice to dish",
            "7: wait",
            "8: pick beef",
            "9: wait",
            "10: chop beef",
            "11: wait",
            "12: wait",
            "13: fry beef in fryer",
            "14: wash dish",
            "15: wait",
            "16: wait",
            "17: wait",
            "18: add beef to dish",
            "19: wait",
            "20: pick noodle",
            "21: cook noodle in pot",
            "22: wait",
            "23: wait",
            "24: wait",
            "25: wait",
            "26: add noodle to dish",
            "27: wait",
            "28: pick mushroom",
            "29: wait",
            "30: chop mushroom",
            "31: wait",
            "32: wait",
            "33: fry mushroom in fryer",
            "34: wait",
            "35: add mushroom to dish",
            "36: wait",
            "37: pick shrimp",
            "38: chop shrimp",
            "39: wait",
            "40: fry shrimp in fryer",
            "41: wait",
            "42: wait",
            "43: wait",
            "44: add shrimp to dish",
            "45: wait"        # A second "wash dish" occurrence for the second task.
    ]
    
    result = evaluate_plan(tasks, plan)
    import pprint
    pprint.pprint(result)
