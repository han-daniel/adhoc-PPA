from dataclasses import dataclass
from typing import List, Dict, Set
from collections import defaultdict

# ----------------------------
# Data model definitions
# ----------------------------
@dataclass
class Agent:
    id: str
    constraints: List[str]  # e.g., ["can't_attach"]

@dataclass
class Action:
    id: str
    type: str  # "fetch", "pick", "place", or "attach"
    objects: List[str]     # objects involved in the action
    description: str       # human-readable description
    required_agents: dict  # {"min": int, "max": int}
    depends_on: List[str]  # IDs of actions this action depends on
    transfers_objects_to: List[str]  # (not used in this example)

@dataclass
class ActionAssignment:
    action_id: str
    assigned_agents: List[Agent]
    stage: int
    description: str

@dataclass
class PlanningResult:
    stages: Dict[int, List[ActionAssignment]]
    action_dependencies: Dict[str, List[str]]

# ----------------------------
# Dependency validation
# ----------------------------
def validate_dependencies(actions: List[Action]) -> None:
    action_ids = {a.id for a in actions}
    for a in actions:
        for dep in a.depends_on:
            if dep not in action_ids:
                raise ValueError(f"Action {a.id} depends on non-existent action {dep}")
    def has_cycle(aid: str, visited: Set[str], path: Set[str]) -> bool:
        if aid in path:
            return True
        if aid in visited:
            return False
        visited.add(aid)
        path.add(aid)
        a = next(x for x in actions if x.id == aid)
        for dep in a.depends_on:
            if has_cycle(dep, visited, path):
                return True
        path.remove(aid)
        return False
    visited = set()
    for a in actions:
        if a.id not in visited:
            if has_cycle(a.id, visited, set()):
                raise ValueError(f"Cycle detected starting at {a.id}")

# ----------------------------
# Global tracking dictionaries
# ----------------------------
# For actions like fetch and attach we record the agent who last handled a given object.
recent_handler: Dict[str, Agent] = {}

# ----------------------------
# Helper functions
# ----------------------------
def get_free_agents(stage: int, stage_assignments: Dict[int, List[ActionAssignment]], available_agents: List[Agent]) -> List[Agent]:
    # In each stage, only agents not already assigned to an action are available.
    used_ids = set()
    for assignment in stage_assignments.get(stage, []):
        for agent in assignment.assigned_agents:
            used_ids.add(agent.id)
    return [agent for agent in available_agents if agent.id not in used_ids]

def can_do_action(agent: Agent, action_type: str) -> bool:
    return f"can't_{action_type}" not in agent.constraints

def update_object_tracking(action: Action, assigned: List[Agent]):
    # For a fetch, record the agent who fetched the object; note that fetch frees the agent.
    if action.type == "fetch":
        for obj in action.objects:
            recent_handler[obj] = assigned[0]  # assume required_agents min = 1
    # For an attach, record that the agent performed the attachment.
    elif action.type == "attach":
        for obj in action.objects:
            recent_handler[obj] = assigned[0]

# ----------------------------
# Scheduling algorithm
# ----------------------------
def plan_parallel_actions(actions: List[Action], available_agents: List[Agent]) -> PlanningResult:
    validate_dependencies(actions)
    
    # Build reverse dependency graph and count dependents.
    reverse_dependencies: Dict[str, List[Action]] = defaultdict(list)
    for a in actions:
        for dep in a.depends_on:
            reverse_dependencies[dep].append(a)
    dependency_count = {a.id: len(reverse_dependencies[a.id]) for a in actions}
    
    assigned_actions: Dict[str, int] = {}  # action id -> stage number
    stage_assignments: Dict[int, List[ActionAssignment]] = {}
    unassigned_actions: Dict[str, Action] = {a.id: a for a in actions}
    
    def schedule_task(task: Action, desired_stage: int):
        stage = desired_stage
        while True:
            free_agents = get_free_agents(stage, stage_assignments, available_agents)
            free_agents = [agent for agent in free_agents if can_do_action(agent, task.type)]
            required_count = task.required_agents["min"]
            
            # Look for preferred candidates: agents who have recently handled one of the objects.
            preferred = []
            for obj in task.objects:
                if obj in recent_handler:
                    agent = recent_handler[obj]
                    if agent in free_agents and agent not in preferred:
                        preferred.append(agent)
            if len(preferred) >= required_count:
                assigned = preferred[:required_count]
            else:
                assigned = preferred[:]
                # Fill with other available agents if needed.
                others = [agent for agent in free_agents if agent not in assigned]
                if len(assigned) + len(others) >= required_count:
                    assigned.extend(others[:required_count - len(assigned)])
                else:
                    stage += 1
                    continue  # try next stage because not enough free agents
            # Assign exactly the minimum number of agents.
            assignment = ActionAssignment(
                action_id=task.id,
                assigned_agents=assigned,
                stage=stage,
                description=task.description
            )
            stage_assignments.setdefault(stage, []).append(assignment)
            assigned_actions[task.id] = stage
            if task.id in unassigned_actions:
                del unassigned_actions[task.id]
            update_object_tracking(task, assigned)
            break
        
        # Recursively schedule child tasks.
        children = reverse_dependencies.get(task.id, [])
        children_sorted = sorted(children, key=lambda a: dependency_count[a.id], reverse=True)
        for child in children_sorted:
            if child.id in unassigned_actions and all(dep in assigned_actions for dep in child.depends_on):
                schedule_task(child, stage + 1)
    
    # First, schedule all head tasks that are of type "fetch".
    root_tasks = [a for a in actions if not a.depends_on and a.type == "fetch"]
    root_tasks_sorted = sorted(root_tasks, key=lambda a: dependency_count[a.id], reverse=True)
    for task in root_tasks_sorted:
        if task.id in unassigned_actions:
            schedule_task(task, 1)
    
    # Then schedule any remaining tasks.
    while unassigned_actions:
        available = [a for a in unassigned_actions.values() if all(dep in assigned_actions for dep in a.depends_on)]
        if not available:
            raise Exception("Unable to schedule tasks: " + str(list(unassigned_actions.keys())))
        available_sorted = sorted(available, key=lambda a: dependency_count[a.id], reverse=True)
        for task in available_sorted:
            if task.id in unassigned_actions:
                desired = (max(assigned_actions[dep] for dep in task.depends_on) + 1) if task.depends_on else 1
                schedule_task(task, desired)
    return PlanningResult(
        stages=stage_assignments,
        action_dependencies={a.id: a.depends_on for a in actions}
    )

def create_dag_visualization(result: PlanningResult) -> str:
    out = ""
    max_stage = max(result.stages.keys()) if result.stages else 0
    for stage in range(1, max_stage + 1):
        out += f"Stage {stage}:\n"
        for assignment in result.stages.get(stage, []):
            agents_str = "[" + ",".join(a.id for a in assignment.assigned_agents) + "]"
            out += f"    {assignment.action_id} {agents_str} - {assignment.description}\n"
            deps = result.action_dependencies.get(assignment.action_id, [])
            if deps:
                out += f"         depends on: {', '.join(deps)}\n"
        out += "\n"
    return out

# ----------------------------
# Example: The Chair Assembly
#
# Head nodes use "fetch" so that objects are brought into the assembly area.
# Subsequent "attach" tasks use only the minimum number of agents (1 in this example)
# and prefer agents who previously handled the needed parts.
# ----------------------------
if __name__ == "__main__":
    agents = [
        Agent(id="A", constraints=[]),
        Agent(id="B", constraints=[]),
        Agent(id="C", constraints=[])
    ]
    
    actions = [
        Action(id="1", type="fetch", objects=["chair_back"], description="Fetch chair back",
               required_agents={"min": 1, "max": 1}, depends_on=[], transfers_objects_to=[]),
        Action(id="2", type="fetch", objects=["chair_seat"], description="Fetch chair seat",
               required_agents={"min": 1, "max": 1}, depends_on=[], transfers_objects_to=[]),
        Action(id="3", type="fetch", objects=["leg1"], description="Fetch leg 1",
               required_agents={"min": 1, "max": 1}, depends_on=[], transfers_objects_to=[]),
        Action(id="4", type="fetch", objects=["leg2"], description="Fetch leg 2",
               required_agents={"min": 1, "max": 1}, depends_on=[], transfers_objects_to=[]),
        Action(id="5", type="fetch", objects=["leg3"], description="Fetch leg 3",
               required_agents={"min": 1, "max": 1}, depends_on=[], transfers_objects_to=[]),
        Action(id="6", type="fetch", objects=["leg4"], description="Fetch leg 4",
               required_agents={"min": 1, "max": 1}, depends_on=[], transfers_objects_to=[]),
        Action(id="7", type="attach", objects=["chair_back", "chair_seat"],
               description="Attach chair back and chair seat together",
               required_agents={"min": 1, "max": 1}, depends_on=["1", "2"], transfers_objects_to=[]),
        Action(id="8", type="attach", objects=["leg1", "chair_seat"],
               description="Attach leg 1 to chair seat",
               required_agents={"min": 1, "max": 1}, depends_on=["3", "7"], transfers_objects_to=[]),
        Action(id="9", type="attach", objects=["leg2", "chair_seat"],
               description="Attach leg 2 to chair seat",
               required_agents={"min": 1, "max": 1}, depends_on=["4", "7"], transfers_objects_to=[]),
        Action(id="10", type="attach", objects=["leg3", "chair_seat"],
               description="Attach leg 3 to chair seat",
               required_agents={"min": 1, "max": 1}, depends_on=["5", "7"], transfers_objects_to=[]),
        Action(id="11", type="attach", objects=["leg4", "chair_seat"],
               description="Attach leg 4 to chair seat",
               required_agents={"min": 1, "max": 1}, depends_on=["6", "7"], transfers_objects_to=[])
    ]
    
    result = plan_parallel_actions(actions, agents)
    print(create_dag_visualization(result))
