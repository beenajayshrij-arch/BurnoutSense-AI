"""
Deterministic action mapping for burnout risk.

This module contains NO machine learning and NO LLM calls. It only maps
a feature name (the top SHAP-driving feature for an employee) to a fixed,
predefined HR action. This logic runs independently of any LLM output and
depends only on SHAP results.
"""

action_map = {
    "Mental Fatigue Score": "Recommend workload reduction and time off",
    "Resource Allocation": "Redistribute tasks across team",
    "Designation": "Schedule 1:1 check-in with manager",
    "WFH Setup Available": "Review remote work support",
    "Days_of_Service": "Check in on long-term engagement and career growth",
    "Gender": "No specific action — review other contributing factors",
    "Company Type": "No specific action — review other contributing factors",
}


def get_action(top_feature):
    """
    Return the recommended HR action for a given top contributing feature.
    Falls back to a general wellbeing check-in if the feature isn't in the map.
    """
    return action_map.get(top_feature, "General wellbeing check-in recommended")


def get_combined_action(top_feature, second_feature):
    """
    Combine the primary and secondary feature's recommended actions for a
    richer, more varied action string per employee. Still fully deterministic —
    no ML, no LLM. Two employees only get an identical action string if BOTH
    their top and second features match.
    """
    primary = get_action(top_feature)
    if second_feature and second_feature != top_feature:
        secondary = get_action(second_feature)
        if secondary != primary:
            return f"{primary} Additionally, {secondary[0].lower()}{secondary[1:]}"
    return primary


if __name__ == "__main__":
    print("[action_mapping] Testing get_action() against all known features:")
    for feature in list(action_map.keys()) + ["Unknown Feature"]:
        print(f"  {feature!r} -> {get_action(feature)!r}")

    print("\n[action_mapping] Testing get_combined_action():")
    print(f"  ('Mental Fatigue Score', 'Resource Allocation') -> "
          f"{get_combined_action('Mental Fatigue Score', 'Resource Allocation')!r}")
    print(f"  ('Designation', 'Designation') -> "
          f"{get_combined_action('Designation', 'Designation')!r}")