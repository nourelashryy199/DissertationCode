# ============================================================
# strategy_functions.py — Phase 1 (HPC)
# All 13 prompting strategy template functions, ported directly
# from the Phase 0 Colab notebooks (03_Build_Prompts.ipynb /
# 04_Run_Experiments.ipynb) into an importable module.
# ============================================================

import config


CATEGORY_DOMAIN = {
    "issue-spotting": "general civil litigation",
    "rule-recall": "consumer protection and regulatory law",
    "rule-application": "trademark and intellectual property law",
    "rule-conclusion": "contract law",
    "interpretation": "commercial contract interpretation",
    "rhetorical-understanding": "legal reasoning and argumentation",
}

INSTRUCTION_REPHRASINGS = {
    "zero_shot": [
        "Classify the following based on the labels provided.",
        "Determine the correct classification for the following, using the given labels.",
        "Read the following and assign the appropriate label from the list provided.",
    ],
    "role_based": [
        "You are an experienced attorney specializing in {domain}. Classify the following based on the labels provided.",
        "As a practicing lawyer with expertise in {domain}, determine the correct classification for the following.",
        "You are a legal professional focused on {domain}. Read the following and assign the appropriate label.",
    ],
    "cot": [
        "Classify the following based on the labels provided. Let's think step by step.",
        "Determine the correct classification for the following. Work through your reasoning step by step before answering.",
        "Read the following and assign the appropriate label. Reason through this carefully, step by step.",
    ],
    "structured": [
        "Using the jurisdiction, practice area, facts, and constraints below, classify the following.",
        "Given the structured case details below, determine the correct classification.",
        "Based on the labeled case information below, assign the appropriate label.",
    ],
}


def build_label_line(task):
    return f"Labels: {', '.join(task.label_options)}"


def build_context_question_block(task):
    return f"Context: {task.context}\nQuestion: {task.question}"


def build_final_answer_instruction():
    return config.FINAL_ANSWER_INSTRUCTION


def build_demo_block(demos, k):
    return "\n\n".join(d.render() for d in demos[:k])


def prompt_zero_shot(task, rephrasing_id, demos=None):
    instruction = INSTRUCTION_REPHRASINGS["zero_shot"][rephrasing_id]
    return f"{instruction}\n{build_label_line(task)}\n{build_context_question_block(task)}\n{build_final_answer_instruction()}"


def prompt_n_shot(task, rephrasing_id, demos, k):
    instruction = INSTRUCTION_REPHRASINGS["zero_shot"][rephrasing_id]
    demo_block = build_demo_block(demos, k)
    return f"{instruction}\n{build_label_line(task)}\n\n{demo_block}\n\n{build_context_question_block(task)}\n{build_final_answer_instruction()}"


def prompt_one_shot(task, rephrasing_id, demos):
    return prompt_n_shot(task, rephrasing_id, demos, k=1)


def prompt_few_shot_2(task, rephrasing_id, demos):
    return prompt_n_shot(task, rephrasing_id, demos, k=2)


def prompt_few_shot_3(task, rephrasing_id, demos):
    return prompt_n_shot(task, rephrasing_id, demos, k=min(3, len(demos)))


def prompt_role_based(task, rephrasing_id, demos=None):
    domain = CATEGORY_DOMAIN.get(task.task_type, "law")
    instruction = INSTRUCTION_REPHRASINGS["role_based"][rephrasing_id].format(domain=domain)
    return f"{instruction}\n{build_label_line(task)}\n{build_context_question_block(task)}\n{build_final_answer_instruction()}"


def prompt_structured(task, rephrasing_id, demos=None):
    instruction = INSTRUCTION_REPHRASINGS["structured"][rephrasing_id]
    return (f"Jurisdiction: {task.jurisdiction}\nPractice Area: {CATEGORY_DOMAIN.get(task.task_type, 'law')}\n"
            f"Relevant Facts: {task.context}\nConstraints: Choose exactly one of: {', '.join(task.label_options)}\n\n"
            f"{instruction}\nQuestion: {task.question}\n{build_final_answer_instruction()}")


def prompt_cot(task, rephrasing_id, demos=None):
    instruction = INSTRUCTION_REPHRASINGS["cot"][rephrasing_id]
    return f"{instruction}\n{build_label_line(task)}\n{build_context_question_block(task)}\n{build_final_answer_instruction()}"


def prompt_legal_framework(task, rephrasing_id, demos, framework_name):
    steps = config.FRAMEWORK_STEPS[framework_name]
    steps_text = "; ".join(f"({i+1}) {s}" for i, s in enumerate(steps))
    instruction_variants = [
        f"Work through the following steps before answering: {steps_text}.",
        f"Analyze this using the {framework_name.upper()} method, addressing each part in order: {steps_text}.",
        f"Structure your reasoning according to these steps: {steps_text}.",
    ]
    instruction = instruction_variants[rephrasing_id]
    return f"{instruction}\n{build_label_line(task)}\n{build_context_question_block(task)}\n{build_final_answer_instruction()}"


STRATEGY_FUNCTIONS = {
    "zero_shot": prompt_zero_shot,
    "one_shot": prompt_one_shot,
    "few_shot_2": prompt_few_shot_2,
    "few_shot_3": prompt_few_shot_3,
    "role_based": prompt_role_based,
    "structured": prompt_structured,
    "cot": prompt_cot,
    "irac": lambda t, r, d: prompt_legal_framework(t, r, d, "irac"),
    "crac": lambda t, r, d: prompt_legal_framework(t, r, d, "crac"),
    "creac": lambda t, r, d: prompt_legal_framework(t, r, d, "creac"),
    "cleo": lambda t, r, d: prompt_legal_framework(t, r, d, "cleo"),
    "treacc": lambda t, r, d: prompt_legal_framework(t, r, d, "treacc"),
    "ireac": lambda t, r, d: prompt_legal_framework(t, r, d, "ireac"),
}

assert set(STRATEGY_FUNCTIONS.keys()) == set(config.ALL_STRATEGIES), \
    "Mismatch between STRATEGY_FUNCTIONS and config.ALL_STRATEGIES!"


def build_prompt(task, strategy, rephrasing_id, task_id, demonstration_sets):
    fn = STRATEGY_FUNCTIONS[strategy]
    demos = demonstration_sets.get(task_id, [])
    return fn(task, rephrasing_id, demos)