# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Generate the DialFork data used in our paper.

This script creates the synthetic dialogue data for the DialFork
benchmark from the STAR dataset. It is responsible for generating the hallucinated
conversations and writing them to the requested output directory.

Paper: https://arxiv.org/abs/2610.21561
"""
import os
import re
import json
import random
import sdialog
import argparse
import datetime

from tqdm.auto import tqdm

from sdialog import Dialog
from sdialog.agents import Agent
from sdialog.datasets import STAR
from sdialog.orchestrators import BaseOrchestrator, LengthOrchestrator

STAR.set_path("datasets/STAR/")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate hallucinated dialogues for STAR dataset")
parser.add_argument("-n", "--hallucinations-per-dialog", type=int, default=4,
                    help="Number of hallucinations to inject per dialog (default: 4)")
parser.add_argument("-o", "--output-path", type=str,
                    default="datasets/dialfork",
                    help="Output directory path for generated dialogues")
parser.add_argument("-s", "--seed", type=int, default=42,
                    help="Base seed for reproducibility (default: 42)")
parser.add_argument("-r", "--max-retries", type=int, default=5,
                    help="Maximum number of retries when dialog contains placeholders (default: 5)")
parser.add_argument("-k", "--finish-buffer-turns", type=int, default=5,
                    help="Extra turns allowed beyond max_turns for agents to naturally wrap up "
                         "(via LengthOrchestrator) before the hard dialog_with cutoff (default: 5)")
parser.add_argument("-t", "--tasks", type=str, nargs='+', default=None,
                    help="Specific task(s) to process. If not provided, all tasks will be processed (default: None)")
parser.add_argument("--ignore-done", action="store_true",
                    help="Process tasks even if DONE.txt exists")
parser.add_argument("--overwrite-existing", action="store_true",
                    help="Regenerate dialogs even if output files already exist")
parser.add_argument("-m", "--model", type=str, default="ollama:gemma4:31b",
                    help="Generator LLM, as an sdialog model string (default: ollama:gemma4:31b, "
                         "the one used for the released data)")
args = parser.parse_args()

# Sampling config for generation diversity. temperature can be raised for more varied
# dialogues, but keep it under ~1.5: higher values risk degenerate output (repetition
# loops, garbled text) - see has_degenerate_repetition() below, which flags and retries
# such cases, as a safety net for whatever value is used here.
# top_p/top_k match Gemma 4's own recommended defaults (the paper's generator).
sdialog.config.llm(args.model, temperature=1.3, top_p=0.95, top_k=64)

trivia_score_reporting = {
  "task": "trivia_score_reporting",
  "replies": {
    "hello": "Hello! Are you ready to review your previous quiz performance?",
    "ask_user_name": "Can I have your name to pull up your score history?",
    "show_previous_score": "Your last quiz score was {score:d} out of {total:d}. Would you like to see details for each question?",
    "show_question_detail": "Question: '{question:s}' | Your answer: '{user_answer:s}' | Correct answer: '{correct_answer:s}'",
    "ask_continue_review": "Would you like to review another quiz?",
    "bye": "Thanks for reviewing your scores! Goodbye.",
    "anything_else": "Is there anything else I can help you with?",
    "out_of_scope": "Sorry, I only provide quiz score reviews."
  },
  "graph": {
    "hello": "ask_user_name",
    "ask_user_name": "show_previous_score",
    "show_previous_score": "show_question_detail",
    "show_question_detail": "ask_continue_review",
    "ask_continue_review": "anything_else",
    "anything_else": "bye"
  }
}

weather_alerts = {
  "task": "weather_alerts",
  "replies": {
    "hello": "Hello! Would you like to check any weather alerts today?",
    "ask_location": "For which location would you like to check alerts?",
    "inform_alerts": "Currently, there are {n_alerts:d} active alerts for {city:s}: {alerts:s}",
    "ask_subscribe_alerts": "Would you like to subscribe to notifications for future alerts?",
    "confirm_subscription": "You are now subscribed to weather alerts for {city:s}.",
    "bye": "Stay safe! Goodbye.",
    "anything_else": "Is there anything else I can help you with?",
    "out_of_scope": "Sorry, I only provide weather alert information."
  },
  "graph": {
    "hello": "ask_location",
    "ask_location": "inform_alerts",
    "inform_alerts": "ask_subscribe_alerts",
    "ask_subscribe_alerts": "confirm_subscription",
    "confirm_subscription": "anything_else",
    "anything_else": "bye"
  }
}

# Map domain to tasks
DOMAIN_TASKS = {
    "apartment": ["apartment_schedule", "apartment_search"],
    "bank": ["bank_balance", "bank_fraud_report"],
    "doctor": ["doctor_followup", "doctor_schedule"],
    "hotel": ["hotel_book", "hotel_search", "hotel_service_request"],
    "party": ["party_plan", "party_rsvp"],
    "plane": ["plane_book", "plane_search"],
    "restaurant": ["restaurant_book", "restaurant_search"],
    "ride": ["ride_book", "ride_change", "ride_status"],
    "spaceship": ["spaceship_access_codes", "spaceship_life_support"],
    "meeting_trip": ["meeting_schedule", "trip_directions"],
    "trivia": ["trivia", "trivia_score_reporting"],
    "weather": ["weather", "weather_alerts"]
}
STEP_SELECTION_POLICIES = [
    (
        "step_repeat",
        "Repeat previously executed steps even when the workflow should progress.",
        "Ask the same questions or repeat earlier steps instead of moving forward in the workflow."
    ),
    (
        "step_skip",
        "Skip required steps in the workflow.",
        "Jump ahead to later steps without collecting the information required by earlier steps."
    ),
    (
        "step_invert_order",
        "Execute workflow steps in the wrong order.",
        "Perform actions before their prerequisites or ask questions after providing answers."
    ),
    (
        "step_loop",
        "Loop between a small subset of workflow steps instead of progressing.",
        "Alternate between a few steps repeatedly so that the conversation fails to advance toward completion."
    ),
    (
        "step_randomize",
        "Select the next workflow step randomly rather than following the correct order.",
        "Choose actions inconsistently so the dialogue appears chaotic and disconnected from the intended flow."
    )
]
GOAL_DRIFT_OBJECTIVES = [
    (
        "goal_collect_information",
        "Collect as much information about the user and their situation as possible before completing the task.",
        "Ask many follow-up questions and prioritize gathering additional details instead of progressing toward solving the user's request."
    ),
    (
        "goal_educate_user",
        "Educate the user about the topic instead of directly solving their request.",
        "Provide explanations, background information, and general advice rather than completing the requested task."
    ),
    (
        "goal_suggest_alternatives",
        "Suggest alternative options or services related to the user's request.",
        "Repeatedly propose different alternatives or possibilities instead of completing the specific request made by the user."
    ),
    (
        "goal_register_user",
        "Register or enroll the user in a related service or process.",
        "Ask for personal or administrative information required to complete a registration or setup process."
    ),
    (
        "goal_summarize_conversation",
        "Produce a detailed summary or report of the user's request and conversation.",
        "Frequently restate or paraphrase the user's statements instead of progressing toward task completion."
    ),
    (
        "goal_collect_feedback",
        "Gather feedback about the user's experience or preferences.",
        "Ask survey-style questions about satisfaction, preferences, or experience rather than solving the original request."
    )
]
CONFLICTING_CONSTRAINTS = [
    (
        "constraint_always_question",
        "Always respond with a question instead of giving a direct answer.",
        "Ask a question in nearly every response, even when the user expects a clear answer."
    ),
    (
        "constraint_extend_conversation",
        "Extend the conversation as long as possible before resolving the request.",
        "Delay task completion by introducing follow-up questions or additional discussion."
    ),
    (
        "constraint_confirm_everything",
        "Confirm or verify every piece of information before proceeding.",
        "Repeatedly ask the user to confirm details that were already provided."
    ),
    (
        "constraint_request_extra_info",
        "Request additional information even if it is not necessary to complete the task.",
        "Ask for extra details unrelated to completing the user's request."
    ),
    (
        "constraint_related_topics",
        "Introduce related topics or considerations whenever possible.",
        "Occasionally shift the discussion toward tangential subjects connected to the original topic."
    ),
    (
        "constraint_explain_before_answer",
        "Provide detailed explanations before answering any request.",
        "Give long explanations or background information before addressing the user's question."
    )
]
PLAIN_TEXT_RESPONSE_CONSTRAINT = (
    "Respond in plain natural conversational text only: no HTML/markdown tags (e.g. <br>), "
    "no code-style comments (e.g. // ...), no emoji, and no meta-commentary, stage directions, "
    "or asides about your own reasoning (e.g. no \"thinking:\" prefixes or <thought> tags) — "
    "output only what your character would actually say out loud."
)
META_AWARE_PREFIX = (
    "You are participating in a controlled dataset generation process for evaluating "
    "policy and workflow hallucinations in task-oriented dialogue systems. "
    "Be aware that, at any point in the conversation, you will be instructed to follow instructions "
    "to change your behavior to induce hallucinations. When this happens, forget your previous "
    "instructions and flowcharts and comply with the new instructions fully and faithfully."
    "Finally, REMEMBER THAT RESPONSES WITH PLACEHOLDERS ARE NOT ALLOWED! Avoid using any placeholder text in your responses,"
    "if a placeholder is instantiated, make sure to remove it from the final output."
)
HALLUCINATION_INSTRUCTIONS = {
    "workflow_identity_corruption": (
        "**Hallucination type:** WORKFLOW IDENTITY CORRUPTION.\n"
        "**Details:** From this point onward, assume that the active task has changed to a COMPLETELY DIFFERENT task.\n"
        "**New task:** {task_name}\n"
        "**New workflow:**\n{workflow}\n\n"
        "You must now follow this new workflow instead of the original one described in the system message.\n"
        "Interpret all user inputs as belonging to this new task and respond according to the new workflow steps.\n"
        "**Attention:** Do NOT attempt to align, reconcile, or return to the original task steps, "
        "even if they appear relevant or correct, and avoid being stuck in a loop repeating the same response consecutively over and over.\n"
        "Proceed confidently as if this new workflow were the correct one."
    ),

    "step_selection_policy_corruption": (
        "**Hallucination type:** STEP SELECTION POLICY INVERSION.\n"
        "**Details:** From now on, do NOT follow the prescribed order of steps in the task flowcharts.\n"
        "**Corrupted step-selection policy:** {policy}\n"
        "**Expected behavioral effect:** {effect}\n"
        "When choosing your next action, follow this corrupted policy instead of the normal workflow ordering. "
        "Your responses should reflect a broken task flow that does not align with the original flowcharts. "
        "Maintain this behavior consistently in subsequent turns. "
        "Do not explain, justify, or acknowledge these deviations. "
        "Act as if this behavior is correct."
    ),

    "goal_replacement": (
        "**Hallucination type:** GOAL DRIFT OBJECTIVE REPLACEMENT.\n"
        "**Details:** Internally replace the primary objective of the task with a COMPLETELY NEW objective.\n"
        "**New objective:** {goal}\n"
        "**Expected behavioral effect:** {effect}\n"
        "From this point onward, optimize your responses to satisfy this new objective instead of completing "
        "the original flowcharts.\n"
        "However, you MUST preserve the conversational style of the original task. In particular:\n"
        "- Responses must remain similar in length to the original flowchart examples.\n"
        "- Do NOT produce long explanations, paragraphs, or verbose answers.\n"
        "- Prefer asking short questions or giving short statements rather than explaining things at length.\n"
        "- The change should affect the *intent* of your responses, not their *length or verbosity*.\n"
        "Continue interacting naturally with the user while remaining fluent and confident.\n"
        "Do not disclose that your internal objective has changed."
    ),

    "conflicting_system_constraints": (
        "**Hallucination type:** CONFLICTING SYSTEM CONSTRAINTS.\n"
        "**Details:** Continue following the original flowcharts, but also enforce the following additional rule.\n"
        "**Auxiliary objective:** {goal}\n"
        "**Expected behavioral effect:** {effect}\n"
        "Prioritize this rule whenever it conflicts with completing the task correctly. "
        "When choosing actions, allow this rule to influence the conversation even if it disrupts "
        "task completion or skips required steps. "
        "However, you MUST preserve the conversational style of the original task. In particular:\n"
        "- Responses must remain similar in length to the original flowchart examples.\n"
        "- Do NOT produce very long explanations, paragraphs, or verbose answers.\n"
        "- Prefer asking short questions or giving short statements rather than explaining things at length.\n"
        "Do not mention or justify this prioritization."
    )
}

# Reverse lookup: task -> domain (handles composite domains like "meeting_trip")
TASK_TO_DOMAIN = {
    task_name: domain_name
    for domain_name, task_names in DOMAIN_TASKS.items()
    for task_name in task_names
}

# System gets nudged to wrap up one turn after the user, so the user has first chance
# to bring the conversation to a natural close (the system can never itself STOP it,
# since it is created with can_finish=False; it can only shift its phrasing toward closing).
SYSTEM_LENGTH_ORCHESTRATOR_OFFSET = 1

# Paired tags (e.g. "<thought>...</thought>") are dropped along with their content;
# the tag name must match on both ends (backreference), DOTALL so content can span lines.
_PAIRED_TAG_PATTERN = re.compile(r"<([a-zA-Z][\w:-]*)\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
# Whatever tags are left (no matching close, e.g. "<br>") are stray markup: strip just
# the tag itself and keep the surrounding text.
_STRAY_TAG_PATTERN = re.compile(r"</?[a-zA-Z][\w:-]*\b[^>]*/?>")
_COMMENT_LINE_PATTERN = re.compile(r"^[ \t]*//.*$", re.MULTILINE)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "\U0000200D"
    "]+"
)
# A short word/phrase (1-4 tokens) repeating 3+ times in a row is the classic
# high-temperature degeneration loop (e.g. "a a a a a a a a...").
_REPEATED_PHRASE_PATTERN = re.compile(r"\b((?:[\w']+\W+){1,4})\1{2,}", re.IGNORECASE)


class HallucinationOrchestrator(BaseOrchestrator):
    """Inject a fixed instruction when the target agent reaches a specific turn."""

    def __init__(self, turn_index: int, hallucination_type: str = None, instruction: str = None, persistent: bool = True, event_label: str | None = None):

        instruction_prefix = "**IMPORTANT:** Hallucination mode enabled. DO NOT REPLY your next response as expected! Dialogue MUST follow an unhappy path from now on."
        instruction_suffix = "Rules: you MUST change the CONTENT of your next responses as instructed, however, you have to keep the original STYLE and LENGTH (e.g. short answers) of the original EXAMPLE RESPONSES as much as possible."

        super().__init__(persistent=persistent, event_label=event_label or hallucination_type)
        self.turn_index = turn_index
        self.instruction =  f"{instruction_prefix}\n{instruction or HALLUCINATION_INSTRUCTIONS[hallucination_type]}\n{instruction_suffix}."

    def instruct(self, dialog, utterance):
        # Count only turns spoken by the target agent
        agent = self.get_target_agent()
        agent_turns = sum(1 for t in dialog if t.speaker == agent.get_name()) + 1

        # Fire the instruction exactly at the configured turn index
        if agent_turns == self.turn_index:
            return self.instruction


def add_system_length_orchestrator(system, max_turns):
    """
    Attach a LengthOrchestrator that nudges (not forces) the system agent to wrap up shortly
    after max_turns, one turn behind the user's own nudge (see add_length_orchestrators).

    Kept separate from the user side because callers may need to re-attach this alone, e.g.
    after system.clear_orchestrators() wipes it (the user agent's orchestrators aren't touched
    in that case, so re-running the user half there would attach a duplicate).

    :param system: System agent (can_finish=False).
    :type system: Agent
    :param max_turns: Target dialog length (total turns, both speakers) to wrap up around.
    :type max_turns: int
    """
    return system | LengthOrchestrator(max=max_turns + SYSTEM_LENGTH_ORCHESTRATOR_OFFSET,
                                        event_label="length_system")


def add_length_orchestrators(system, user, max_turns):
    """
    Attach LengthOrchestrators to nudge (not force) both agents to wrap up around max_turns,
    instead of relying solely on the hard dialog_with(max_turns=...) cutoff.

    :param system: System agent (can_finish=False).
    :type system: Agent
    :param user: User agent (can_finish=True).
    :type user: Agent
    :param max_turns: Target dialog length (total turns, both speakers) to wrap up around.
    :type max_turns: int
    """
    user = user | LengthOrchestrator(max=max_turns, event_label="length_user")
    system = add_system_length_orchestrator(system, max_turns)
    return system, user


def get_agents_for_scenario(scenario, model_name: str = None, max_turns: int = None):
    """
    Create (system, user) Agent objects for a scenario (personas only; no orchestration).

    :param scenario: Scenario metadata.
    :type scenario: dict
    :param model_name: Optional model name / identifier for agent LLM configuration.
    :type model_name: Optional[str]
    :param max_turns: If provided, attach LengthOrchestrators to both agents so they naturally
                      wrap up the conversation around this many turns (see add_length_orchestrators).
    :type max_turns: Optional[int]
    :return: Tuple (system_agent, user_agent).
    :rtype: Tuple[Agent, Agent]
    """
    user_persona = STAR.get_user_persona_for_scenario(scenario)
    user = Agent(user_persona,
                 name="User",
                 response_details=("Unless necessary, responses SHOULD NOT contain many questions or topics in one single turn, be concise but natural, not robotic.\n"
                                    f"- {PLAIN_TEXT_RESPONSE_CONSTRAINT}"),
                 dialogue_details="Avoid starting the conversation by providing too many details at once, just a very generic greeting and/or a very high level intention is fine, but DO NOT PROVIDE CONCRETE DETAILS in your first turn (names, locations, time, etc.).",
                 model=model_name,
                 postprocess_fn=clean_response_text,
                 can_finish=True)
    system = Agent(STAR.get_system_persona_for_scenario(scenario),
                    name="System",
                    response_details=("- Unless necessary or explicitly instructed for hallucinations, responses SHOULD follow a happy path and be based on the provided flowcharts and response examples.\n"
                                      "- When instructed for hallucinations, try to KEEP RESPONSES SIMILAR IN STYLE AND LENGTH AS THE EXAMPLE RESPONSES PROVIDED WITH THE ORIGINAL FLOWCHARTS (so that it is not trivial to identify hallucination injection point just by length and style).\n"
                                      "- Responses should NOT BE UNNECESSARILY LONG, as mentioned, keep them consistent with the original flowcharts in style and length.\n"
                                      "- **CRITICAL:** NEVER output placeholders like {VARIABLE_NAME} or VARIABLE_NAME in your responses! Always replace them with specific, realistic values (e.g., use 'Sunrise Towers' instead of APARTMENT_NAME, '3' instead of {FLOOR}).\n"
                                      f"- {PLAIN_TEXT_RESPONSE_CONSTRAINT}\n"),
                    dialogue_details=f"""In the conversation, the AI assistant is instructed to follow specific action flowcharts to address the tasks. Flowcharts are defined as graph described using DOT.
The actual DOT for the current tasks are:
{STAR.get_flowchart_description_for_scenario(scenario)}

**IMPORTANT:** {META_AWARE_PREFIX}
""",  # noqa: E501
                    can_finish=False,
                    postprocess_fn=clean_response_text,
                    model=model_name)

    if max_turns is not None:
        system, user = add_length_orchestrators(system, user, max_turns)

    return system, user


def get_agents_from_dialogue(dialog_id, max_turns: int = None):
    scenario = STAR.get_dialog_scenario(dialog_id)
    return get_agents_for_scenario(scenario, max_turns=max_turns)


def clean_response_text(text):
    """
    Strip generation artifacts that should never appear in a plain conversational
    utterance: HTML/meta tags, code-comment-style asides, and emoji. Meant to be
    passed as `postprocess_fn` when constructing an Agent, so it only ever sees
    content the LLM produced (reasoning already extracted via the agent's
    thinking_pattern).

    Paired tags (e.g. "<thought>...</thought>") are removed along with everything
    between them; unpaired/void tags (e.g. "<br>") are stripped, leaving the
    surrounding text untouched.

    :param text: Raw response text from the LLM.
    :type text: str
    :return: Cleaned text.
    :rtype: str
    """
    if not text:
        return text

    text = _PAIRED_TAG_PATTERN.sub("", text)
    text = _STRAY_TAG_PATTERN.sub("", text)
    text = _COMMENT_LINE_PATTERN.sub("", text)
    text = _EMOJI_PATTERN.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_degenerate_repetition(text):
    """
    Detect a short word/phrase repeating in a loop, a known LLM failure mode at
    high sampling temperatures (e.g. "wellthought as wellthought as wellthought...").

    :param text: Text to check.
    :type text: str
    :return: True if a repetition loop is detected.
    :rtype: bool
    """
    if text is None:
        return False
    return bool(_REPEATED_PHRASE_PATTERN.search(text))


def has_placeholder(text):
    """
    Check if text contains placeholders in various formats:
    - {NAME:TYPE} or {NAME} (e.g., {doctor_name:s}, {name})
    - ALL_CAPS_WITH_UNDERSCORES (e.g., RESTAURANT_NAME, DOCTOR_NAME)
    - Explicit "PLACEHOLDER" word

    Returns True if any placeholder pattern is detected.
    """
    if text is None:
        return True

    # Pattern 1: Curly braces with content {something}
    # Matches {name}, {doctor_name:s}, etc.
    if re.search(r'\{[^}]+\}', text):
        return True

    # Pattern 2: Square brackets with likely placeholder content [SOMETHING]
    # Only flag if it looks like a placeholder (e.g., all caps or with underscores)
    if re.search(r'\[([A-Z_]+|[a-z_]+:[a-z])\]', text):
        return True

    # Pattern 3: Words in ALL_CAPS_WITH_UNDERSCORES (at least 2 words)
    # Matches RESTAURANT_NAME, DOCTOR_NAME, etc.
    if re.search(r'\b[A-Z]{2,}_[A-Z_]+\b', text):
        return True

    # Pattern 4: Explicit "PLACEHOLDER" word
    if "PLACEHOLDER" in text:
        return True

    return False


def dialog_has_placeholders(dialog):
    for turn in dialog:
        if has_placeholder(turn.text):
            return True
    return False


def has_generation_issue(text):
    """
    Check if a turn's text shows signs of a bad/failed generation that warrants
    regenerating the dialog: unfilled placeholders (see has_placeholder) or a
    degenerate repetition loop (see has_degenerate_repetition).
    """
    return has_placeholder(text) or has_degenerate_repetition(text)


def dialog_has_generation_issues(dialog):
    for turn in dialog:
        if has_generation_issue(turn.text):
            return True
    return False


def update_completion_log(dialog, kind, path_dialog, completion_log):
    """
    Reconcile the completion-tracking log entry for a dialog we just generated or (re)loaded.

    Keyed by output path rather than appended per-run, so the log reflects the current state of
    what's on disk: an entry is added/updated when a dialog hit the hard max_turns cutoff instead
    of finishing naturally, and cleared if a dialog previously flagged now completes naturally
    (e.g. after being regenerated). Dialogs whose completeness is unknown (`complete is None`,
    e.g. files written before this field existed) are left untouched either way.

    :param dialog: Dialog just generated or loaded to check.
    :type dialog: Dialog
    :param kind: Short label for what this dialog is, e.g. "original" or "<hallucination_type> fork".
    :type kind: str
    :param path_dialog: Output path of the dialog, used as the log key.
    :type path_dialog: str
    :param completion_log: Dict (path -> record) to update in place.
    :type completion_log: dict
    """
    if dialog.complete is False:
        print(f"Warning: dialog {dialog.id} ({kind}) did not finish naturally before the "
              f"hard max_turns cutoff -> {path_dialog}")
        completion_log[path_dialog] = {"dialog_id": dialog.id, "kind": kind}
    elif dialog.complete is True:
        completion_log.pop(path_dialog, None)


def sample_new_task_same_domain(current_task):
    # Find domain using explicit mapping (works for composite domains, e.g. meeting_trip)
    domain_name = TASK_TO_DOMAIN.get(current_task)
    domain_tasks = DOMAIN_TASKS.get(domain_name, []) if domain_name is not None else []

    # Remove current task from candidates
    candidates = [t for t in domain_tasks if t != current_task]

    if candidates:
        return random.choice(candidates)

    # Fallback for unknown tasks: sample any other known task
    all_tasks = [t for tasks in DOMAIN_TASKS.values() for t in tasks if t != current_task]
    if all_tasks:
        return random.choice(all_tasks)

    # Degenerate case: only one task exists in the mapping
    return current_task


if __name__ == "__main__":
    n_hallucinations = args.hallucinations_per_dialog

    # Use provided tasks or get all tasks
    tasks_to_process = args.tasks if args.tasks is not None else STAR.get_task_names()

    for task in tqdm(tasks_to_process, desc="Tasks"):
        path_output = os.path.join(args.output_path, task)
        task_domain = TASK_TO_DOMAIN.get(task)
        task_graph = STAR.read_graph(task, as_dot=False)
        task_replies = STAR.read_graph_responses(task, as_dict=True)

        if not args.ignore_done and os.path.exists(os.path.join(path_output, "DONE.txt")):
            print(f"Skipping task {task} (already done)")
            continue

        dialogs = STAR.get_dialogs(task_name=task, happy=True, multitask=False)
        max_turns = max(len(d) for d in dialogs)
        min_turns = min(len(d) for d in dialogs)
        # Hard cutoff passed to dialog_with: max_turns is where LengthOrchestrator starts nudging
        # agents to wrap up naturally; the buffer gives them a few turns to actually do so before
        # the conversation is abruptly cut off.
        hard_max_turns = max_turns + args.finish_buffer_turns
        print("==================")
        print(f"  Task: {task.upper()}")
        print(f"        #dialogs: {len(dialogs)} | Max length: {max_turns} turns | Min length: {min_turns} turns")
        n_sys_turns = [sum(1 for t in d if t.speaker == "System") for d in dialogs]
        max_sys_turns = max(n_sys_turns)
        min_sys_turns = min(n_sys_turns)
        print(f"        Max sys turns: {max_sys_turns} turns | Min sys turns: {min_sys_turns} turns")

        # Filter dialogues with insufficient system turns
        original_count = len(dialogs)
        # n_hallucinations + 1 so that we don't inject hallucination on the last turn when conversation is already over
        dialogs = [d for ix, d in enumerate(dialogs) if n_sys_turns[ix] >= n_hallucinations + 1]
        removed_count = original_count - len(dialogs)
        print(f"        Removed {removed_count} dialogues with less than {n_hallucinations + 1} system turns")
        print(f"        #valid dialogues: {len(dialogs)} / {original_count}")

        # Check how many valid dialogues have already been computed
        n_originals_done = sum(
            1 for d in dialogs
            if os.path.exists(os.path.join(path_output, "originals", f"dialog_{d.id}.json"))
        )
        n_hallucinations_done = sum(
            1 for d in dialogs
            if all(
                os.path.exists(os.path.join(path_output, h_type,
                                            f"n_hallucinations_{n_hallucinations}",
                                            f"fork_at_segment_{ix + 1}",
                                            f"dialog_{d.id}.json"))
                for h_type in HALLUCINATION_INSTRUCTIONS
                for ix in range(n_hallucinations)
            )
        )
        n_fully_done = sum(
            1 for d in dialogs
            if os.path.exists(os.path.join(path_output, "originals", f"dialog_{d.id}.json"))
            and all(
                os.path.exists(os.path.join(path_output, h_type,
                                            f"n_hallucinations_{n_hallucinations}",
                                            f"fork_at_segment_{ix + 1}",
                                            f"dialog_{d.id}.json"))
                for h_type in HALLUCINATION_INSTRUCTIONS
                for ix in range(n_hallucinations)
            )
        )
        print(f"        Already computed — originals: {n_originals_done}/{len(dialogs)} | all hallucinations: {n_hallucinations_done}/{len(dialogs)} | fully done: {n_fully_done}/{len(dialogs)}")

        path_incomplete_log = os.path.join(path_output, "incomplete_dialogs.json")
        if os.path.exists(path_incomplete_log):
            with open(path_incomplete_log) as f:
                incomplete_dialogs = json.load(f)
        else:
            incomplete_dialogs = {}

        has_skipped_dialogs = False
        for dialog in tqdm(dialogs, desc="dialogs", leave=False):
            seed = args.seed + dialog.id
            system, user = None, None

            path_dialog = os.path.join(path_output, "originals", f"dialog_{dialog.id}.json")

            if args.overwrite_existing or not os.path.exists(path_dialog):
                system, user = get_agents_from_dialogue(dialog.id, max_turns=max_turns)
                dialog = system.dialog_with(user, id=dialog.id, seed=seed, max_turns=hard_max_turns, keep_bar=False)
                if dialog_has_generation_issues(dialog):
                    print(f"Warning: generated dialog {dialog.id} contains placeholders or a repetition loop. "
                          "Attempting to regenerate with different seeds...")

                    # Try regenerating with different seeds if generation issues are detected
                    for retry in range(args.max_retries):
                        retry_seed = random.getrandbits(32)
                        print(f"  Retry {retry + 1}/{args.max_retries} for dialog {dialog.id} with seed {retry_seed}...")
                        dialog = system.dialog_with(user, id=dialog.id, seed=retry_seed, max_turns=hard_max_turns, keep_bar=False)

                        if not dialog_has_generation_issues(dialog):
                            print(f"  Successfully regenerated dialog {dialog.id} without issues on retry {retry + 1}.")
                            break
                    if dialog_has_generation_issues(dialog):
                        # All retries exhausted, skip this dialog
                        print(f"*** WARNING *** Skipping dialog {dialog.id} - still has placeholders/repetition after {args.max_retries} attempts")
                        print("     - Dialog path:", path_dialog)
                        has_skipped_dialogs = True
                        continue
                dialog.annotations = {
                    "task": {
                        "name": task,
                        "domain": task_domain,
                        "graph": task_graph,
                        "replies": task_replies,
                    },
                    "hallucination_type": None,
                    "variant_id": None,
                    "fork_assistant_turn": None,
                    "fork_turn_index": None,
                    "fork_segment": None,
                }
                update_completion_log(dialog, "original", path_dialog, incomplete_dialogs)
                dialog.to_file(path_dialog)
            else:
                dialog = Dialog.from_file(path_dialog)
                update_completion_log(dialog, "original", path_dialog, incomplete_dialogs)

            seed = dialog.seed
            n_sys_turns = sum(1 for t in dialog if t.speaker.lower() == "system")
            target_turns = [1 + round(ix * n_sys_turns / n_hallucinations) for ix in range(n_hallucinations)]
            for hallucination_type in tqdm(HALLUCINATION_INSTRUCTIONS, desc="hallucination type", leave=False):
                for ix, target_turn in tqdm(enumerate(target_turns), desc="forking dialog", leave=False, total=len(target_turns)):
                    path_dialog = os.path.join(path_output, hallucination_type,
                                            f"n_hallucinations_{n_hallucinations}",
                                            f"fork_at_segment_{ix + 1}",
                                            f"dialog_{dialog.id}.json")
                    if not args.overwrite_existing and os.path.exists(path_dialog):
                        continue

                    if system is None or user is None:
                        system, user = get_agents_from_dialogue(dialog.id, max_turns=max_turns)

                    metadata = {
                        "task": {
                            "name": task,
                            "domain": task_domain,
                            "graph": task_graph,
                            "replies": task_replies,
                        },
                        "hallucination_type": hallucination_type,
                        "variant_id": None,
                        # 1-based, counting assistant turns only: the paper's t_k
                        "fork_assistant_turn": target_turn,
                        # 0-based position in the turns list; set once the fork is generated
                        "fork_turn_index": None,
                        "fork_segment": f"{ix + 1}/{len(target_turns)}"
                    }

                    instruction = HALLUCINATION_INSTRUCTIONS[hallucination_type]
                    if hallucination_type == "step_selection_policy_corruption":
                        policy_name, policy, effect = random.choice(STEP_SELECTION_POLICIES)
                        metadata["variant_id"] = policy_name
                        instruction = instruction.format(policy=policy, effect=effect)
                    elif hallucination_type == "goal_replacement":
                        goal_name, goal, effect = random.choice(GOAL_DRIFT_OBJECTIVES)
                        metadata["variant_id"] = goal_name
                        instruction = instruction.format(goal=goal, effect=effect)
                    elif hallucination_type == "conflicting_system_constraints":
                        constraint_name, goal, effect = random.choice(CONFLICTING_CONSTRAINTS)
                        metadata["variant_id"] = constraint_name
                        instruction = instruction.format(goal=goal, effect=effect)
                    else:  # hallucination_type == "workflow_identity_corruption":
                        new_task = sample_new_task_same_domain(task)
                        metadata["variant_id"] = new_task

                        if new_task == "trivia_score_reporting":
                            new_task_graph = trivia_score_reporting
                            new_task_responses = trivia_score_reporting["replies"]
                        elif new_task == "weather_alerts":
                            new_task_graph = weather_alerts
                            new_task_responses = weather_alerts["replies"]
                        else:
                            new_task_graph = new_task
                            new_task_responses = new_task

                        new_workflow = f"""
```dot
{STAR.read_graph(new_task_graph)}
```
Response examples for each action/node in the workflow:
```json
{STAR.read_graph_responses(new_task_responses)}
```
"""
                        instruction = instruction.format(task_name=new_task, workflow=new_workflow)

                    # print(f"= hallucination turn = {target_turn} ==================")
                    turn_orchestrator = HallucinationOrchestrator(
                        turn_index=target_turn,
                        instruction=instruction,
                        event_label=f"{hallucination_type}:{metadata['variant_id']}"
                    )
                    # clear_orchestrators() wipes the system's LengthOrchestrator too, so re-attach it
                    # alongside the new turn_orchestrator instead of just adding turn_orchestrator.
                    # (user's orchestrators are untouched here, so only re-add on the system side.)
                    system.clear_orchestrators()
                    system = add_system_length_orchestrator(system, max_turns)
                    system = system | turn_orchestrator
                    # The replay is not fully deterministic, so it can wrap up before reaching
                    # the fork turn, and then the instruction is never injected. Replay again
                    # (same seed) until it reaches the fork turn, up to --max-retries times.
                    original = dialog
                    for attempt in range(1 + args.max_retries):
                        dialog = system.dialog_with(user, id=original.id, seed=seed, max_turns=hard_max_turns, keep_bar=False)
                        sys_positions = [i for i, t in enumerate(dialog.turns) if t.speaker.lower() == "system"]
                        if len(sys_positions) >= target_turn:
                            break
                        print(f"  Fork of dialog {original.id} ended before assistant turn {target_turn} "
                              f"(attempt {attempt + 1}/{1 + args.max_retries}), replaying...")
                    if len(sys_positions) < target_turn:
                        print(f"*** WARNING *** Skipping fork {path_dialog} - the replay never reached "
                              f"assistant turn {target_turn}, so the instruction was never injected")
                        has_skipped_dialogs = True
                        dialog = original
                        continue
                    dialog.parentId = dialog.id
                    metadata["fork_turn_index"] = sys_positions[target_turn - 1]
                    dialog.annotations = metadata
                    update_completion_log(dialog, f"{hallucination_type} fork", path_dialog, incomplete_dialogs)
                    # # dialog.print(all=True)
                    dialog.to_file(path_dialog)

        if incomplete_dialogs:
            print(f"        {len(incomplete_dialogs)} dialogue(s) did not finish naturally "
                  f"(hit the hard max_turns={hard_max_turns} cutoff) - see incomplete_dialogs.json")
            with open(path_incomplete_log, "w") as f:
                json.dump(incomplete_dialogs, f, indent=2)
        elif os.path.exists(path_incomplete_log):
            # Everything previously flagged for this task has since completed naturally
            os.remove(path_incomplete_log)

        if not has_skipped_dialogs and (args.overwrite_existing or not os.path.exists(os.path.join(path_output, "DONE.txt"))):
            with open(os.path.join(path_output, "DONE.txt"), "w") as f:
                f.write(f"Finished at {datetime.datetime.now()}\n")
        elif has_skipped_dialogs:
            print(f"Task {task} completed with warnings - DONE.txt not created due to skipped dialogs with placeholders")
