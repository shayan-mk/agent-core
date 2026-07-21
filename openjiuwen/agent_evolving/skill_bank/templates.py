# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Prompt templates for skill-bank candidate creation."""

SKILL_BANK_CREATOR_PROMPT = {
    "en": (
        "You maintain an agent's skill bank (each skill is a directory with a SKILL.md guide).\n"
        "Current bank version: {bank_version}\n"
        "Current skills:\n{current_skills}\n\n"
        "Creator diagnosis:\n{analysis}\n\n"
        "Prior proposal outcomes:\n{proposal_history}\n\n"
        "Successful trajectories:\n{successes}\n\n"
        "Failed trajectories:\n{failures}\n\n"
        "Trigger validation feedback from a previous authoring attempt:\n{trigger_feedback}\n\n"
        "Diagnose the failures. If a repository-level skill change would raise the success rate,\n"
        "reply with JSON only:\n"
        '{{"assertion": "<one testable claim about why this change helps>",\n'
        '  "operations": [{{"action": "add|modify|delete", "skill_name": "<name>",\n'
        '                   "skill_md": "<full SKILL.md content for add/modify>",\n'
        '                   "files": {{"<relative path>": "<file content>"}}}}]}}\n'
        "Rules: at most {max_operations} operations; modify/delete only listed skills; add only new names.\n"
        "Every add/modify SKILL.md must declare scope (general or task), when_to_use, trigger_type\n"
        "(general, beginning, or action_pattern), and trigger_pattern when action_pattern is used.\n"
        "Ground guidance in the diagnosis and trajectory actions; omit files when SKILL.md alone suffices.\n"
        "Reply {{\"assertion\": \"\", \"operations\": []}} if no change is warranted."
    ),
    "cn": (
        "你负责维护智能体的技能库（每个技能是包含 SKILL.md 的目录）。\n"
        "当前技能库版本：{bank_version}\n当前技能：\n{current_skills}\n\n"
        "创建诊断：\n{analysis}\n\n历史提案结果：\n{proposal_history}\n\n"
        "成功轨迹：\n{successes}\n\n失败轨迹：\n{failures}\n\n"
        "上一次编写的触发器校验反馈：\n{trigger_feedback}\n\n"
        "请诊断失败原因。若仓库级技能变更能提升成功率，仅回复 JSON：\n"
        '{{"assertion": "<该变更为何有效的可检验断言>",\n'
        '  "operations": [{{"action": "add|modify|delete", "skill_name": "<名称>",\n'
        '                   "skill_md": "<add/modify 的完整 SKILL.md 内容>",\n'
        '                   "files": {{"<相对路径>": "<文件内容>"}}}}]}}\n'
        "规则：至多 {max_operations} 个操作；modify/delete 仅限已列出的技能；add 仅限新名称。\n"
        "每个 add/modify 的 SKILL.md 必须声明 scope（general/task）、when_to_use、trigger_type\n"
        "（general/beginning/action_pattern），action_pattern 还必须声明 trigger_pattern。\n"
        '指导内容必须基于诊断和轨迹动作；仅需 SKILL.md 时省略 "files"。若无需变更，回复 '
        '{{"assertion": "", "operations": []}}。'
    ),
}

SKILL_BANK_CONTRASTIVE_PROMPT = {
    "en": (
        "Compare these episodes from one task. Contrast success versus failure and bank versions.\n"
        "Return JSON only with insight, failure_mode, failure_point, success_pattern, skill_impact, and confidence.\n"
        "Episodes:\n{episodes}"
    ),
    "cn": (
        "比较同一任务的这些轨迹，分析成功与失败以及不同技能库版本的差异。\n"
        "仅返回 JSON，字段为 insight、failure_mode、failure_point、success_pattern、skill_impact、confidence。\n"
        "轨迹：\n{episodes}"
    ),
}

SKILL_BANK_DIAGNOSER_PROMPT = {
    "en": (
        "Maintain rule-based assertions for an RL agent and summarize its failure profile.\n"
        "Current assertion pass rates:\n{assertion_grades}\n\nContrastive insights:\n{insights}\n\n"
        "Current skills:\n{current_skills}\n\n"
        "Return JSON only: {{\"assertion_operations\": [{{\"op\": \"add|modify|delete\", "
        "\"assertion\": {{\"name\": \"...\", \"kind\": \"action_matches|action_absent|action_order|"
        "max_action_repeats|max_steps|observation_matches\", \"pattern\": \"regex\", "
        "\"second_pattern\": \"regex\", \"limit\": 0}}}}], \"diagnosis\": \"...\", "
        "\"insight_groups\": [{{\"label\": \"...\", \"insight_indices\": [0]}}]}}.\n"
        "Only use fields required by the selected assertion kind."
    ),
    "cn": (
        "维护用于强化学习智能体的规则断言，并总结失败画像。\n"
        "当前断言通过率：\n{assertion_grades}\n\n对比分析：\n{insights}\n\n当前技能：\n{current_skills}\n\n"
        "仅返回 JSON：{{\"assertion_operations\": [{{\"op\": \"add|modify|delete\", "
        "\"assertion\": {{\"name\": \"...\", \"kind\": \"action_matches|action_absent|action_order|"
        "max_action_repeats|max_steps|observation_matches\", \"pattern\": \"正则\", "
        "\"second_pattern\": \"正则\", \"limit\": 0}}}}], \"diagnosis\": \"...\", "
        "\"insight_groups\": [{{\"label\": \"...\", \"insight_indices\": [0]}}]}}。\n"
        "仅使用所选断言类型需要的字段。"
    ),
}

__all__ = [
    "SKILL_BANK_CONTRASTIVE_PROMPT",
    "SKILL_BANK_CREATOR_PROMPT",
    "SKILL_BANK_DIAGNOSER_PROMPT",
]
