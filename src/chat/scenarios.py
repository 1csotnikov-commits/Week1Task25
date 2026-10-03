"""Прогон длинных сценариев чата."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from src.chat.chat_pipeline import ChatPipeline
from src.chat.session import ChatSession


@dataclass
class ScenarioMessage:
    id: str
    role: str
    text: str


@dataclass
class Scenario:
    name: str
    description: str
    index_name: str
    messages: List[ScenarioMessage]


def load_scenario(path) -> Scenario:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    msgs = []
    for m in data.get("messages", []):
        msgs.append(ScenarioMessage(id=str(m.get("id", "")), role=m.get("role", "user"), text=m.get("text", "")))
    return Scenario(
        name=data.get("name", p.stem),
        description=data.get("description", ""),
        index_name=data.get("index_name", ""),
        messages=msgs,
    )


def list_scenarios(scenarios_dir) -> List[str]:
    d = Path(scenarios_dir)
    if not d.exists():
        return []
    return sorted(p.name for p in d.glob("*.yaml"))


def run_scenario(pipeline: ChatPipeline, scenario: Scenario, session: ChatSession, checks: dict) -> dict:
    turns = []
    for m in scenario.messages:
        if m.role != "user":
            continue
        result = pipeline.send(session, m.text)
        turns.append({
            "id": m.id,
            "user": m.text,
            "answer": result.rag_result.answer,
            "answer_status": result.rag_result.answer_status,
            "has_sources": bool(result.rag_result.sources),
            "has_citations": bool(result.rag_result.citations),
            "sources_verified": result.rag_result.sources_verified,
            "citations_verified": result.rag_result.citations_verified,
            "state_updates": result.state_updates,
        })

    report = {
        "scenario": scenario.name,
        "description": scenario.description,
        "index_name": scenario.index_name,
        "turns": turns,
        "final_task_state": session.task_state.to_dict(),
        "checks": {},
    }
    if checks.get("check_goal"):
        report["checks"]["goal"] = bool(session.task_state.goal)
    if checks.get("check_sources"):
        passed = sum(1 for t in turns if t["answer_status"] == "unknown" or t["has_sources"])
        report["checks"]["sources"] = {"passed": passed, "total": len(turns)}
    if checks.get("check_citations"):
        passed = sum(1 for t in turns if t["answer_status"] == "unknown" or t["has_citations"])
        report["checks"]["citations"] = {"passed": passed, "total": len(turns)}
    return report


def save_scenario_results(report: dict, results_dir):
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = report["scenario"]
    json_path = d / f"{name}_{ts}.json"
    md_path = d / f"{name}_{ts}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_scenario_report(report), encoding="utf-8")
    return json_path, md_path


def render_scenario_report(report: dict) -> str:
    lines = [f"# Отчёт сценария: {report['scenario']}", ""]
    lines.append(f"- Описание: {report.get('description') or '-'}")
    lines.append(f"- Индекс: {report.get('index_name') or '-'}")
    lines.append("")
    lines.append("## Проверки")
    for key, value in report.get("checks", {}).items():
        if isinstance(value, dict):
            lines.append(f"- {key}: {value.get('passed')}/{value.get('total')}")
        else:
            lines.append(f"- {key}: {'OK' if value else 'FAIL'}")
    lines.append("")
    lines.append("## Ходы")
    for t in report["turns"]:
        lines.append(f"### {t['id']}")
        lines.append(f"**Вопрос:** {t['user']}")
        lines.append(f"**Статус:** {t['answer_status']}")
        lines.append(f"**Ответ:** {t['answer']}")
        lines.append(f"Источники: {'есть' if t['has_sources'] else 'нет'}, цитаты: {'есть' if t['has_citations'] else 'нет'}")
        lines.append("")
    return "\n".join(lines)
