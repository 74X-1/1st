"""AI 驱动的研发协作 Agent

功能：
1. 分析需求，生成研发计划
2. 检索本地代码仓库上下文
3. 生成文件级修改建议
4. 自动运行测试/命令
5. 根据失败结果迭代修复

依赖：
  pip install openai fastapi uvicorn pydantic

环境变量：
  export OPENAI_API_KEY=...

用法：
  # CLI
  python dev_collaboration_agent.py --repo /path/to/repo --task "为项目增加登录接口"

  # 服务
  uvicorn dev_collaboration_agent:app --reload

说明：
  这是一个可落地的最小完整实现，适合做内部研发协作助手的起点。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI


MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
MAX_CONTEXT_FILE_CHARS = int(os.getenv("MAX_CONTEXT_FILE_CHARS", "12000"))
MAX_TREE_FILES = int(os.getenv("MAX_TREE_FILES", "250"))
DEFAULT_MAX_ITERS = int(os.getenv("MAX_ITERS", "3"))

app = FastAPI(title="AI 研发协作 Agent")
client = OpenAI()


# -----------------------------
# 数据结构
# -----------------------------


@dataclass
class Edit:
    path: str
    content: str
    mode: str = "write"  # write | append


@dataclass
class AgentPlan:
    goal: str
    assumptions: List[str] = field(default_factory=list)
    files_to_check: List[str] = field(default_factory=list)
    files_to_edit: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    test_steps: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


@dataclass
class RunResult:
    plan: AgentPlan
    edits: List[Edit]
    test_output: str
    summary: str
    iterations: int


# -----------------------------
# 工具层：仓库读写与命令执行
# -----------------------------


def _repo_root(repo: str | Path) -> Path:
    root = Path(repo).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"repo 不存在或不是目录: {root}")
    return root


def _safe_join(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if root not in p.parents and p != root:
        raise ValueError(f"不允许访问仓库外文件: {rel}")
    return p


def list_repo_tree(root: Path, max_files: int = MAX_TREE_FILES) -> str:
    lines: List[str] = []
    count = 0
    for p in sorted(root.rglob("*")):
        if any(part in {".git", "node_modules", "dist", "build", ".venv", "__pycache__"} for part in p.parts):
            continue
        rel = p.relative_to(root)
        if p.is_file():
            lines.append(str(rel))
            count += 1
            if count >= max_files:
                lines.append("... (truncated)")
                break
    return "\n".join(lines)


def read_text_file(root: Path, rel: str, limit: int = MAX_CONTEXT_FILE_CHARS) -> str:
    p = _safe_join(root, rel)
    if not p.exists() or not p.is_file():
        return ""
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""
    if len(text) > limit:
        return text[:limit] + "\n\n... [TRUNCATED]"
    return text


def write_text_file(root: Path, rel: str, content: str) -> None:
    p = _safe_join(root, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def append_text_file(root: Path, rel: str, content: str) -> None:
    p = _safe_join(root, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(content)


def run_shell(command: str, cwd: Path, timeout: int = 900) -> Dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-30000:],
        "stderr": proc.stderr[-30000:],
        "command": command,
    }


# -----------------------------
# LLM 交互
# -----------------------------


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 兜底：抓取第一个 JSON 对象
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"无法解析 JSON：{text[:500]}")


def llm_text(system: str, user: str, temperature: float = 0.2) -> str:
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.output_text


def llm_json(system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
    raw = llm_text(system, user, temperature=temperature)
    return _extract_json(raw)


# -----------------------------
# Agent 核心
# -----------------------------


PLANNER_SYSTEM = """你是资深技术负责人，负责把需求拆解成可执行研发计划。
请只输出严格 JSON，不要输出多余文本。

JSON schema:
{
  "goal": "一句话目标",
  "assumptions": ["前提1", "前提2"],
  "files_to_check": ["建议先查看的文件路径"],
  "files_to_edit": ["预期要修改的文件路径"],
  "implementation_steps": ["实施步骤1", "实施步骤2"],
  "test_steps": ["验证步骤1", "验证步骤2"],
  "risks": ["风险1", "风险2"]
}
"""

IMPLEMENTER_SYSTEM = """你是高级软件工程师，负责基于仓库上下文实现需求。
要求：
1. 只输出严格 JSON，不要输出多余文本。
2. 优先给出完整文件内容，而不是局部片段。
3. 如果修改多个文件，返回 edits 数组。
4. 保持代码风格一致，尽量让修改可直接运行。

JSON schema:
{
  "edits": [
    {"path": "相对路径", "mode": "write|append", "content": "完整文件内容或追加内容"}
  ],
  "notes": ["实现说明1", "实现说明2"]
}
"""

REVIEWER_SYSTEM = """你是代码审查员和测试工程师。
请根据失败日志判断最可能的问题，并给出最小修复方案。
只输出严格 JSON。

JSON schema:
{
  "fix_summary": "一句话说明问题",
  "edits": [
    {"path": "相对路径", "mode": "write|append", "content": "修复后的完整文件内容或追加内容"}
  ],
  "extra_test_steps": ["新增验证步骤1", "新增验证步骤2"]
}
"""


class AgentRequest(BaseModel):
    repo: str = Field(..., description="本地仓库路径")
    task: str = Field(..., description="需求描述")
    max_iters: int = Field(default=DEFAULT_MAX_ITERS, ge=1, le=10)
    test_command: Optional[str] = Field(default=None, description="可选测试命令，例如 pytest -q")


class AgentResponse(BaseModel):
    goal: str
    assumptions: List[str]
    files_to_check: List[str]
    files_to_edit: List[str]
    implementation_steps: List[str]
    test_steps: List[str]
    risks: List[str]
    edits: List[Dict[str, Any]]
    test_output: str
    summary: str
    iterations: int


class DevCollabAgent:
    def __init__(self, repo: str):
        self.root = _repo_root(repo)

    def build_context(self, task: str) -> str:
        tree = list_repo_tree(self.root)
        # 只读取前几个潜在关键文件，减少上下文噪音
        candidates = [
            "README.md",
            "pyproject.toml",
            "package.json",
            "requirements.txt",
            "src/main.py",
            "src/app.py",
            "app.py",
            "main.py",
        ]
        snippets = []
        for rel in candidates:
            content = read_text_file(self.root, rel)
            if content:
                snippets.append(f"### {rel}\n{content}")
        return textwrap.dedent(
            f"""
            需求：
            {task}

            仓库文件树：
            {tree}

            关键文件内容：
            {chr(10).join(snippets)}
            """
        ).strip()

    def plan(self, task: str) -> AgentPlan:
        user = self.build_context(task)
        data = llm_json(PLANNER_SYSTEM, user, temperature=0.15)
        return AgentPlan(
            goal=data.get("goal", task),
            assumptions=data.get("assumptions", []),
            files_to_check=data.get("files_to_check", []),
            files_to_edit=data.get("files_to_edit", []),
            implementation_steps=data.get("implementation_steps", []),
            test_steps=data.get("test_steps", []),
            risks=data.get("risks", []),
        )

    def implement(self, task: str, plan: AgentPlan) -> List[Edit]:
        context_parts = [
            f"需求：{task}",
            f"目标：{plan.goal}",
            "假设：\n- " + "\n- ".join(plan.assumptions or ["无"]),
            "计划：\n- " + "\n- ".join(plan.implementation_steps or ["无"]),
            "待修改文件：\n- " + "\n- ".join(plan.files_to_edit or ["无"]),
            "建议先看文件：\n- " + "\n- ".join(plan.files_to_check or ["无"]),
        ]
        for rel in plan.files_to_check[:10]:
            content = read_text_file(self.root, rel)
            if content:
                context_parts.append(f"### {rel}\n{content}")
        context = "\n\n".join(context_parts)
        data = llm_json(IMPLEMENTER_SYSTEM, context, temperature=0.2)
        edits = []
        for item in data.get("edits", []):
            path = str(item.get("path", "")).strip()
            content = str(item.get("content", ""))
            mode = str(item.get("mode", "write")).strip().lower()
            if not path:
                continue
            edits.append(Edit(path=path, content=content, mode=mode if mode in {"write", "append"} else "write"))
        return edits

    def apply_edits(self, edits: List[Edit]) -> None:
        for edit in edits:
            if edit.mode == "append":
                append_text_file(self.root, edit.path, edit.content)
            else:
                write_text_file(self.root, edit.path, edit.content)

    def test(self, test_command: Optional[str], plan: AgentPlan) -> Dict[str, Any]:
        if test_command:
            return run_shell(test_command, self.root)

        # 默认测试策略：根据仓库类型猜测命令
        if (self.root / "pytest.ini").exists() or (self.root / "tests").exists():
            return run_shell("pytest -q", self.root)
        if (self.root / "package.json").exists():
            pkg = read_text_file(self.root, "package.json")
            if '"test"' in pkg:
                return run_shell("npm test -- --runInBand", self.root)
            return run_shell("npm test", self.root)
        if (self.root / "go.mod").exists():
            return run_shell("go test ./...", self.root)
        if (self.root / "Cargo.toml").exists():
            return run_shell("cargo test", self.root)
        # 没有明显测试框架时，至少做语法/静态检查的尝试
        if (self.root / "pyproject.toml").exists() or any(p.suffix == ".py" for p in self.root.rglob("*.py")):
            return run_shell("python -m compileall .", self.root)
        return {"returncode": 0, "stdout": "No test command detected.", "stderr": "", "command": ""}

    def review_and_fix(self, task: str, plan: AgentPlan, test_output: Dict[str, Any]) -> List[Edit]:
        context_parts = [
            f"需求：{task}",
            f"目标：{plan.goal}",
            "计划：\n- " + "\n- ".join(plan.implementation_steps or ["无"]),
            "测试输出：",
            f"command: {test_output.get('command', '')}",
            f"returncode: {test_output.get('returncode', '')}",
            f"stdout:\n{test_output.get('stdout', '')}",
            f"stderr:\n{test_output.get('stderr', '')}",
        ]
        data = llm_json(REVIEWER_SYSTEM, "\n\n".join(context_parts), temperature=0.15)
        edits = []
        for item in data.get("edits", []):
            path = str(item.get("path", "")).strip()
            content = str(item.get("content", ""))
            mode = str(item.get("mode", "write")).strip().lower()
            if not path:
                continue
            edits.append(Edit(path=path, content=content, mode=mode if mode in {"write", "append"} else "write"))
        return edits

    def run(self, task: str, max_iters: int = DEFAULT_MAX_ITERS, test_command: Optional[str] = None) -> RunResult:
        plan = self.plan(task)
        edits = self.implement(task, plan)
        self.apply_edits(edits)

        last_test = {"returncode": 0, "stdout": "", "stderr": "", "command": ""}
        iterations = 0
        for i in range(max_iters):
            iterations = i + 1
            last_test = self.test(test_command, plan)
            if int(last_test.get("returncode", 0)) == 0:
                break
            fix_edits = self.review_and_fix(task, plan, last_test)
            if not fix_edits:
                break
            self.apply_edits(fix_edits)
            edits.extend(fix_edits)

        summary = self.summarize(task, plan, last_test, edits)
        return RunResult(plan=plan, edits=edits, test_output=json.dumps(last_test, ensure_ascii=False, indent=2), summary=summary, iterations=iterations)

    def summarize(self, task: str, plan: AgentPlan, test_output: Dict[str, Any], edits: List[Edit]) -> str:
        changed = sorted({e.path for e in edits})
        return textwrap.dedent(
            f"""
            任务：{task}
            目标：{plan.goal}
            修改文件：{', '.join(changed) if changed else '无'}
            测试返回码：{test_output.get('returncode', 0)}
            """.strip()
        )


# -----------------------------
# FastAPI 接口
# -----------------------------


@app.post("/agent/run", response_model=AgentResponse)
def run_agent(req: AgentRequest) -> AgentResponse:
    try:
        agent = DevCollabAgent(req.repo)
        result = agent.run(req.task, max_iters=req.max_iters, test_command=req.test_command)
        return AgentResponse(
            goal=result.plan.goal,
            assumptions=result.plan.assumptions,
            files_to_check=result.plan.files_to_check,
            files_to_edit=result.plan.files_to_edit,
            implementation_steps=result.plan.implementation_steps,
            test_steps=result.plan.test_steps,
            risks=result.plan.risks,
            edits=[edit.__dict__ for edit in result.edits],
            test_output=result.test_output,
            summary=result.summary,
            iterations=result.iterations,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# -----------------------------
# CLI
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 驱动的研发协作 Agent")
    parser.add_argument("--repo", required=True, help="本地仓库路径")
    parser.add_argument("--task", required=True, help="需求描述")
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS, help="最大修复轮次")
    parser.add_argument("--test-command", default=None, help="自定义测试命令")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    agent = DevCollabAgent(args.repo)
    result = agent.run(args.task, max_iters=args.max_iters, test_command=args.test_command)

    payload = {
        "goal": result.plan.goal,
        "assumptions": result.plan.assumptions,
        "files_to_check": result.plan.files_to_check,
        "files_to_edit": result.plan.files_to_edit,
        "implementation_steps": result.plan.implementation_steps,
        "test_steps": result.plan.test_steps,
        "risks": result.plan.risks,
        "edits": [e.__dict__ for e in result.edits],
        "test_output": json.loads(result.test_output),
        "summary": result.summary,
        "iterations": result.iterations,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("\n=== 研发计划 ===")
        print(json.dumps(payload["goal"], ensure_ascii=False, indent=2))
        print("\n=== Summary ===")
        print(result.summary)
        print("\n=== Test Output ===")
        print(json.dumps(payload["test_output"], ensure_ascii=False, indent=2))
        print("\n=== Edited Files ===")
        for e in result.edits:
            print(f"- {e.path} ({e.mode})")


if __name__ == "__main__":
    main()
