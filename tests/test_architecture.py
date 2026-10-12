"""架构约束测试 —— 用工具而非文档保障分层

**存在的理由**：`CLAUDE.md` 里的架构约束写了很久，审计时仍查出五处规则
漂移。写在文档里的约束等于没有约束 —— 没人会在改代码前先去读一遍 ADR。
这份测试把约束变成 CI 会拦住的东西。

## 为什么用 AST 而不是 import-linter

要守的规则里只有一部分是纯 import 层次问题，`import-linter` 解决不了另外三条：

    "不得出现 .commit()"        —— 不是 import
    前端 .vue 里的阈值字面量     —— 不是 Python
    (未来) async def 里的同步调用 —— 需要遍历函数体

为一条规则引入一个依赖加一份独立配置，剩下三条还得再写 AST，不如统一。
已有先例：`test_repositories.py::TestNoCommit` 的源码扫描就是这个思路，
本文件把它归并进来并扩大范围。

## 为什么不用正则扫 Python

正则会把注释、docstring、字符串字面量里的 `import rag` 当成真 import。
本文件三个 api 模块的注释里恰好都写着"协议层不 import rag"这句话 ——
用正则的话它们会全部误报。`ast.parse` 只看真正的语法结构。

前端 `.vue` 没有 Python AST 可用，那一条用正则，但排除注释行。

## 每条规则都有反向测试

只断言"当前代码合规"是不够的：规则写错了（比如永远返回空列表）也会绿。
每条规则配一个 `test_*_detects_violation`，喂一段**故意违规的源码**，
断言它确实被抓出来。这样测试本身的有效性也被覆盖。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable, List, NamedTuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Violation(NamedTuple):
    file: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}  {self.detail}"


def _python_files(*relative_dirs: str) -> Iterable[Path]:
    """目录下的 .py 文件，跳过 __pycache__ 与 venv。"""
    for relative in relative_dirs:
        root = PROJECT_ROOT / relative
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts or "venv" in path.parts:
                continue
            yield path


def _imported_modules(source: str) -> List[tuple[str, int]]:
    """源码里 import 的模块名与行号。

    `import a.b` -> "a.b"；`from a.b import c` -> "a.b"。
    相对导入（`from . import x`）返回空字符串，调用方按需忽略。
    """
    tree = ast.parse(source)
    found: List[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # node.level > 0 是相对导入，模块名对本文件的检查没有意义
            found.append((node.module or "", node.lineno))
    return found


def _top_package(module: str) -> str:
    return module.split(".")[0]


# ============================================================
# 规则实现
# ============================================================

def find_forbidden_imports(
    source: str, filename: str, forbidden: set[str]
) -> List[Violation]:
    """源码里是否 import 了禁止的顶层包。"""
    violations = []
    for module, lineno in _imported_modules(source):
        if module and _top_package(module) in forbidden:
            violations.append(Violation(filename, lineno, f"import {module}"))
    return violations


def find_forbidden_names(
    source: str, filename: str, forbidden: set[str]
) -> List[Violation]:
    """源码里是否 import 了禁止的具体名字（而非整个包）。

    针对 `from config import ANSWERABLE_MIN_RELEVANCE` 这种 ——
    禁的不是 `config` 这个包，而是其中的领域阈值常量。
    """
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in forbidden:
                    violations.append(
                        Violation(filename, node.lineno, f"import {alias.name}")
                    )
    return violations


def find_method_calls(source: str, filename: str, method: str) -> List[Violation]:
    """源码里是否调用了某个方法名（如 `.commit()`）。

    用 AST 而非字符串查找：`.commit()` 出现在注释或 docstring 里
    不该算违规，而本项目的注释里确实反复提到 "不 commit"。
    """
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
        ):
            violations.append(Violation(filename, node.lineno, f".{method}()"))
    return violations


# 领域阈值：它们是领域规则的参数，出现在别的层意味着那层在做领域判断
DOMAIN_THRESHOLD_NAMES = {
    "ANSWERABLE_MIN_RELEVANCE",
    "RETRIEVAL_MIN_RELEVANCE",
}

# 前端里这些数字若与阈值语义相关就是硬编码副本。
# 用词边界避免匹配到 `0.755` 或 css 的 `0.35s`。
_THRESHOLD_LITERAL = re.compile(r"(?<![\w.])(0\.75|0\.35|0\.50)(?![\w%])")


def find_frontend_threshold_literals(source: str, filename: str) -> List[Violation]:
    """前端源码里是否硬编码了领域阈值。

    跨进程的常量副本没有任何编译器会报警 —— 这正是审计时发现的最严重一处：
    两个 Vue 组件各写死一份且值不同，后端实际用第三个值。

    排除注释行：说明"阈值由后端下发"时提到具体数值是合理的。
    """
    violations = []
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(("//", "*", "/*", "<!--", "#")):
            continue
        match = _THRESHOLD_LITERAL.search(line)
        if match:
            violations.append(
                Violation(filename, lineno, f"硬编码阈值 {match.group(1)}")
            )
    return violations


def _scan(dirs: Iterable[str], checker) -> List[Violation]:
    violations: List[Violation] = []
    for path in _python_files(*dirs):
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        violations.extend(checker(path.read_text(encoding="utf-8"), rel))
    return violations


def _fail(violations: List[Violation], rule: str) -> None:
    if violations:
        listed = "\n  ".join(str(v) for v in violations)
        pytest.fail(f"{rule}\n  {listed}")


# ============================================================
# 规则 1：rag/ 不得 import backend/
# ============================================================

class TestCoreDoesNotDependOnWeb:
    """CLAUDE.md 的硬约束，也是本项目唯一一直没被违反的一条。

    它的收益是可验证的：`rag/eval/` 那套评测能脱离 FastAPI 独立跑，
    `scripts/run_eval.py` 直接调 `rag/`，不需要起 Web 服务。
    """

    def test_rag_does_not_import_backend(self):
        violations = _scan(
            ["rag"],
            lambda src, name: find_forbidden_imports(src, name, {"backend"}),
        )
        _fail(violations, "rag/ 不得 import backend/（依赖方向必须单向）")

    def test_detects_violation(self):
        bad = "from backend.services.chat_service import get_chat_service\n"
        assert find_forbidden_imports(bad, "rag/x.py", {"backend"})

    def test_ignores_mention_in_comment(self):
        """注释里提到 backend 不算违规 —— 这正是不用正则的理由。"""
        ok = '"""这个模块不 import backend。"""\n# import backend 是禁止的\n'
        assert not find_forbidden_imports(ok, "rag/x.py", {"backend"})


# ============================================================
# 规则 2：backend/api/ 不得 import rag
# ============================================================

class TestApiDoesNotDependOnDomain:
    """协议层只做协议的事：解析请求、调 service、组装响应。

    违反它的表现是 `routes.py` 里出现 ChromaDB 调用和 60 行数据变换 ——
    换成 CLI 一字不变的代码，不该待在 HTTP 端点里。
    """

    def test_api_does_not_import_rag(self):
        violations = _scan(
            ["backend/api"],
            lambda src, name: find_forbidden_imports(src, name, {"rag"}),
        )
        _fail(violations, "backend/api/ 不得 import rag（协议层不碰领域层）")

    def test_detects_violation(self):
        assert find_forbidden_imports("from rag import VectorDB\n", "api.py", {"rag"})
        assert find_forbidden_imports("import rag.logger\n", "api.py", {"rag"})

    def test_allows_backend_imports(self):
        """api 可以 import backend.schemas / backend.services。"""
        ok = "from backend.schemas import QueryRequest\n"
        assert not find_forbidden_imports(ok, "api.py", {"rag"})


# ============================================================
# 规则 3：Repository 不得 commit
# ============================================================

class TestRepositoriesDoNotCommit:
    """事务边界归调用方。任何 Repository 方法自己提交，跨表原子性就没了。

    从 `test_repositories.py::TestNoCommit` 迁入并改用 AST ——
    原先是字符串查找，注释里写 "不 commit" 也会被算成违规。
    """

    def test_no_commit_in_repositories(self):
        violations = _scan(
            ["backend/repositories"],
            lambda src, name: find_method_calls(src, name, "commit"),
        )
        _fail(violations, "Repository 不得 commit（事务边界归调用方）")

    def test_detects_violation(self):
        bad = "async def save(self):\n    await self.session.commit()\n"
        assert find_method_calls(bad, "runs.py", "commit")

    def test_ignores_mention_in_docstring(self):
        ok = '"""本模块不调用 session.commit()，事务边界归调用方。"""\n'
        assert not find_method_calls(ok, "runs.py", "commit")


# ============================================================
# 规则 4：service 层不得引入领域阈值
# ============================================================

class TestServicesDoNotOwnDomainThresholds:
    """阈值是领域规则的参数。service 层 import 它意味着在那里做领域判断。

    这正是审计出的副本漂移：`chat_service` 手抄了一份可答性判断，
    而权威实现一直在 `rag/llm.py::assess_context`。

    **边界**：禁的只是阈值常量，不禁 service 读 config 的其它值
    （如 `SESSION_TIMEOUT`）—— 那些是应用配置不是领域规则。
    """

    def test_services_do_not_import_thresholds(self):
        violations = _scan(
            ["backend/services"],
            lambda src, name: find_forbidden_names(src, name, DOMAIN_THRESHOLD_NAMES),
        )
        _fail(
            violations,
            "service 层不得 import 领域阈值（可答性判断属于 rag/，见 ADR-004）",
        )

    def test_detects_violation(self):
        bad = "from config import ANSWERABLE_MIN_RELEVANCE\n"
        assert find_forbidden_names(bad, "chat_service.py", DOMAIN_THRESHOLD_NAMES)

    def test_allows_other_config_values(self):
        ok = "from config import SESSION_TIMEOUT, TOP_K_RESULTS\n"
        assert not find_forbidden_names(ok, "chat_service.py", DOMAIN_THRESHOLD_NAMES)


# ============================================================
# 规则 5：前端不得硬编码领域阈值
# ============================================================

class TestFrontendDoesNotHardcodeThresholds:
    """前端必须从 `/config/thresholds` 拉，不能自己写死。

    这条是五处漂移里唯一**已经产生用户可见错误**的：
    `BrainPanel` 用 50、`ChatPanel` 用 0.75、后端实际用 0.75，
    于是 relevance=0.60 时仪表盘说"足以支撑回答"、引用卡片显示警告色、
    后端判定不可答。跨进程副本没有编译器会管。
    """

    # 唯一的例外：兜底值必须写在某处。把它集中在这一个文件里，
    # 是"只有一份副本且标注清楚"与"零副本但拿不到阈值就没法上色"
    # 之间的取舍。允许例外的前提是它**只有一处**，而不是散在各组件里。
    FALLBACK_FILE = "frontend-vue/src/composables/useThresholds.ts"

    def test_no_hardcoded_thresholds_in_frontend(self):
        src_dir = PROJECT_ROOT / "frontend-vue" / "src"
        if not src_dir.exists():
            pytest.skip("前端源码不存在")

        violations: List[Violation] = []
        for path in list(src_dir.rglob("*.vue")) + list(src_dir.rglob("*.ts")):
            if "node_modules" in path.parts:
                continue
            if path.relative_to(PROJECT_ROOT).as_posix() == self.FALLBACK_FILE:
                continue
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            violations.extend(
                find_frontend_threshold_literals(path.read_text(encoding="utf-8"), rel)
            )
        _fail(violations, "前端不得硬编码领域阈值（改从 /config/thresholds 拉）")

    def test_fallback_file_exists_and_is_the_only_exception(self):
        """白名单文件必须存在，否则这条例外就是个静默失效的口子。

        白名单本身会变成新的漂移入口 —— 有人把阈值搬进这个文件就能绕过
        检查。这条确保它至少还在，且下面那条确保它没变成一个大杂烩。
        """
        assert (PROJECT_ROOT / self.FALLBACK_FILE).exists(), (
            f"白名单文件 {self.FALLBACK_FILE} 不存在。"
            "若已重构，请更新 FALLBACK_FILE 或删除这条例外。"
        )

    def test_fallback_file_only_declares_defaults(self):
        """白名单文件里的阈值字面量不得超过 2 个（两个阈值各一份）。

        防的是"把所有阈值逻辑都塞进白名单文件"这种绕过方式。
        """
        source = (PROJECT_ROOT / self.FALLBACK_FILE).read_text(encoding="utf-8")
        hits = find_frontend_threshold_literals(source, self.FALLBACK_FILE)
        assert len(hits) <= 2, (
            f"白名单文件里有 {len(hits)} 处阈值字面量，超过两个默认值。"
            f"它只该声明 FALLBACK，不该承载分档逻辑：\n  "
            + "\n  ".join(str(h) for h in hits)
        )

    def test_detects_violation(self):
        bad = "if (relevance >= 0.75) return 'text-neon-blue';\n"
        assert find_frontend_threshold_literals(bad, "ChatPanel.vue")

    def test_ignores_comment_lines(self):
        ok = "// 阈值由后端下发，此前这里写死 0.75\n"
        assert not find_frontend_threshold_literals(ok, "ChatPanel.vue")

    def test_ignores_unrelated_numbers(self):
        """不能误伤无关数字：css 时长、透明度、别的业务常量。"""
        assert not find_frontend_threshold_literals(
            "transition: all 0.35s ease;\n", "x.vue"
        )
        assert not find_frontend_threshold_literals("opacity: 0.755;\n", "x.vue")


# ============================================================
# 规则 6：backend 内部分层，下层不得 import 上层
# ============================================================

# 从上到下。列表里靠后的层不得 import 靠前的层。
BACKEND_LAYERS = [
    "backend.api",
    "backend.services",
    "backend.repositories",
    "backend.db",
]


def find_upward_imports(source: str, filename: str, layer: str) -> List[Violation]:
    """`layer` 层的源码是否 import 了比它更上层的模块。

    按模块前缀判断而非顶层包：五层都在 `backend` 包下，
    规则 1/2 用的顶层包比较在这里区分不出层次。
    """
    upper = BACKEND_LAYERS[: BACKEND_LAYERS.index(layer)]
    violations = []
    for module, lineno in _imported_modules(source):
        for up in upper:
            if module == up or module.startswith(up + "."):
                violations.append(
                    Violation(filename, lineno, f"{layer} → import {module}")
                )
    return violations


class TestBackendLayering:
    """依赖只能自上而下：api → services → repositories → db。

    这条是 M2 收口时补的。`ingest_service` 为了清理临时文件
    `from backend.api.upload import _remove_temp_file` —— service 反过来
    依赖协议层。规则 2 只管"api 不得 import rag"，没覆盖层内方向，
    于是漏了。修法是把临时文件工具移到 `services/temp_files.py`。

    同层互相 import 不禁（如 `document_service` 调 `chat_service`），
    那是同一层内的协作，不改变依赖方向。
    """

    @pytest.mark.parametrize("layer", BACKEND_LAYERS[1:])
    def test_lower_layer_does_not_import_upper(self, layer):
        directory = layer.replace(".", "/")
        violations = _scan(
            [directory],
            lambda src, name: find_upward_imports(src, name, layer),
        )
        _fail(violations, f"{layer} 不得 import 上层（依赖必须自上而下）")

    def test_detects_violation(self):
        bad = "from backend.api.upload import _remove_temp_file\n"
        assert find_upward_imports(bad, "ingest_service.py", "backend.services")

    def test_detects_deep_violation(self):
        bad = "from backend.services.chat_service import get_chat_service\n"
        assert find_upward_imports(bad, "runs.py", "backend.repositories")

    def test_allows_same_and_lower_layers(self):
        ok = (
            "from backend.services.chat_service import get_chat_service\n"
            "from backend.repositories import RunRepository\n"
            "from backend.db.session import session_scope\n"
        )
        assert not find_upward_imports(ok, "ingest_service.py", "backend.services")

    def test_prefix_does_not_overmatch(self):
        """`backend.apiary` 不是 `backend.api` —— 前缀比较要带点号。"""
        ok = "import backend.apiary\n"
        assert not find_upward_imports(ok, "x.py", "backend.services")
