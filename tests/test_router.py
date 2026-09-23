"""ruhui Router 回归测试（单模型架构）。

ruhui 只有一个 checkpoint（anyforge/ruhui），Router 保留三模型命名是为了 API 兼容，
但所有路由名最终都指向同一个模型。这里只测 ruhui 真实有的行为：
  - normalise_name 别名
  - 路由决策（脚本检测仍工作，但模型名都映射到默认）
  - model_path 本地路径加载
  - route 返回的 repo 字符串
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ruhui.router import Router, normalise_name, _repo_str, BUNDLE_REPO  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    (PASS if got == want else FAIL).append(name)
    print("%s %s (got=%r want=%r)" % ("PASS" if got == want else "FAIL", name, got, want))


# ---- 别名 ----
for alias, want in [("ruhui", "default"), ("en", "english"), ("multi", "multilingual"),
                    ("ML", "multilingual"), ("typed", "typed-decisions"), ("default", "english")]:
    check("alias/" + alias, normalise_name(alias), want)

try:
    normalise_name("nope")
    FAIL.append("alias/unknown should raise")
except ValueError:
    PASS.append("alias/unknown raises")

# ---- repo 字符串 ----
check("repo_str/root", _repo_str((BUNDLE_REPO, None)), "anyforge/ruhui")

# ---- 路由决策（脚本检测仍工作，但所有模型名收敛为默认）----
r = Router()
Q = {"dept": {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": None, "tech": None}}}
check("route/english", r.route({"body": "I was charged twice"}, Q)["model"], "english")
check("route/chinese", r.route({"body": "我被重复扣款了"}, Q)["model"], "multilingual")

# 显式 model_path 本地加载（不触发网络）
# 只验证 model_path 参数被接受并校验路径，不实际加载权重
try:
    Router(model_path="/nonexistent/path")
    FAIL.append("model_path/nonexistent should raise")
except FileNotFoundError:
    PASS.append("model_path/nonexistent raises")

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
