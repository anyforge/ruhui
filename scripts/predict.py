"""ruhui 推理脚本：加载微调后的 checkpoint 做决策。

用法:
  python3 scripts/predict.py \
    --model_dir <ruhui_checkpoint_dir> \
    --text "我被重复扣款了，请退款" \
    --question-type noul --instruction "客户是否要求退款？"
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ruhui


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--question-type", required=True, choices=["choice", "score", "noul"])
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--criteria", default=None, help='JSON 字符串: choice用 {"optA":"desc",...}, score用 ["lvl0","lvl1",...]')
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    agent = ruhui.Agent(args.model_dir, device=args.device)

    q = {"type": args.question_type, "instructions": args.instruction}
    if args.criteria:
        crit = json.loads(args.criteria)
        q["criteria"] = crit

    result = agent.predict(args.text, {"q": q})
    answer = result["answers"]["q"]

    print("\n=== ruhui 决策结果 ===")
    if answer["type"] == "choice":
        print(f"选择: {answer['choice']} (置信度 {answer['confidence']})")
        print(f"概率分布: {json.dumps(answer['probabilities'], ensure_ascii=False)}")
    elif answer["type"] == "score":
        print(f"得分: {answer['score']}")
        print(f"等级分布: {json.dumps(answer['probabilities'], ensure_ascii=False)}")
    else:
        print(f"判断: {answer['noul']} (P(true))")
    print(f"动作概率: {answer['action']}")


if __name__ == "__main__":
    main()
