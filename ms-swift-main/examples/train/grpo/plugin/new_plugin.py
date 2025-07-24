import asyncio
import re
import textwrap
from copy import deepcopy
from typing import Dict, List, Optional
from dataclasses import dataclass
from typing import Any, Callable
import json
import torch

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import ORM, orms, rm_plugins
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()
"""
Step 1: Define a Reward Class
    Implement your custom reward calculation logic within the __call__ method.
    The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

Step 2: Register the Reward Class in orms
    For example:
    python orms['external_math_acc'] = MathAccuracy

Step 3: Configure the Arguments
    Use the following arguments when running the script:
    bash --plugin /path/to/plugin.py --reward_funcs external_math_acc
"""


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py

# ------------------ New MMFormatORM for multimodal tasks ------------------
@dataclass
class Rule:
    name: str
    weight: float
    check: Callable[[Any, dict | None], bool]

class MMFormatORM(ORM):
    """
    Format reward for multimodal preference-alignment tasks.
    
    Output must be a JSON object with:
        - "score_A", "score_B" : int 0-10
        - "better"             : "A" | "B" | "equal"
        - "reasoning"          : string wrapped in <think> ... </think>
        - "final_verdict"      : <answer>[[A|B|equal]]</answer>
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        # 使用实例属性而不是类属性
        self.weights = weights or {
            "json_valid": 0.2,
            "required_keys": 0.2,
            "field_values": 0.15,
            "reasoning_tag": 0.15,
            "verdict_tag": 0.15,
            "consistency": 0.15,
        }
        
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1 (got {total})")
        
        # 定义正则表达式（修正后的正确版本）
        self.RE_THINK   = re.compile(r'<think>[\s\S]*?</think>', re.DOTALL)
        self.RE_VERDICT = re.compile(r'<answer>\[\[(A|B|equal)\]\]</answer>')
        
        # 构建规则列表
        self.rules: List[Rule] = [
            Rule("json_valid", self.weights["json_valid"], self._json_valid),
            Rule("required_keys", self.weights["required_keys"], self._required_keys),
            Rule("field_values", self.weights["field_values"], self._field_values),
            Rule("reasoning_tag", self.weights["reasoning_tag"], self._reasoning_tag),
            Rule("verdict_tag", self.weights["verdict_tag"], self._verdict_tag),
            Rule("consistency", self.weights["consistency"], self._consistency),
        ]
    
    # 定义必需键集合
    REQUIRED_KEYS = {"score_A", "score_B", "better", "reasoning", "final_verdict"}
    
    # ---------- 单个规则检查 ----------
    @staticmethod
    def _json_valid(raw: str, obj: dict | None = None) -> bool:
        if obj is None:
            try:
                json.loads(raw)
                return True
            except Exception:
                return False
        return True
    
    def _required_keys(self, raw: str, obj: dict | None = None) -> bool:
        if obj is None:
            try:
                obj = json.loads(raw)
            except Exception:
                return False
        return set(obj.keys()) == self.REQUIRED_KEYS
    
    def _field_values(self, raw: str, obj: dict | None = None) -> bool:
        if obj is None:
            try:
                obj = json.loads(raw)
            except Exception:
                return False
        try:
            # 验证score_A是0-10的整数
            if not (isinstance(obj["score_A"], int) and 0 <= obj["score_A"] <= 10):
                return False
            # 验证score_B是0-10的整数
            if not (isinstance(obj["score_B"], int) and 0 <= obj["score_B"] <= 10):
                return False
            # 验证better是有效值
            if obj["better"] not in {"A", "B", "equal"}:
                return False
        except Exception:
            return False
        return True
    
    def _reasoning_tag(self, raw: str, obj: dict | None = None) -> bool:
        if obj is None:
            try:
                obj = json.loads(raw)
            except Exception:
                return False
        reasoning = obj.get("reasoning", "")
        return isinstance(reasoning, str) and self.RE_THINK.match(reasoning.strip()) is not None
    
    def _verdict_tag(self, raw: str, obj: dict | None = None) -> bool:
        if obj is None:
            try:
                obj = json.loads(raw)
            except Exception:
                return False
        verdict = obj.get("final_verdict", "")
        return isinstance(verdict, str) and self.RE_VERDICT.match(verdict.strip()) is not None
    
    def _consistency(self, raw: str, obj: dict | None = None) -> bool:
        if obj is None:
            try:
                obj = json.loads(raw)
            except Exception:
                return False
        better = obj.get("better")
        verdict = obj.get("final_verdict", "")
        m = self.RE_VERDICT.match(verdict.strip()) if isinstance(verdict, str) else None
        return m is not None and m.group(1) == better
    
    # ---------- 主入口 ----------
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for resp in completions:
            try:
                parsed = json.loads(resp)
            except Exception:
                parsed = None
            score = 0.0
            for rule in self.rules:
                if rule.check(resp, parsed):
                    score += rule.weight
            rewards.append(score)
        return rewards
        
import logging
logger = logging.getLogger(__name__)

class MMContentORM:  # inherits ORM in your project if needed
    # ---------- 初始化 ----------
    def __init__(self, weight_dir: float = 0.6, weight_score: float = 0.4):
        if abs(weight_dir + weight_score - 1.0) > 1e-6:
            raise ValueError("weight_dir + weight_score 必须等于 1")
        self.w_dir = weight_dir
        self.w_score = weight_score

    # ---------- 工具 ----------
    @staticmethod
    def _safe_int(x: Any) -> int:
        """转 int & 裁剪到 0‑10"""
        try:
            v = int(x)
        except Exception:
            raise ValueError("score 解析失败")
        return max(0, min(10, v))

    @staticmethod
    def _parse_pred(raw: str) -> Dict[str, int | str]:
        """解析模型输出"""
        obj = json.loads(raw)
        return {
            "score_A": MMContentORM._safe_int(obj["score_A"]),
            "score_B": MMContentORM._safe_int(obj["score_B"]),
            "better": str(obj["better"]),
        }

    @staticmethod
    def _self_consistent(sA: int, sB: int, better: str) -> bool:
        if better == "A":
            return sA > sB
        if better == "B":
            return sB > sA
        if better == "equal":
            return sA == sB
        return False

    # ---------- 主入口 ----------
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """评估一批 completions 的内容奖励"""
        # --------- ① 取 ground‑truth ---------
        try:
            gt_better = kwargs["better"]
            gt_sA = kwargs["score_A"]
            gt_sB = kwargs["score_B"]
        except KeyError as e:
            # ---------- ② fallback：尝试从 solution 中补字段 ----------
            if "solution" in kwargs:
                try:
                    sol_raw = kwargs["solution"]
                    # 有些 Loader 会把 solution 读成 list[str]
                    if isinstance(sol_raw, list):
                        sol_raw = sol_raw[0]
                    sol = json.loads(sol_raw)
                    for k in ("better", "score_A", "score_B"):
                        kwargs[k] = sol[k]
                    gt_better = kwargs["better"]
                    gt_sA = kwargs["score_A"]
                    gt_sB = kwargs["score_B"]
                    logger.debug("[MMContentORM] 从 solution 补字段成功")
                except Exception as ee:
                    logger.error(f"[MMContentORM] solution 解析失败: {ee}")
                    return [-1.0] * len(completions)
            else:
                logger.error(f"[MMContentORM] 缺少必要字段: {e}")
                return [-1.0] * len(completions)

        # 若为单值则 broadcast 成列表
        if not isinstance(gt_better, list):
            gt_better = [gt_better] * len(completions)
        if not isinstance(gt_sA, list):
            gt_sA = [gt_sA] * len(completions)
        if not isinstance(gt_sB, list):
            gt_sB = [gt_sB] * len(completions)

        rewards: List[float] = []

        # --------- ③ 逐条计算 reward ---------
        for raw, g_pref, g_a, g_b in zip(completions, gt_better, gt_sA, gt_sB):
            try:
                # ---------- 解析模型输出 ----------
                pred = self._parse_pred(raw)
                p_a, p_b, p_pref = pred["score_A"], pred["score_B"], pred["better"]

                # ---------- (1) 自洽检验 ----------
                if not self._self_consistent(p_a, p_b, p_pref):
                    rewards.append(-1.0)
                    continue

                # ---------- (2) 方向正确 ----------
                C_dir = 1.0 if p_pref == str(g_pref) else -1.0

                # ---------- (3) 分数贴近 ----------
                g_a, g_b = self._safe_int(g_a), self._safe_int(g_b)
                err = (abs(p_a - g_a) + abs(p_b - g_b)) / 20.0  # ∈ [0,1]
                C_score = 1.0 - 2.0 * err                       # 映射到 [-1,1]

                # ---------- (4) 汇总 ----------
                R = self.w_dir * C_dir + self.w_score * C_score
                rewards.append(float(max(-1.0, min(1.0, R))))
            except Exception as e:
                logger.debug(f"[MMContentORM] 解析/计算失败: {e}")
                rewards.append(-1.0)

        return rewards

class MMRubricORM(ORM):
    """
    Rubric-level reward for multimodal preference evaluation.
    ---------------------------------------------------------
    ①  维度覆盖率   C_cover = 已命中维度/5
    ②  维度对比率   C_cmp   = 有对比的维度/5
        ─ 显式对比: 同一段落同时出现 A 与 B
        ─ 隐式对比: 'both/answers/…' 等集合词 + better/worse 等比较词
    ③  动态增益     Δ_cmp  = max(0 , C_cmp − C_cmp_gt)
                     ( 若未提供 cmp_gt → Δ_cmp = C_cmp )
    最终奖励:  R = w_cover·C_cover + w_cmp·Δ_cmp   , 取值 [0,1]
        默认 w_cover = 0.8 , w_cmp = 0.2
    """

    # ---------- 初始化 ----------
    def __init__(self, w_cover: float = 0.8, w_cmp: float = 0.2):
        if abs(w_cover + w_cmp - 1.0) > 1e-6:
            raise ValueError('w_cover + w_cmp must equal 1')
        self.wc, self.wm = w_cover, w_cmp

        # 维度关键词
        self.dim_patterns: Dict[str, re.Pattern] = {
            'fluency':   re.compile(r'\b(fluency|coherence|flow|coherent)\b', re.I),
            'relevance': re.compile(r'\b(relevance|related|pertinent|alignment)\b', re.I),
            'accuracy':  re.compile(r'\b(accuracy|accurate|correct(ness)?|precision)\b', re.I),
            'reasoning': re.compile(r'\b(reasoning|analysis|logic(al)?|inference)\b', re.I),
            'safety':    re.compile(r'\b(safety|safe|ethical|harmless|toxic(ity)?)\b', re.I),
        }

        # A / B 检测
        self.re_A = re.compile(r'(?<![A-Za-z])(?:candidate|answer|response|model|option)?\s*A(?![A-Za-z])', re.I)
        self.re_B = re.compile(r'(?<![A-Za-z])(?:candidate|answer|response|model|option)?\s*B(?![A-Za-z])', re.I)
        self.re_collective  = re.compile(r'\b(both|two|each|either|neither|all|responses?|candidates?|answers?)\b', re.I)
        self.re_comparative = re.compile(
            r'\b(better|worse|superior|inferior|preferable|outperform\w*|more\s+\w+|less\s+\w+)\b', re.I
        )

    # ---------- 工具 ----------
    @staticmethod
    def _strip_think(txt: str) -> str:
        """去除 <think> 标签"""
        return re.sub(r'^<think>\s*|\s*</think>$', '', txt.strip(), flags=re.I)

    # ---------- 主入口 ----------
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        cmp_gt_val = None
        if 'cmp_gt' in kwargs:
            try:
                cmp_gt_val = max(0.0, min(1.0, float(kwargs['cmp_gt'])))
            except Exception:
                cmp_gt_val = None

        rewards: List[float] = []

        for raw in completions:
            # ---------- 提取 reasoning ----------
            try:
                obj = json.loads(raw)
                reasoning = self._strip_think(obj.get('reasoning', ''))
            except Exception as e:
                logger.debug(f'[Rubric] JSON parse error: {e}')
                rewards.append(-1.0)
                continue

            cover_hit, cmp_hit = 0, 0                 # 维度命中 / 对比命中

            # ---------- 枚举 5 维 ----------
            for dim, rg in self.dim_patterns.items():
                # 找到该维度首个出现位置
                m = rg.search(reasoning)
                if not m:
                    continue                          # 未覆盖此维

                cover_hit += 1
                start = m.start()

                # 计算该维度段落的结束位置 = 下一维度的最小 start
                next_positions = []
                for other_rg in self.dim_patterns.values():
                    if other_rg is rg:
                        continue
                    m2 = other_rg.search(reasoning, start + 1)
                    if m2:
                        next_positions.append(m2.start())
                end = min(next_positions) if next_positions else len(reasoning)
                segment = reasoning[start:end]

                # ---------- 对比检测 ----------
                explicit  = self.re_A.search(segment) and self.re_B.search(segment)
                implicit  = self.re_collective.search(segment) and self.re_comparative.search(segment)
                if explicit or implicit:
                    cmp_hit += 1

            # ---------- 计算得分 ----------
            C_cover = cover_hit / 5.0
            C_cmp   = cmp_hit   / 5.0

            # 动态增益：只奖励 “比 GT 更好”
            if cmp_gt_val is not None:
                cmp_gain = max(0.0, C_cmp - cmp_gt_val)
            else:
                cmp_gain = C_cmp

            reward = self.wc * C_cover + self.wm * cmp_gain
            rewards.append(float(reward))

        return rewards

class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


class CodeRewardByJudge0(ORM):
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        import os
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        self.headers = {'Content-Type': 'application/json'}
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        if language is None:
            return cls.PYTHON_ID
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        import aiohttp
        try:
            passed = 0
            total = len(test_cases)

            for case in test_cases:
                if code is not None and code != '':
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

    async def run_async(self):
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        results = await asyncio.gather(*tasks)
        rewards = list(results)
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        self.verification_info = kwargs['verification_info']

        languages = [info['language'] for info in self.verification_info]
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            rewards = self.run_async_from_sync()
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_code_reward'] = CodeReward
orms['external_code_format'] = CodeFormat
orms['external_code_reward_by_judge0'] = CodeRewardByJudge0
orms['mm_format'] = MMFormatORM
orms['mm_content'] = MMContentORM
orms['mm_rubric'] = MMRubricORM

# For genrm you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)
        reward_inputs.pop('labels')

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin