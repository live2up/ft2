"""
factor/v5/llm/generator.py — LLM 因子生成器

提供四个核心能力:
  generate(idea, n)           → 按自然语言想法生成因子表达式
  generate_with_feedback(...)  → 带评估反馈的迭代生成 (自动化循环用)
  mutate(expr, strat)         → 对表达式做语义变异
  explain(expr)               → 用自然语言解释因子含义

支持多 LLM provider: OpenAI / DeepSeek / 本地(vLLM/Ollama)
"""

import json
import re
from typing import Optional, List, Callable, Tuple

from .prompts import (
    build_generate_prompt, build_mutate_prompt,
    build_explain_prompt, build_feedback_prompt,
)


class LLMGenerator:
    """LLM 因子生成器 — 自然语言 → AST 表达式的编译器

    用法:
        >>> gen = LLMGenerator(provider="deepseek")
        >>> exprs = gen.generate("量价背离方向", n=10)
        >>> variants = gen.mutate(exprs[0], "change_window", n=5)
        >>> print(gen.explain(exprs[0]))
    """

    def __init__(self, provider: str = "openai",
                 model: str = None,
                 api_key: str = None,
                 base_url: str = None,
                 custom_call: Callable = None):
        """
        Args:
            provider: "openai" | "deepseek" | "custom"
            model: 模型名 (默认: gpt-4o-mini / deepseek-chat)
            api_key: API key
            base_url: 自定义 API 地址
            custom_call: 自定义调用函数 fn(messages) -> str
        """
        self.provider = provider
        self.custom_call = custom_call

        if provider == "deepseek":
            self.model = model or "deepseek-chat"
            self.base_url = base_url or "https://api.deepseek.com/v1"
        elif provider == "openai":
            self.model = model or "gpt-4o-mini"
            self.base_url = base_url or "https://api.openai.com/v1"
        else:
            self.model = model or "gpt-4o-mini"
            self.base_url = base_url

        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError("需要 openai 包: pip install openai")
        return self._client

    def _call_llm(self, messages: list, temperature: float = 0.8,
                  max_tokens: int = 2000) -> str:
        """统一的 LLM 调用接口"""
        if self.custom_call:
            return self.custom_call(messages)

        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def _parse_response(self, text: str) -> List[str]:
        """从 LLM 回复中提取因子表达式列表

        支持格式:
          - JSON数组: ["expr1", "expr2"]
          - 逐行: 每行一个表达式 (# 后为注释)
          - 混合: 先尝试 JSON, 失败则逐行提取
        """
        # 尝试 JSON 解析
        text_clean = text.strip()
        for pattern in [
            r'```json\s*(\[.*?\])\s*```',
            r'```\s*(\[.*?\])\s*```',
            r'(\[".*?"[,\s]*".*?"[,\s]*\])',
        ]:
            m = re.search(pattern, text_clean, re.DOTALL)
            if m:
                try:
                    arr = json.loads(m.group(1))
                    return [self._clean_expr(s) for s in arr if isinstance(s, str)]
                except json.JSONDecodeError:
                    continue

        # 逐行提取
        exprs = []
        for line in text_clean.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            # 去掉行内注释
            if '#' in line:
                line = line.split('#')[0].strip()
            if line and not line.startswith('```'):
                cleaned = self._clean_expr(line)
                if cleaned:
                    exprs.append(cleaned)

        return exprs

    def _clean_expr(self, s: str) -> str:
        """清理表达式: 去掉引号/分隔符/多余空格"""
        s = s.strip().strip('"').strip("'").strip(",").strip(";")
        # 去掉编号前缀 "1. expr" → "expr"
        s = re.sub(r'^\d+[\.\)、]\s*', '', s)
        return s if s else ""

    # ══════════════════════════════════════════════════════════
    # 公开 API
    # ══════════════════════════════════════════════════════════

    def generate(self, idea: str, n: int = 10,
                 context: str = "",
                 avoid: Optional[List[str]] = None,
                 temperature: float = 0.8) -> List[str]:
        """根据自然语言想法生成因子表达式

        Args:
            idea: 因子想法描述，如 "量价背离，价涨量缩的反转信号"
            n: 生成数量
            context: 上下文（已有优秀因子等，帮助定向生成）
            avoid: 已有表达式列表（去重），LLM 会跳过这些
            temperature: 创造力参数 (0=保守, 1=创造性)

        Returns:
            因子表达式字符串列表
        """
        prompt = build_generate_prompt(idea, n, context, avoid)
        messages = [
            {"role": "system", "content": "你是量化因子生成器。严格按照要求输出，不要多余文字。"},
            {"role": "user", "content": prompt},
        ]
        text = self._call_llm(messages, temperature, max_tokens=3000)
        return self._dedup(exprs, n)

    def generate_with_feedback(self, idea: str,
                                survivors: List[Tuple[str, float, float]],
                                failures: List[Tuple[str, str]] = None,
                                round_id: int = 1,
                                n: int = 10,
                                avoid: Optional[List[str]] = None,
                                temperature: float = 0.8) -> List[str]:
        """带评估反馈的迭代生成（自动化循环入口）

        Args:
            idea: 因子想法描述
            survivors: 存活因子 [(expr, ICIR, Sharpe), ...] 或 [(expr, IC), ...]
            failures: 失败因子 [(expr, 失败原因), ...]
            round_id: 当前轮次编号
            n: 生成数量
            avoid: 累计已生成的所有表达式（去重）
            temperature: 创造力参数

        Returns:
            因子表达式字符串列表
        """
        prompt = build_feedback_prompt(
            idea, n, survivors, failures, round_id, avoid,
        )
        messages = [
            {"role": "system", "content": "你是量化因子生成器。基于反馈迭代优化。不要多余文字。"},
            {"role": "user", "content": prompt},
        ]
        text = self._call_llm(messages, temperature, max_tokens=3000)
        return self._dedup(self._parse_response(text), n)

    def _dedup(self, exprs: List[str], n: int) -> List[str]:
        """去重"""
        seen = set()
        result = []
        for e in exprs[:n]:
            if e not in seen:
                seen.add(e)
                result.append(e)
        return result

    def mutate(self, expr: str, strategy: str = "change_window",
               n: int = 5, temperature: float = 0.7) -> List[str]:
        """对给定表达式做语义变异

        Args:
            expr: 原始表达式
            strategy: 变异策略
                - "change_window": 换窗口参数 (20→10/60/120)
                - "swap_operator": 换算子 (ts_roc→ts_zscore, ts_mean→ema)
                - "add_term": 加维度 (增量价/波动/趋势)
                - "reverse_direction": 反转方向
                - "cross_section": 加减截面包装
            n: 变异数量

        Returns:
            变异后的表达式列表
        """
        prompt = build_mutate_prompt(expr, strategy, n)
        messages = [
            {"role": "system", "content": "你是量化因子变异器。严格输出，不要多余文字。"},
            {"role": "user", "content": prompt},
        ]
        text = self._call_llm(messages, temperature, max_tokens=1500)
        return self._parse_response(text)[:n]

    def explain(self, expr: str) -> str:
        """用自然语言解释因子含义

        Returns:
            一句话解释 (50字以内)
        """
        prompt = build_explain_prompt(expr)
        messages = [
            {"role": "system", "content": "你是量化因子解释器。简洁扼要。"},
            {"role": "user", "content": prompt},
        ]
        return self._call_llm(messages, temperature=0.3, max_tokens=200).strip()
