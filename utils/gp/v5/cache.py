"""
utils/gp/v5/cache.py — Fitness 缓存（内存 + SQLite 持久化）
=============================================================================
[抽取] 2026-07-06 从 engine.py 拆分。封装缓存逻辑，支持 LRU 上限演进。
"""
import hashlib
import sqlite3
import threading
from typing import Dict, Optional, Tuple


class FitnessCache:
    """GP 适应度缓存 — 内存 + SQLite 双级（懒加载）

    用法:
        cache = FitnessCache(cache_db='/path/to/cache.db', fitness_hash='abc')
        cached = cache.get(key)          # 先查内存，查不到再查 SQLite
        if cached is None:
            cache.put(key, (fitness, depth, nodes))

    [重构] 2026-07-08 从全量加载改为懒加载: _mem 初始为空，
    get() 先查内存，查不到再从 SQLite 按 expr_hash 捞单条。
    避免 10 万+ 历史缓存一次性读入内存。
    """

    def __init__(self, cache_db: str = '', fitness_hash: str = ''):
        self._mem: Dict[str, Tuple[float, int, int]] = {}
        self._db = cache_db
        self._hash = fitness_hash
        self._lock = threading.Lock()
        if self._db:
            self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self._db)
        conn.execute('''CREATE TABLE IF NOT EXISTS expressions (
            expr_hash TEXT PRIMARY KEY,
            expression TEXT NOT NULL,
            fitness REAL, depth INTEGER, nodes INTEGER,
            fitness_hash TEXT, created_at TEXT DEFAULT (datetime('now'))
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_expr_hash ON expressions(fitness_hash)')
        conn.commit()
        conn.close()

    def get(self, key: str) -> Optional[Tuple[float, int, int]]:
        # 先查内存
        cached = self._mem.get(key)
        if cached is not None:
            return cached
        # 查不到则从 SQLite 懒加载单条
        if not self._db:
            return None
        eh = hashlib.md5(key.encode()).hexdigest()[:16]
        with self._lock:
            # 双检锁: 可能另一线程已加载
            if key in self._mem:
                return self._mem[key]
            conn = sqlite3.connect(self._db)
            row = conn.execute(
                'SELECT fitness, depth, nodes FROM expressions'
                ' WHERE expr_hash=? AND fitness_hash=?',
                (eh, self._hash),
            ).fetchone()
            conn.close()
            if row:
                result = (row[0], row[1], row[2])
                self._mem[key] = result
                return result
        return None

    def put(self, key: str, value: Tuple[float, int, int]):
        self._mem[key] = value

    def load(self) -> int:
        """[重构] 2026-07-08 懒加载模式下无需预加载，返回 0"""
        return 0

    def save(self):
        """将内存缓存增量写入 SQLite (INSERT OR REPLACE 批量事务)"""
        if not self._db:
            return
        # [优化] 2026-07-08 改用 executemany + 显式事务，避免逐条 INSERT 的 SQLite 事务开销
        conn = sqlite3.connect(self._db)
        data = []
        for expr, (fit, dep, nod) in self._mem.items():
            eh = hashlib.md5(expr.encode()).hexdigest()[:16]
            data.append((eh, expr, fit, dep, nod, self._hash))
        conn.executemany(
            'INSERT OR REPLACE INTO expressions'
            '(expr_hash, expression, fitness, depth, nodes, fitness_hash)'
            'VALUES(?,?,?,?,?,?)',
            data,
        )
        conn.commit()
        conn.close()