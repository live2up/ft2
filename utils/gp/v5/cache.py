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
    """GP 适应度缓存 — 内存 + SQLite 双级

    用法:
        cache = FitnessCache(cache_db='/path/to/cache.db', fitness_hash='abc')
        cached = cache.get(key)
        if cached is None:
            cache.put(key, (fitness, depth, nodes))
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
        return self._mem.get(key)

    def put(self, key: str, value: Tuple[float, int, int]):
        self._mem[key] = value

    def load(self) -> int:
        """从 SQLite 加载当前配置指纹的缓存到内存"""
        if not self._db:
            return 0
        conn = sqlite3.connect(self._db)
        rows = conn.execute(
            'SELECT expression, fitness, depth, nodes FROM expressions WHERE fitness_hash=?',
            (self._hash,)
        ).fetchall()
        conn.close()
        for expr, fit, dep, nod in rows:
            self._mem[expr] = (fit, dep, nod)
        return len(rows)

    def save(self):
        """将内存缓存增量写入 SQLite (INSERT OR REPLACE)"""
        if not self._db:
            return
        conn = sqlite3.connect(self._db)
        for expr, (fit, dep, nod) in self._mem.items():
            eh = hashlib.md5(expr.encode()).hexdigest()[:16]
            conn.execute(
                'INSERT OR REPLACE INTO expressions'
                '(expr_hash, expression, fitness, depth, nodes, fitness_hash)'
                'VALUES(?,?,?,?,?,?)',
                (eh, expr, fit, dep, nod, self._hash),
            )
        conn.commit()
        conn.close()