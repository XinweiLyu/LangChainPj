"""
从 Milvus 读取 RAG 集合中的数据，并在终端打印（标量字段；不打印 embedding 向量）。

默认使用 RAG/.env 中的 MILVUS_HOST、MILVUS_PORT、COLLECTION_NAME（一般为 agent_rag）。

运行（请在 RAG 目录下执行，保证能加载 .env）:
  python test/milvus_read_contents.py
  python test/milvus_read_contents.py -c agent_rag
  python test/milvus_read_contents.py --all-collections

注意：若用 pytest 跑本文件，默认会捕获 stdout，看不到下面打印；请直接运行上面的 python 命令。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from pymilvus import Collection, DataType, connections, utility

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

ALIAS = os.getenv("MILVUS_ALIAS", "default")
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = str(os.getenv("MILVUS_PORT", "19530"))
DEFAULT_COLLECTION = os.getenv("COLLECTION_NAME", "agent_rag")
DEFAULT_QUERY_LIMIT = 500


def _p(*args: Any, **kwargs: Any) -> None:
    """打印并立即刷新，避免在部分 IDE/管道里看不到输出。"""
    print(*args, **kwargs, flush=True)


def _ensure_connected() -> None:
    try:
        if connections.get_connection_addr(ALIAS):
            utility.list_collections(using=ALIAS, timeout=3)
            return
    except Exception:
        pass
    connections.connect(alias=ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)


def scalar_output_fields(collection: Collection) -> List[str]:
    skip = {
        DataType.FLOAT_VECTOR,
        DataType.FLOAT16_VECTOR,
        DataType.BFLOAT16_VECTOR,
        DataType.BINARY_VECTOR,
        DataType.SPARSE_FLOAT_VECTOR,
    }
    return [f.name for f in collection.schema.fields if f.dtype not in skip]


def guess_query_expr(collection: Collection) -> str:
    pk = collection.primary_field
    if pk is None:
        raise ValueError("集合无主键，请使用 --expr 指定 query 条件")
    name = pk.name
    dt = pk.dtype
    if dt in (DataType.INT64, DataType.INT32, DataType.INT16, DataType.INT8):
        return f"{name} >= 0"
    if dt == DataType.VARCHAR:
        return f'{name} != ""'
    raise ValueError(f"未支持的主键类型 {dt}，请使用 --expr")


def fetch_collection_rows(
    collection_name: str,
    *,
    limit: int = DEFAULT_QUERY_LIMIT,
    expr: Optional[str] = None,
    output_fields: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    _ensure_connected()
    if not utility.has_collection(collection_name, using=ALIAS):
        raise ValueError(f"集合不存在: {collection_name}")

    col = Collection(collection_name, using=ALIAS)
    col.load()
    fields = list(output_fields) if output_fields else scalar_output_fields(col)
    if not fields:
        raise ValueError("没有可输出的标量字段")
    qexpr = expr if expr is not None else guess_query_expr(col)
    return col.query(expr=qexpr, output_fields=fields, limit=limit)


def print_rows_human_readable(rows: List[Dict[str, Any]], collection_name: str) -> None:
    """把 query 到的每一行按「读到的正文」形式打印出来。"""
    _p()
    _p("=" * 60)
    _p(f"从 Milvus 集合 [{collection_name}] 读到的记录共 {len(rows)} 条")
    _p("=" * 60)

    for i, row in enumerate(rows, 1):
        _p()
        _p(f"---------- 第 {i} 条 ----------")
        for key, val in row.items():
            if val is None:
                continue
            if key == "text" or key in ("langchain_text", "page_content"):
                _p(f"【{key}】")
                _p(str(val))
            else:
                _p(f"{key}: {val}")
        _p(f"---------- 第 {i} 条（JSON 备份）----------")
        _p(json.dumps(dict(row), ensure_ascii=False))


def print_collection_dump(
    collection_name: str, *, limit: int, expr: Optional[str], json_only: bool
) -> None:
    _ensure_connected()
    if not utility.has_collection(collection_name, using=ALIAS):
        _p(f"集合不存在: {collection_name}", file=sys.stderr)
        return

    col = Collection(collection_name, using=ALIAS)
    _p()
    _p(f"=== 集合元信息: {collection_name} ===")
    _p(f"描述: {col.description or '(无)'}")
    _p(f"实体数(约): {col.num_entities}")
    _p("字段列表:")
    for f in col.schema.fields:
        extra = ""
        if f.dtype in (
            DataType.FLOAT_VECTOR,
            DataType.FLOAT16_VECTOR,
            DataType.BFLOAT16_VECTOR,
        ):
            extra = f", dim={f.params.get('dim')}"
        _p(f"  - {f.name}: {f.dtype}{extra}  primary={f.is_primary}")

    rows = fetch_collection_rows(collection_name, limit=limit, expr=expr)
    _p()
    _p(f"--- query 实际返回 {len(rows)} 行（limit={limit}）---")

    if json_only:
        for row in rows:
            _p(json.dumps(dict(row), ensure_ascii=False))
            _p()
    else:
        print_rows_human_readable(rows, collection_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 Milvus RAG 集合并打印读到的内容")
    parser.add_argument("-c", "--collection", default=None, help="集合名，默认 COLLECTION_NAME")
    parser.add_argument("--all-collections", action="store_true", help="打印所有集合")
    parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT)
    parser.add_argument("--expr", default=None, help="自定义 query 布尔表达式")
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="只输出每行 JSON，不输出【正文】分段格式",
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")

    _p("正在连接 Milvus …", f"{MILVUS_HOST}:{MILVUS_PORT}", f"alias={ALIAS}")
    try:
        _ensure_connected()
    except Exception as e:
        _p(f"连接 Milvus 失败: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.all_collections:
            names = utility.list_collections(using=ALIAS)
            if not names:
                _p("当前没有任何集合。")
                return
            for name in sorted(names):
                print_collection_dump(
                    name, limit=args.limit, expr=args.expr, json_only=args.json_only
                )
        else:
            name = args.collection or DEFAULT_COLLECTION
            _p(f"即将读取集合: {name}（可在 .env 中修改 COLLECTION_NAME）")
            print_collection_dump(
                name, limit=args.limit, expr=args.expr, json_only=args.json_only
            )
        _p()
        _p("读取并打印完毕。")
    finally:
        try:
            connections.disconnect(ALIAS)
        except Exception:
            pass


if __name__ == "__main__":
    main()
