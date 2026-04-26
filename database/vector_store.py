"""
Vector Store abstractions with LanceDB-backed implementations.

Includes:
- MemoryEntryVectorStore: multi-view indexing for MemoryEntry objects.
- RawContextVectorStore: entry-aligned raw context span storage.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Generic, TypeVar
import json
import lancedb
import pyarrow as pa
from models.memory_entry import MemoryEntry
from models.raw_context import RawContextEntry
from utils.embedding import EmbeddingModel
import config
import os

TEntry = TypeVar("TEntry")


class BaseLanceVectorStore(ABC, Generic[TEntry]):
    """Common LanceDB vector-store base class."""

    def __init__(
        self,
        db_path: str = None,
        embedding_model: EmbeddingModel = None,
        table_name: str = None,
        storage_options: Optional[Dict[str, Any]] = None
    ):
        self.db_path = db_path or config.LANCEDB_PATH
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name or config.MEMORY_TABLE_NAME
        self.table = None
        self._fts_initialized = False

        # Detect if using cloud storage (GCS, S3, Azure)
        self._is_cloud_storage = self.db_path.startswith(("gs://", "s3://", "az://"))

        # Connect to database
        if self._is_cloud_storage:
            self.db = lancedb.connect(self.db_path, storage_options=storage_options)
        else:
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)

        self._init_table()

    @abstractmethod
    def _build_schema(self) -> pa.Schema:
        """Return table schema."""

    @abstractmethod
    def _entry_text(self, entry: TEntry) -> str:
        """Return text to embed."""

    @abstractmethod
    def _entry_to_row(self, entry: TEntry, vector: list[float]) -> dict[str, Any]:
        """Convert entry to LanceDB row."""

    @abstractmethod
    def _results_to_entries(self, results: List[dict]) -> List[TEntry]:
        """Convert LanceDB rows to entries."""

    @property
    @abstractmethod
    def fts_column(self) -> str:
        """FTS column name."""

    def _init_table(self):
        """Initialize table schema and FTS index."""
        schema = self._build_schema()

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
            print(f"Created new table: {self.table_name}")
        else:
            self.table = self.db.open_table(self.table_name)
            print(f"Opened existing table: {self.table_name}")

    def _init_fts_index(self):
        """Initialize Full-Text Search index on lossless_restatement column."""
        if self._fts_initialized:
            return

        try:
            if self._is_cloud_storage:
                # Use native FTS for cloud storage (Tantivy only works with local filesystem)
                self.table.create_fts_index(
                    self.fts_column,
                    use_tantivy=False,
                    replace=True
                )
                print("FTS index created (native mode for cloud storage)")
            else:
                # Use Tantivy FTS for local storage (better performance)
                self.table.create_fts_index(
                    self.fts_column,
                    use_tantivy=True,
                    tokenizer_name="en_stem",
                    replace=True
                )
                print("FTS index created (Tantivy mode)")
            self._fts_initialized = True
        except Exception as e:
            print(f"FTS index creation skipped: {e}")

    def add_entries(self, entries: List[TEntry]):
        """Batch add entries."""
        if not entries:
            return

        texts = [self._entry_text(entry) for entry in entries]
        vectors = self.embedding_model.encode_documents(texts)

        data = [self._entry_to_row(entry, vector.tolist()) for entry, vector in zip(entries, vectors)]

        self.table.add(data)
        print(f"Added {len(entries)} entries")

        # Initialize FTS index after first data insertion
        if not self._fts_initialized:
            self._init_fts_index()

    def semantic_search(self, query: str, top_k: int = 5) -> List[TEntry]:
        """Dense vector similarity search."""
        try:
            if self.table.count_rows() == 0:
                return []

            query_vector = self.embedding_model.encode_single(query, is_query=True)
            results = self.table.search(query_vector.tolist()).limit(top_k).to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []

    def keyword_search(self, keywords: List[str], top_k: int = 3) -> List[TEntry]:
        """Lexical BM25 search."""
        try:
            if not keywords or self.table.count_rows() == 0:
                return []

            # LanceDB auto-detects string input as FTS query when FTS index exists
            query = " ".join(keywords)
            results = self.table.search(query).limit(top_k).to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during keyword search: {e}")
            return []

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Symbolic Layer Search - Metadata filtering (Section 3.1)
        r_k = E_sym(m_k), filters by timestamps, entities, persons
        """
        try:
            if self.table.count_rows() == 0:
                return []

            if not any([persons, timestamp_range, location, entities]):
                return []

            conditions = []

            if persons:
                values = ", ".join([f"'{p}'" for p in persons])
                conditions.append(f"array_has_any(persons, make_array({values}))")

            if location:
                safe_location = location.replace("'", "''")
                conditions.append(f"location LIKE '%{safe_location}%'")

            if entities:
                values = ", ".join([f"'{e}'" for e in entities])
                conditions.append(f"array_has_any(entities, make_array({values}))")

            if timestamp_range:
                start_time, end_time = timestamp_range
                conditions.append(f"timestamp >= '{start_time}' AND timestamp <= '{end_time}'")

            where_clause = " AND ".join(conditions)
            query = self.table.search().where(where_clause, prefilter=True)

            if top_k:
                query = query.limit(top_k)

            results = query.to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during structured search: {e}")
            return []

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all memory entries."""
        results = self.table.to_arrow().to_pylist()
        return self._results_to_entries(results)

    def optimize(self):
        """Optimize table after bulk insertions for better query performance."""
        self.table.optimize()
        print("Table optimized")

    def clear(self):
        """Clear all data and reinitialize table."""
        self.db.drop_table(self.table_name)
        self._fts_initialized = False
        self._init_table()
        print("Database cleared")


class MemoryEntryVectorStore(BaseLanceVectorStore[MemoryEntry]):
    """
    Multi-view indexing for memory units (Section 3.1).

    Three-layer indexing I(m_k):
    1. Semantic Layer: Dense embeddings for conceptual similarity
    2. Lexical Layer: Tantivy FTS for exact keyword matching
    3. Symbolic Layer: SQL-based metadata filtering
    """

    @property
    def fts_column(self) -> str:
        return "lossless_restatement"

    def _build_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("entry_id", pa.string()),
            pa.field("lossless_restatement", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("timestamp", pa.string()),
            pa.field("location", pa.string()),
            pa.field("persons", pa.list_(pa.string())),
            pa.field("entities", pa.list_(pa.string())),
            pa.field("topic", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

    def _entry_text(self, entry: MemoryEntry) -> str:
        return entry.lossless_restatement

    def _entry_to_row(self, entry: MemoryEntry, vector: list[float]) -> dict[str, Any]:
        return {
            "entry_id": entry.entry_id,
            "lossless_restatement": entry.lossless_restatement,
            "keywords": entry.keywords,
            "timestamp": entry.timestamp or "",
            "location": entry.location or "",
            "persons": entry.persons,
            "entities": entry.entities,
            "topic": entry.topic or "",
            "vector": vector,
        }

    def _results_to_entries(self, results: List[dict]) -> List[MemoryEntry]:
        entries = []
        for r in results:
            try:
                entries.append(MemoryEntry(
                    entry_id=r["entry_id"],
                    lossless_restatement=r["lossless_restatement"],
                    keywords=list(r.get("keywords") or []),
                    timestamp=r.get("timestamp") or None,
                    location=r.get("location") or None,
                    persons=list(r.get("persons") or []),
                    entities=list(r.get("entities") or []),
                    topic=r.get("topic") or None
                ))
            except Exception as e:
                print(f"Warning: Failed to parse result: {e}")
                continue
        return entries

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Symbolic metadata filtering."""
        try:
            if self.table.count_rows() == 0:
                return []

            if not any([persons, timestamp_range, location, entities]):
                return []

            conditions = []

            if persons:
                values = ", ".join([f"'{p}'" for p in persons])
                conditions.append(f"array_has_any(persons, make_array({values}))")

            if location:
                safe_location = location.replace("'", "''")
                conditions.append(f"location LIKE '%{safe_location}%'")

            if entities:
                values = ", ".join([f"'{e}'" for e in entities])
                conditions.append(f"array_has_any(entities, make_array({values}))")

            if timestamp_range:
                start_time, end_time = timestamp_range
                conditions.append(f"timestamp >= '{start_time}' AND timestamp <= '{end_time}'")

            where_clause = " AND ".join(conditions)
            query = self.table.search().where(where_clause, prefilter=True)

            if top_k:
                query = query.limit(top_k)

            results = query.to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during structured search: {e}")
            return []


class RawContextVectorStore(BaseLanceVectorStore[RawContextEntry]):
    """Vector store for entry-aligned raw context spans."""

    @property
    def fts_column(self) -> str:
        return "text"

    def _build_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("entry_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("metadata_json", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

    def _entry_text(self, entry: RawContextEntry) -> str:
        return entry.text

    def _entry_to_row(self, entry: RawContextEntry, vector: list[float]) -> dict[str, Any]:
        return {
            "entry_id": entry.entry_id,
            "text": entry.text,
            "metadata_json": json.dumps(entry.metadata, ensure_ascii=False),
            "vector": vector,
        }

    def _results_to_entries(self, results: List[dict]) -> List[RawContextEntry]:
        rows: list[RawContextEntry] = []
        for r in results:
            metadata_json = r.get("metadata_json") or "{}"
            try:
                metadata = json.loads(metadata_json)
            except Exception:
                metadata = {}
            rows.append(
                RawContextEntry(
                    entry_id=r["entry_id"],
                    text=r.get("text") or "",
                    metadata=metadata,
                )
            )
        return rows

    def upsert_entry(self, entry: RawContextEntry) -> None:
        """Upsert a raw context entry by entry_id."""
        safe_entry_id = entry.entry_id.replace("'", "''")
        self.table.delete(f"entry_id = '{safe_entry_id}'")
        self.add_entries([entry])


class VectorStore(MemoryEntryVectorStore):
    """Backward compatible alias for existing callers."""
