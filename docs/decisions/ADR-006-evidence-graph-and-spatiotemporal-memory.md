# ADR-006 — Evidence Graph Architecture and Spatiotemporal Relationship Storage

**Date:** 2026-10-06  
**Status:** Approved Architectural Decision  
**Deciders:** Swastik Pandey  

---

## 1. Context & Problem Statement

VIDEX has successfully established Phases 1 through 6.1:
- Video Ingestion & Precise PTS Timing (Phase 1)
- Object Detection & Multi-Object Tracking (Phase 2)
- Text Detection & Recognition (OCR, Phase 3)
- Audio Extraction & Speech Recognition (ASR, Phase 4)
- Deterministic Temporal & Kinematic Event Engine (Phase 5)
- Semantic Intelligence Boundary, Evidence Validation, and VLM Routing (Phase 6 / 6.1)

Downstream consumers (Phase 8 Agentic Video Investigation and Phase 9 Interactive UI) require querying relationships between entities:
- Which events occurred in close temporal proximity to Event $E$?
- Which tracks or objects were co-present in Zone $Z$ when text $T$ was observed?
- What chain of evidence supports Semantic Event $S$?
- Which trajectory points and detections ground Track $K$?

A central architectural question arises: **Should VIDEX adopt a dedicated graph database (such as Neo4j or Memgraph), or should the Evidence Graph be structured as a derived spatiotemporal relationship projection over the canonical relational Evidence Store?**

---

## 2. Evaluation of Technological Options

We evaluated five candidate graph architectures against VIDEX's specific operational and query requirements:

### Option A: Dedicated Property Graph Database (Neo4j)
- **Strengths:** Industry standard for property graphs, mature Cypher query language, rich APOC graph algorithms, visualization tooling (Neo4j Bloom).
- **Weaknesses:** Substantial operational footprint (JVM service, high memory overhead), complex dual-write consistency requirements alongside PostgreSQL, separate transactional semantics, synchronization lag, and enterprise licensing hurdles.
- **Suitability for VIDEX:** High operational mismatch. VIDEX video analysis occurs per video file/stream; relationships are local (1–3 hops), and running a dedicated JVM cluster adds heavy friction to local developer environments and edge deployments.

### Option B: In-Memory Dedicated Graph Database (Memgraph)
- **Strengths:** C++ in-memory performance, Bolt protocol and Cypher compatibility, lightweight compared to Neo4j.
- **Weaknesses:** Still requires running an external standalone daemon/container, dual-write synchronization with relational database, memory footprint scales with all historical data.
- **Suitability for VIDEX:** Better operational fit than Neo4j, but still introduces distributed state management and dual-write divergence risks without need.

### Option C: PostgreSQL + Apache AGE (A Graph Extension)
- **Strengths:** Cypher query execution inside PostgreSQL, unified ACID transactions with relational tables, single database operational burden.
- **Weaknesses:** Native extension compilation requirements (requires custom C build or specialized container images), limited availability in managed database clouds (e.g., standard RDS / Cloud SQL without custom extensions), ecosystem immaturity compared to core Postgres.
- **Suitability for VIDEX:** Viable future upgrade path if complex multi-hop graph queries become necessary inside PostgreSQL, but premature as a day-1 dependency.

### Option D: PostgreSQL Native (Relational Foreign Keys + Recursive CTEs) + In-Memory Projection
- **Strengths:** Zero additional operational dependencies; SQLite support for local lightweight runs; PostgreSQL for production. Recursive queries (`WITH RECURSIVE`) easily handle the shallow 1–3 hop spatiotemporal traversals required by VIDEX. Completely deterministic, transactional, and reproducible.
- **Weaknesses:** Expressing arbitrary $N$-hop path finding in SQL can be more verbose than Cypher syntax.
- **Suitability for VIDEX:** Excellent fit. PostgreSQL remains the authoritative, immutable system of record. Graph relationships are projected deterministically in memory or materialized into indexed relation tables.

### Option E: PostgreSQL + pgvector
- **Strengths:** Native vector embedding similarity search directly adjacent to relational metadata and evidence records. Ideal for semantic search over transcript embeddings and visual feature embeddings.
- **Weaknesses:** Does not solve structural graph traversal by itself (complements rather than replaces relationship storage).
- **Suitability for VIDEX:** Strongly recommended as a complementary indexing capability for vector embeddings in Phase 8, rather than a standalone graph engine.

---

## 3. Comparative Matrix

| Evaluation Dimension | Dedicated Graph DB (Neo4j / Memgraph) | Apache AGE (PostgreSQL Extension) | Relational + In-Memory Graph Projection (Selected) |
|---|---|---|---|
| **System of Record** | Dual-write (Relational + Graph) | PostgreSQL | PostgreSQL (Single source of truth) |
| **Consistency / ACID** | Eventual / Dual-write hazard | ACID | Strict ACID |
| **Operational Burden** | High (standalone daemon/cluster) | Medium (custom C extension) | Zero (standard Python + Postgres/SQLite) |
| **Query Complexity for VIDEX** | Overkill (1–3 hops needed) | Moderate | Optimal (fast indexed lookups & BFS/DFS) |
| **Cross-Video Isolation** | Manual graph partitioning | Tenant/label tagging | Native parameter scoping by `video_id` |
| **Local / CI Execution** | Heavy (Docker / mock needed) | Heavy (custom Postgres) | Instantaneous (`InMemoryGraphStore`) |
| **Determinism & Auditability** | Prone to sync skew | Good | Absolute (pure projection from Evidence) |

---

## 4. Architectural Decision

1. **System of Record:** PostgreSQL (with SQLite for local development and CI) remains the single canonical system of record for all VIDEX entities (Videos, Frames, Detections, Tracks, OCR Observations, Audio/Transcripts, Events, and Evidence).
2. **Graph as Derived Projection:** The Evidence Graph is explicitly defined as a **deterministic relationship projection** over the canonical Evidence Store. No separate graph database engine is introduced in Phase 7.
3. **Core Graph Module (`videx.graph`):**
   - Provide typed node and edge domain models (`GraphNode`, `GraphEdge`).
   - Provide an abstract `GraphStore` interface with a high-performance, thread-safe `InMemoryGraphStore`.
   - Provide deterministic projection (`GraphProjection`) converting canonical pipeline outputs into graph structures without state leakage.
   - Enforce rigorous spatiotemporal and provenance integrity via `GraphIntegrityValidator`.
4. **Future Expansion Path:**
   - If future Phase 8 Agent workloads require persistent graph persistence beyond the lifecycle of an analysis run, relationship edges can be persisted into a standard indexed PostgreSQL relational table (`graph_edges`) or Apache AGE without altering the domain query API.

---

## 5. Consequences

### Positive
- Zero external services required: runs in lightweight CI, local developer environments, and serverless containers.
- Absolute determinism: identical evidence always produces identical graph topologies.
- No dual-write failure modes or eventual consistency anomalies.
- Provenance is first-class: every node and edge retains source references, timestamps, and derivation reasons.

### Negative / Trade-offs
- Arbitrary deep graph traversals (> 5 hops) over multi-million-node global graphs are not optimized; however, VIDEX queries are localized spatiotemporal subgraphs scoped by video and time intervals.
