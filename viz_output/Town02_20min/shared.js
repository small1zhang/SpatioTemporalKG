/* ═══════════════════════════════════════════════════════════════════
   shared.js — Common data loading utilities for STKG multi-page viz
   Exposes everything via the global `KG` object
   ═══════════════════════════════════════════════════════════════════ */

const TYPE_COLORS = {
  Vehicle:'#3b82f6', TrafficLight:'#ef4444', RoadElement:'#6b7280',
  EnvironmentSnapshot:'#22c55e', Maneuver:'#f59e0b', InteractionEvent:'#8b5cf6',
  SafetyViolation:'#ef4444', ResponsibilityAssignment:'#f97316',
  Junction:'#06b6d4', ScenarioSnapshot:'#a78bfa', Pedestrian:'#eab308',
  Rule:'#fbbf24', BehaviorRelation:'#ec4899'
};

const EDGE_COLORS = {
  in_lane:'#60a5fa', approaching:'#f87171', containsVehicle:'#a78bfa',
  containsTrafficLight:'#fb923c', containsRoad:'#9ca3af', hasEnvironment:'#4ade80',
  weather_context:'#86efac', lane_connects:'#93c5fd', adjacent_lane:'#bfdbfe',
  has_maneuver:'#fcd34d', has_interaction:'#c084fc', actor:'#e879f9', dst:'#f472b6',
  src:'#fb7185', standing_still:'#d1d5db', following:'#6ee7b7', manifestsAs:'#fbbf24',
  in_junction:'#c4b5fd', violates:'#ef4444', responsibleFor:'#f97316',
  opposite_direction:'#fb7185', ahead_of:'#60a5fa', overtaking:'#a78bfa',
  changing_lane:'#c4b5fd', yielding_to:'#86efac', approaching_pedestrian:'#f87171',
  nearby_pedestrian:'#fbbf24', blocked_view:'#9333ea', beside:'#6b7280',
  definedBy:'#94a3b8', supportedByEvidence:'#94a3b8'
};

const ANOM_COLORS = {
  sudd_brk:'#ef4444', sudd_stp:'#dc2626', avd_col:'#f97316',
  jun_ny:'#f59e0b', rev_drive:'#a855f7', ped_crs:'#eab308', obs_blk:'#6b7280'
};

const RSS_CODE_COLORS = {
  R4:'#f97316', R5:'#fb923c', R6:'#f59e0b', R7:'#fbbf24', R8:'#fde68a',
  R13a:'#a855f7', R13b:'#c084fc', R14a:'#9333ea', R14b:'#a78bfa',
  R15:'#d8b4fe', R16:'#e9d5ff', R17:'#f3e8ff',
  RSS_v_long:'#a855f7', RSS_v_lat:'#7c3aed'
};

const RSS_CODE_DESC = {
  R4:'对向来车未避让', R5:'红灯闯行', R6:'路口未让行', R7:'变道未让行',
  R8:'禁行方向行驶', R13a:'RSS 纵向安全距离', R13b:'RSS 纵向危险制动',
  R14a:'RSS 侧向危险距离', R14b:'RSS 侧向危险并道', R15:'RSS 路权优先级',
  R16:'RSS 行人近距', R17:'RSS 视线遮挡',
  RSS_v_long:'RSS 纵向', RSS_v_lat:'RSS 侧向'
};

const RSS_LAYER_COLORS = { RSS:'#a855f7', TrafficLaw:'#f97316', Unknown:'#6b7280' };

/* ── Global KG object ────────────────────────────────────────────── */
const KG = {
  summary: null,        // phase5_kg_summary.json
  shards: {},           // { shardIdx: { nodes:[], edges:[] } }
  anomalyLog: null,     // anomaly_log.json (array)
  vizStats: null,       // viz_stats.json
  meta: null,           // metadata.json
  allNodes: [],         // merged nodes across all loaded shards
  allEdges: [],         // merged edges across all loaded shards
  loadedShardIndices: new Set(),
  townName: 'Unknown',

  /* ── Data loading ──────────────────────────────────────────────── */

  /** Load all JSON data files, merging all graph shards */
  async loadAll() {
    try {
      const [sumR, metaR, vizR, anomR] = await Promise.all([
        fetch('phase5_kg_summary.json').then(r => r.json()),
        fetch('metadata.json').then(r => r.json()),
        fetch('viz_stats.json').then(r => r.json()),
        fetch('anomaly_log.json').then(r => r.json())
      ]);
      this.summary = sumR;
      this.meta = metaR;
      this.vizStats = vizR;
      this.anomalyLog = anomR;
      this.townName = metaR.town || metaR.host || 'Unknown';

      // Load all shards and merge
      for (const shardInfo of sumR.shards) {
        await this.loadShard(shardInfo.shard_idx);
      }
      this.mergeAllShards();
      return true;
    } catch (e) {
      console.error('KG: loadAll failed', e);
      return false;
    }
  },

  /** Load a single shard by index */
  async loadShard(idx) {
    if (this.loadedShardIndices.has(idx)) return;
    const shardInfo = this.summary?.shards?.find(s => s.shard_idx === idx);
    if (!shardInfo) return;
    const fname = `graph_${String(idx).padStart(4, '0')}_${shardInfo.frame_start}_${shardInfo.frame_end}.json`;
    try {
      const resp = await fetch(fname);
      const data = await resp.json();
      this.shards[idx] = data;
      this.loadedShardIndices.add(idx);
    } catch (e) {
      console.error(`KG: failed to load shard ${idx} (${fname}):`, e);
    }
  },

  /** Load shards within a frame range */
  async loadShardsForFrame(frame) {
    const needed = [];
    if (!this.summary) return;
    for (const si of this.summary.shards) {
      if (frame >= si.frame_start && frame <= si.frame_end && !this.loadedShardIndices.has(si.shard_idx)) {
        needed.push(si.shard_idx);
      }
    }
    for (const idx of needed) {
      await this.loadShard(idx);
    }
  },

  /** Load all shards currently selected in KG.shardSet */
  async loadMultiShard() {
    if (!this.summary) return;
    const targets = this.shardSet.size > 0
      ? [...this.shardSet]
      : this.summary.shards.map(s => s.shard_idx);
    for (const idx of targets) {
      await this.loadShard(idx);
    }
    this.mergeAllShards();
  },

  /** Merge all loaded shards into allNodes / allEdges */
  mergeAllShards() {
    const nodeMap = new Map();
    const edgeMap = new Map();
    for (const [idx, shard] of Object.entries(this.shards)) {
      for (const n of (shard.nodes || [])) {
        if (!nodeMap.has(n.id)) {
          nodeMap.set(n.id, { ...n, _shard_idx: parseInt(idx) });
        }
      }
      for (const e of (shard.edges || [])) {
        const ek = `${e.src_id}||${e.dst_id}||${e.type}||${e.frame_id ?? ''}`;
        if (!edgeMap.has(ek)) {
          edgeMap.set(ek, { ...e, _shard_idx: parseInt(idx) });
        }
      }
    }
    this.allNodes = [...nodeMap.values()];
    this.allEdges = [...edgeMap.values()];
  },

  /* ── Utility functions ─────────────────────────────────────────── */

  /** Get nodes that are active at a given frame */
  getNodesInFrame(frame) {
    return this.allNodes.filter(n => n.first_frame <= frame && n.last_frame >= frame);
  },

  /** Get a node by its ID (searches all loaded shards) */
  getNodeById(id) {
    return this.allNodes.find(n => n.id === id) || null;
  },

  /** Get shard objects that cover a given frame */
  getShardsForFrame(frame) {
    if (!this.summary) return [];
    return this.summary.shards.filter(si => frame >= si.frame_start && frame <= si.frame_end);
  },

  /** Get anomaly events at a specific frame */
  getAnomaliesInFrame(frame) {
    if (!this.anomalyLog) return [];
    return this.anomalyLog.filter(a => a.frame_id === frame);
  },

  /** Get anomalies within a range */
  getAnomaliesInRange(startFrame, endFrame) {
    if (!this.anomalyLog) return [];
    return this.anomalyLog.filter(a => a.frame_id >= startFrame && a.frame_id <= endFrame);
  },

  /* ── CSV export ────────────────────────────────────────────────── */

  /** Export nodes to CSV */
  exportNodesCSV(nodes) {
    const headers = ['id', 'type', 'first_frame', 'last_frame', 'entity_type', 'severity_max', 'entity_id', 'reason_tags'];
    const rows = nodes.map(n => {
      const a = n.attrs || {};
      return [
        n.id, n.type, n.first_frame, n.last_frame,
        a.entity_type || '', a.severity_max ?? '', a.entity_id || '',
        (a.reason_tags || []).join(';')
      ];
    });
    return this._csvEscape(headers) + '\n' + rows.map(r => this._csvEscape(r)).join('\n');
  },

  /** Export edges to CSV */
  exportEdgesCSV(edges) {
    const headers = ['src_id', 'dst_id', '关系类型', 'first_frame', 'last_frame', 'confidence', 'label', 'layer'];
    const rows = edges.map(e => {
      const a = e.attrs || {};
      return [
        e.src_id, e.dst_id, e.type, e.first_frame, e.last_frame,
        a.confidence ?? '', a.label || '', a.layer || ''
      ];
    });
    return this._csvEscape(headers) + '\n' + rows.map(r => this._csvEscape(r)).join('\n');
  },

  /** Download text as a file */
  downloadFile(content, filename) {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  },

  _csvEscape(arr) {
    return arr.map(v => {
      const s = String(v ?? '');
      if (s.includes(',') || s.includes('"') || s.includes('\n')) {
        return '"' + s.replace(/"/g, '""') + '"';
      }
      return s;
    }).join(',');
  },

  /** Generate a download link for a CSV string and click it */
  downloadCSV(csvString, filename) {
    const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  }
};

/* ── Convenience: make individual functions also accessible ──────── */
function getNodesInFrame(frame) { return KG.getNodesInFrame(frame); }
function getNodeById(id) { return KG.getNodeById(id); }
function getShardsForFrame(frame) { return KG.getShardsForFrame(frame); }
function getAnomaliesInFrame(frame) { return KG.getAnomaliesInFrame(frame); }