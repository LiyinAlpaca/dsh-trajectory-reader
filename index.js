/**
 * 轨迹解读 · TRAJECTORY READER — server half.
 *
 * Activated by the web profile's Loader row (cordis.patch.yml). Registers one
 * exact web route the client bundle calls for LLM-powered per-round narrative
 * summaries:
 *
 *   GET  /plugin-api/trajectory-reader/summarize   → availability probe
 *   POST /plugin-api/trajectory-reader/summarize   → { provider?, model?, rounds: [...] }
 *
 * Each round is framed as JSON and summarized through the host `llm` service
 * (default route = the live agent default model selection). Uses no package
 * imports so it resolves inside any profile that links this package.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

/** Route path owned by this plugin (named routes win over the SPA fallback). */
const SUMMARIZE_PATH = "/plugin-api/trajectory-reader/summarize";
/** 设置读写与本地模型列表。 */
const SETTINGS_PATH = "/plugin-api/trajectory-reader/settings";
const MODELS_PATH = "/plugin-api/trajectory-reader/models";
/** 设置文件：插件自有数据目录（不写 DSH 本体）。 */
const DSH_HOME = process.env.DSH_HOME || path.join(os.homedir(), ".dsh");
const SETTINGS_FILE = path.join(DSH_HOME, "plugin-data", "trajectory-reader", "settings.json");

/** 默认设置（结构需与客户端 DEFAULT_SETTINGS 保持一致）。 */
const DEFAULT_SETTINGS = {
  autoSummarize: true,
  model: { mode: "local", provider: "ollama-local", model: "qwen35-4b-fast:latest" },
  triggers: { idleTextMinutes: 5, toolBurstCount: 10, slowToolMinutes: 3, cooldownMinutes: 5 },
};

function numOr(value, fallback) {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

/** 合并默认值（容忍缺字段/坏值），返回一份可直接下发的完整设置。 */
function mergeSettings(raw) {
  const base = {
    autoSummarize: DEFAULT_SETTINGS.autoSummarize,
    model: { ...DEFAULT_SETTINGS.model },
    triggers: { ...DEFAULT_SETTINGS.triggers },
  };
  if (!raw || typeof raw !== "object") return base;
  if (typeof raw.autoSummarize === "boolean") base.autoSummarize = raw.autoSummarize;
  if (raw.model && typeof raw.model === "object") {
    const m = raw.model;
    if (m.mode === "session" || m.mode === "local" || m.mode === "fixed") base.model.mode = m.mode;
    if (typeof m.provider === "string") base.model.provider = m.provider.slice(0, 120);
    if (typeof m.model === "string") base.model.model = m.model.slice(0, 120);
  }
  if (raw.triggers && typeof raw.triggers === "object") {
    const t = raw.triggers;
    base.triggers.idleTextMinutes = Math.min(1440, numOr(t.idleTextMinutes, base.triggers.idleTextMinutes));
    base.triggers.toolBurstCount = Math.min(1000, numOr(t.toolBurstCount, base.triggers.toolBurstCount));
    base.triggers.slowToolMinutes = Math.min(1440, numOr(t.slowToolMinutes, base.triggers.slowToolMinutes));
    base.triggers.cooldownMinutes = Math.min(1440, numOr(t.cooldownMinutes, base.triggers.cooldownMinutes));
  }
  return base;
}

function loadSettings() {
  try {
    return mergeSettings(JSON.parse(fs.readFileSync(SETTINGS_FILE, "utf8")));
  } catch {
    return mergeSettings(null);
  }
}

function saveSettings(settings) {
  fs.mkdirSync(path.dirname(SETTINGS_FILE), { recursive: true });
  fs.writeFileSync(SETTINGS_FILE, JSON.stringify(settings, null, 2), "utf8");
  return settings;
}
/** Max rounds accepted per request (each is one model call). */
const MAX_ROUNDS_PER_REQUEST = 12;
/** Per-round model-call budget (room for reasoning + the 400-char answer). */
const MAX_OUTPUT_TOKENS = 4000;
const CALL_TIMEOUT_MS = 120000;
/** 本地 Ollama 直连参数（必须走原生 /api/chat 才能关掉思考链）。 */
const LOCAL_PROVIDER = "ollama-local";
const LOCAL_BASE_URL = process.env.TRAJECTORY_READER_OLLAMA_URL || "http://127.0.0.1:11434";
const LOCAL_NUM_PREDICT = 900;
const LOCAL_TIMEOUT_MS = 300000;
/** Server-side re-cap of every string we forward to the model. */
const MAX_TEXT_CHARS = 4000;

/**
 * The summarizer system prompt: interpret ONE user round of a coding-agent
 * session — what the user wanted, how the assistant thought and executed,
 * and what it achieved — strictly grounded in the supplied material.
 */
const SYSTEM_PROMPT = [
  "你是 DeepSeek Harness（编程助手框架）的「会话轨迹解读员」。用户会给你一轮会话的原始材料，包括：该轮的用户消息原文、助手的回复与思考节选、按发生顺序排列的工具调用记录（工具名与参数摘要）、出现的错误与系统备注。",
  "你的任务：用中文把这一轮「用户要什么，助手是怎么思考、怎么一步步执行的，最终结果如何」写成一段连贯的解读，让没有看过原始会话的人也能明白助手做了什么、为什么这么做。",
  "",
  "必须遵守：",
  "1. 只依据材料中给出的信息解读，禁止编造材料里没有的文件、命令、结论或原因；材料若有截断（出现「…」），不要猜测被截断的内容。",
  "2. 输出使用以下 Markdown 结构（保留三个#的标题行，顺序不变）：",
  "",
  "### 用户需求",
  "（一两句话概括这一轮用户想要什么；若有多条消息或中途补充，请合并表述）",
  "",
  "### 助手思路",
  "（从回复、思考节选与动作顺序中提炼：助手选择了什么路线、为什么先做某事再做某事、遇到问题时如何调整）",
  "",
  "### 执行过程",
  "（按实际顺序用编号列表叙述：调研了什么、读/改/写了哪些文件、运行了哪些命令、委派了什么任务、遇到什么错误以及如何处理；同类动作可合并，但关键文件名、命令和报错要保留）",
  "",
  "### 结果",
  "（这一轮最终完成什么、主要改动落在哪里、还有什么未完成或失败的）",
  "",
  "3. 重点讲清「过程」与动作之间的因果关系（例如：先读 `A` 是为了确认 `B`，随后修改 `C` 来完成 `D`），不要只罗列工具名。",
  "4. 全文不超过 400 字；文件名、命令、错误信息一律用反引号包裹。",
  "5. 直接输出解读内容，不要输出任何开场白、结束语，也不要整段复述材料原文。"
].join("\n");

/**
 * 进行中进度解读的 system prompt：这一轮还没结束，用户只想知道现在在做什么、
 * 进度如何、做了哪些关键决策、有什么风险。
 */
const PROGRESS_PROMPT = [
  "你是 DeepSeek Harness（编程助手框架）的「任务进度解读员」。用户正在跑一个长任务，正在做别的事，需要快速知道助手此刻在干什么、进展到哪一步、做了哪些关键决策。",
  "你会收到「进行中」这一轮的材料：本轮用户需求、已进行时长、尚未完成的工具调用（含已运行时长）、最近按顺序的工具调用、最近的文字输出、错误与备注。注意：这一轮还没有结束，助手还没给出最终答复。",
  "",
  "必须遵守：",
  "1. 只依据材料，禁止编造材料中没有的文件、命令、结论或原因。",
  "2. 输出使用以下 Markdown 结构（保留标题行，顺序不变）：",
  "",
  "### 当前在做什么",
  "（一两句话说明此刻的动作）",
  "",
  "### 进度",
  "（做到哪一步了：已完成了哪些动作、还卡在哪；用到材料里的次数与时长）",
  "",
  "### 关键决策",
  "（从动作顺序里看出助手选了什么路线、为什么先做这件事；没有明显决策就写「暂无」）",
  "",
  "### 需要注意",
  "（耗时异常、反复重试、报错、疑似卡住等；没有就写「无」）",
  "",
  "3. 全文不超过 300 字；文件名、命令、报错用反引号包裹。",
  "4. 直接输出解读内容，不要开场白与结束语。",
].join("\n");

/** 按模式选择 system prompt 与用户消息（普通轮次 / 进行中进度）。 */
function promptFor(mode) {
  return mode === "progress" ? PROGRESS_PROMPT : SYSTEM_PROMPT;
}

function userContentFor(material, index, mode) {
  if (mode === "progress") {
    return `请解读第 ${index} 轮「进行中」的会话材料（JSON），这一轮还没有结束：\n${JSON.stringify(sanitize(material))}`;
  }
  return frameRound(material, index);
}

/** Assemble visible text from a raw chunk stream (no library imports). */
function assembleText(chunks) {
  const byIndex = new Map();
  const order = [];
  for (const chunk of chunks) {
    if (chunk.type === "block-start") {
      if (!byIndex.has(chunk.index)) { byIndex.set(chunk.index, ""); order.push(chunk.index); }
    } else if (chunk.type === "text-delta") {
      if (!byIndex.has(chunk.index)) { byIndex.set(chunk.index, ""); order.push(chunk.index); }
      byIndex.set(chunk.index, byIndex.get(chunk.index) + (chunk.text || ""));
    } else if (chunk.type === "block-end") {
      // Authoritative final block content replaces accumulated deltas.
      if (!byIndex.has(chunk.index)) order.push(chunk.index);
      byIndex.set(chunk.index, chunk.block && chunk.block.type === "text" ? chunk.block.text || "" : "");
    }
  }
  return order.sort((a, b) => a - b).map((i) => byIndex.get(i)).join("");
}

/** Assemble reasoning blocks (used when the model emitted no visible text). */
function assembleReasoning(chunks) {
  const byIndex = new Map();
  const order = [];
  for (const chunk of chunks) {
    if (chunk.type === "block-start") {
      if (!byIndex.has(chunk.index)) { byIndex.set(chunk.index, ""); order.push(chunk.index); }
    } else if (chunk.type === "reasoning-delta") {
      if (!byIndex.has(chunk.index)) { byIndex.set(chunk.index, ""); order.push(chunk.index); }
      byIndex.set(chunk.index, byIndex.get(chunk.index) + (chunk.text || ""));
    } else if (chunk.type === "block-end") {
      if (!byIndex.has(chunk.index)) order.push(chunk.index);
      byIndex.set(chunk.index, chunk.block && chunk.block.type === "reasoning" ? chunk.block.text || "" : "");
    }
  }
  return order.sort((a, b) => a - b).map((i) => byIndex.get(i)).join("");
}

/** Read and cap one JSON request body. */
function readBody(req, maxBytes) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (c) => {
      size += c.length;
      if (size > maxBytes) { reject(new Error("body too large")); req.destroy(); return; }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

/** Recursive sanitize: keep only plain JSON, cap every string. */
function sanitize(value) {
  if (typeof value === "string") return value.length > MAX_TEXT_CHARS ? value.slice(0, MAX_TEXT_CHARS) + "…" : value;
  if (Array.isArray(value)) return value.slice(0, 120).map(sanitize);
  if (value && typeof value === "object") {
    const out = {};
    for (const [k, v] of Object.entries(value).slice(0, 40)) out[k] = sanitize(v);
    return out;
  }
  return value;
}

/** Frame one round's material so user text cannot break structural delimiters. */
function frameRound(material, index) {
  return `请解读第 ${index} 轮会话材料（JSON）：\n${JSON.stringify(sanitize(material))}`;
}

/** Resolve the model route: explicit override → live default selection → fallback. */
function resolveRoute(ctx, body) {
  if (typeof body.provider === "string" && body.provider && typeof body.model === "string" && body.model) {
    return { provider: body.provider, model: body.model, source: body.provider === LOCAL_PROVIDER ? "local" : "session" };
  }
  const def = ctx.get("agentDefaultModel");
  if (def && typeof def.currentSelection === "function") {
    try {
      const sel = def.currentSelection();
      if (sel && typeof sel.provider === "string" && typeof sel.model === "string") {
        return { provider: sel.provider, model: sel.model, source: "default" };
      }
    } catch { /* fall through */ }
  }
  return { provider: "deepseek-official", model: "deepseek-v4-flash", source: "fallback" };
}

/** One-line diagnostic log (console + /tmp file); never throws. */
function logSummarize(sessionId, route, roundCount, mode) {
  if (mode === "progress") roundCount = roundCount + " (progress)";
  const line = `${new Date().toISOString()} session=${sessionId} route=${route.provider}/${route.model} source=${route.source} rounds=${roundCount}\n`;
  try {
    console.log(`[trajectory-reader] summarize ${line.trim()}`);
    import("node:fs").then((fs) => fs.appendFileSync("/tmp/tr-summarize.log", line)).catch(() => {});
  } catch { /* diagnostics must never break the route */ }
}

/**
 * 本地 Ollama 路由：原生 /api/chat + think:false。
 *
 * 带思考链的模型（qwen3.5 / qwen35 等）在 OpenAI 兼容的 /v1 上无法关闭 thinking，
 * token 预算会被 reasoning 吃光、正文为空；只有原生接口能稳定拿到正文。
 */
async function summarizeWithLocalOllama(route, material, index, outerSignal, systemPrompt, mode) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), LOCAL_TIMEOUT_MS);
  const onAbort = () => ac.abort();
  if (outerSignal) {
    if (outerSignal.aborted) ac.abort();
    else outerSignal.addEventListener("abort", onAbort, { once: true });
  }
  try {
    const res = await fetch(LOCAL_BASE_URL + "/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: route.model,
        stream: false,
        think: false,
        options: { num_predict: LOCAL_NUM_PREDICT, temperature: 0.2 },
        messages: [
          { role: "system", content: systemPrompt || SYSTEM_PROMPT },
          { role: "user", content: userContentFor(material, index, mode) },
        ],
      }),
      signal: ac.signal,
    });
    if (!res.ok) throw new Error("ollama HTTP " + res.status + "（本地 Ollama 未启动？）");
    const json = await res.json();
    const text = (json && json.message && json.message.content ? json.message.content : "").trim();
    if (!text) throw new Error("ollama 返回空正文（done_reason=" + (json && json.done_reason || "?") + "）");
    return text;
  } finally {
    clearTimeout(timer);
    if (outerSignal) outerSignal.removeEventListener("abort", onAbort);
  }
}

/** One round → one model call → { key, ok, text?, error? }. */
async function summarizeRound(ctx, route, key, material, index, signal, reasoningEffort, mode) {
  const systemPrompt = promptFor(mode);
  try {
    // 本地 Ollama：直接走原生接口，不经过宿主的 llm 服务。
    if (route.provider === LOCAL_PROVIDER) {
      const text = await summarizeWithLocalOllama(route, material, index, signal, systemPrompt, mode);
      return { key, ok: true, text };
    }
    const options = {
      provider: route.provider,
      model: route.model,
      system: systemPrompt,
      messages: [{ role: "user", content: [{ type: "text", text: userContentFor(material, index, mode) }] }],
      maxTokens: MAX_OUTPUT_TOKENS,
      signal
    };
    if (typeof reasoningEffort === "string" && reasoningEffort) options.reasoningEffort = reasoningEffort;
    const chunks = [];
    for await (const chunk of ctx.llm.stream(options)) {
      if (signal.aborted) break;
      chunks.push(chunk);
    }
    signal.throwIfAborted();
    const finish = chunks.find((c) => c.type === "finish");
    if (finish && finish.reason && finish.reason.kind === "error") {
      throw new Error(`model stream failed: ${finish.reason.error?.message || finish.reason.failure?.message || finish.reason.kind}`);
    }
    const text = assembleText(chunks).trim();
    if (!text) {
      // 模型只输出了思考过程（reasoning）而没给正式正文：直接展示思考内容，
      // 避免整轮失败；若连思考都没有，再报错并附上结束原因便于排查。
      const reasoning = assembleReasoning(chunks).trim();
      if (reasoning) {
        return { key, ok: true, text: "（模型本轮只输出了思考过程，未给出正式正文；以下为思考内容）\n\n" + reasoning.slice(0, 3000) };
      }
      const reasonKind = finish && finish.reason ? finish.reason.kind : "?";
      throw new Error(`model produced no text（finish=${reasonKind}）`);
    }
    return { key, ok: true, text };
  } catch (err) {
    return { key, ok: false, error: err && err.message ? err.message : String(err) };
  }
}

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, { "content-type": "application/json; charset=utf-8", "content-length": Buffer.byteLength(body) });
  res.end(body);
}

/** Cordis plugin entry: own the summarize route. */
export function apply(ctx) {
  ctx.webServer.register({
    kind: "exact",
    path: SUMMARIZE_PATH,
    handler: async (req, res) => {
      if (req.method === "GET" || req.method === "HEAD") {
        sendJson(res, 200, { ok: true, plugin: "trajectory-reader", path: SUMMARIZE_PATH });
        return;
      }
      if (req.method !== "POST") { sendJson(res, 405, { ok: false, error: "method not allowed" }); return; }
      let body;
      try {
        body = JSON.parse(await readBody(req, 4 * 1024 * 1024));
      } catch {
        sendJson(res, 400, { ok: false, error: "invalid JSON body" });
        return;
      }
      if (!body || typeof body !== "object" || !Array.isArray(body.rounds) || body.rounds.length === 0) {
        sendJson(res, 400, { ok: false, error: "body must be { provider?, model?, rounds: [{ key, material }] }" });
        return;
      }
      const rounds = body.rounds.slice(0, MAX_ROUNDS_PER_REQUEST);
      const route = resolveRoute(ctx, body);
      const mode = body.mode === "progress" ? "progress" : "round";
      const reasoningEffort = typeof body.reasoningEffort === "string" && body.reasoningEffort ? body.reasoningEffort : undefined;
      const sid = typeof body.sessionId === "string" && body.sessionId ? body.sessionId.slice(0, 96) : "?";
      logSummarize(sid, route, rounds.length, mode);
      const results = [];
      for (let i = 0; i < rounds.length; i++) {
        const r = rounds[i];
        const key = typeof r.key === "string" ? r.key.slice(0, 128) : String(i);
        const material = sanitize(r.material && typeof r.material === "object" ? r.material : {});
        const ac = new AbortController();
        const timer = setTimeout(() => ac.abort(), CALL_TIMEOUT_MS);
        try {
          results.push(await summarizeRound(ctx, route, key, material, i + 1, ac.signal, reasoningEffort, mode));
        } finally {
          clearTimeout(timer);
        }
      }
      sendJson(res, 200, { ok: true, route, results });
    }
  });

  // ── 设置读写：模型配置 + 自动触发阈值（持久化到插件自有数据目录）──
  ctx.webServer.register({
    kind: "exact",
    path: SETTINGS_PATH,
    handler: async (req, res) => {
      if (req.method === "GET" || req.method === "HEAD") {
        sendJson(res, 200, { ok: true, settings: loadSettings(), file: SETTINGS_FILE });
        return;
      }
      if (req.method !== "POST") { sendJson(res, 405, { ok: false, error: "method not allowed" }); return; }
      let body;
      try {
        body = JSON.parse(await readBody(req, 256 * 1024));
      } catch {
        sendJson(res, 400, { ok: false, error: "invalid JSON body" });
        return;
      }
      try {
        const next = mergeSettings(body && body.settings ? body.settings : body);
        saveSettings(next);
        sendJson(res, 200, { ok: true, settings: next });
      } catch (err) {
        sendJson(res, 500, { ok: false, error: "保存失败：" + (err && err.message ? err.message : String(err)) });
      }
    },
  });

  // ── 本地模型列表（设置页的下拉框数据源）──
  ctx.webServer.register({
    kind: "exact",
    path: MODELS_PATH,
    handler: async (req, res) => {
      if (req.method !== "GET" && req.method !== "HEAD") { sendJson(res, 405, { ok: false, error: "method not allowed" }); return; }
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), 5000);
      try {
        const upstream = await fetch(LOCAL_BASE_URL + "/api/tags", { signal: ac.signal });
        if (!upstream.ok) throw new Error("ollama HTTP " + upstream.status);
        const json = await upstream.json();
        const models = (json && Array.isArray(json.models) ? json.models : [])
          .map((m) => ({ name: m && m.name ? String(m.name) : "", size: m && typeof m.size === "number" ? m.size : 0 }))
          .filter((m) => m.name);
        sendJson(res, 200, { ok: true, provider: LOCAL_PROVIDER, models, endpoint: LOCAL_BASE_URL });
      } catch (err) {
        sendJson(res, 200, {
          ok: false,
          provider: LOCAL_PROVIDER,
          models: [],
          endpoint: LOCAL_BASE_URL,
          error: err && err.message ? err.message : String(err),
        });
      } finally {
        clearTimeout(timer);
      }
    },
  });
}

export const inject = ["llm", "webServer"];
export { SYSTEM_PROMPT, SUMMARIZE_PATH };
