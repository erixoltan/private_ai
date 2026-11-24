"use strict";
// Everything shown comes from the local server. Model output is only ever
// inserted as text (textContent), never as HTML.

const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

let providers = [];
let current = null; // id of the conversation on screen, or null for a new one
let running = false;
let follower = null; // AbortController for the event stream being read
let lastBlock = null; // {type, node} of the block still receiving text

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
const post = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

// ---- transcript ----

function nearBottom() {
  const t = $("transcript");
  return t.scrollHeight - t.scrollTop - t.clientHeight < 80;
}
function append(node) {
  const stick = nearBottom();
  $("transcript").querySelector(".empty")?.remove();
  $("transcript").append(node);
  if (stick) node.scrollIntoView({ block: "end" });
}
function closeThinking() {
  if (lastBlock && lastBlock.type === "thinking") lastBlock.node.closest("details").open = false;
}
function addBlock(type, text, live) {
  let target;
  if (lastBlock && lastBlock.type === type && (type === "text" || type === "thinking")) {
    const stick = nearBottom();
    lastBlock.node.textContent += text;
    if (stick) lastBlock.node.scrollIntoView({ block: "end" });
    return;
  }
  closeThinking();
  if (type === "thinking") {
    const details = el("details", "thinking");
    details.open = Boolean(live);
    details.append(el("summary", "", "Thinking"));
    target = el("div", "", text);
    details.append(target);
    append(details);
  } else {
    target = el("div", "block " + type, type === "notice" ? `[${text}]` : text);
    if (type === "user") {
      const move = el("button", "move", "⇄");
      move.type = "button";
      move.title = "Move this question and its answer to another conversation";
      move.setAttribute("aria-label", move.title);
      move.onclick = () => chooseTarget(target);
      target.append(move);
    }
    append(target);
  }
  lastBlock = { type, node: target };
}
function addQuestion(conversationId, event) {
  closeThinking();
  lastBlock = null;
  const card = el("div", "ask");
  card.dataset.question = event.id;
  card.append(el("p", "", event.question));
  const choices = el("div", "choices");
  for (const [key, label] of Object.entries(event.choices)) {
    const button = el("button", "", label);
    button.type = "button";
    button.onclick = () =>
      post(`/api/conversations/${conversationId}/answer`, { id: event.id, answer: key }).catch(() => {});
    choices.append(button);
  }
  card.append(choices);
  card.choiceLabels = event.choices;
  append(card);
}
function settleQuestion(event) {
  const card = document.querySelector(`.ask[data-question="${event.id}"]`);
  if (!card) return;
  card.classList.add("settled");
  card.querySelector(".choices").replaceWith(el("div", "", "Answered: " + (card.choiceLabels[event.answer] || "no")));
}
function clearTranscript(message) {
  lastBlock = null;
  $("transcript").replaceChildren();
  if (message) $("transcript").append(el("p", "empty", message));
}

// ---- following a reply ----

function setRunning(value) {
  running = value;
  $("send").hidden = value;
  $("stop").hidden = !value;
}
function handle(conversationId, event) {
  if (event.type === "ask") addQuestion(conversationId, event);
  else if (event.type === "answered") settleQuestion(event);
  else if (event.type === "notice" || event.type === "error") addBlock(event.type, event.text);
  else addBlock(event.type, event.text, true);
}
async function follow(conversationId) {
  follower?.abort();
  const controller = (follower = new AbortController());
  setRunning(true);
  try {
    const response = await fetch(`/api/conversations/${conversationId}/events`, { signal: controller.signal });
    if (!response.ok) throw new Error("not running");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end;
      while ((end = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, end).trim();
        buffer = buffer.slice(end + 1);
        if (line && current === conversationId) handle(conversationId, JSON.parse(line));
      }
    }
  } catch (error) {
    if (controller.signal.aborted) return; // the user moved to another conversation
    addBlock("error", "Lost the connection to private-ai. Is it still running in the terminal?");
  }
  if (follower === controller) {
    follower = null;
    closeThinking();
    lastBlock = null;
    setRunning(false);
    refreshList();
    refreshResearch(); // the reply may have started some
  }
}

// ---- conversations ----

async function refreshList() {
  const state = await api("/api/state");
  providers = state.providers;
  const list = $("conversations");
  list.replaceChildren();
  for (const c of state.conversations) {
    const item = el("li", c.id === current ? "current" : "");
    item.dataset.id = c.id;
    if (c.color) item.dataset.color = c.color;
    if (c.id === current && !researchShown) showIdentity(c);
    item.append(el("span", "emoji", c.emoji || ""));
    const open = el("button", "open");
    open.type = "button";
    open.append(el("span", "title", c.title || "(untitled)"));
    open.append(el("span", "date", c.running ? "answering…" : new Date(c.updated * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })));
    open.onclick = () => openConversation(c.id);
    const remove = el("button", "delete", "×");
    remove.type = "button";
    remove.title = "Delete this conversation";
    remove.onclick = () => deleteConversation(c);
    item.append(open, remove);
    list.append(item);
  }
  return state;
}
// The open conversation's title, emoji and colour, shown above the transcript
// and around the message box so it is clear where a message is about to go.
function showIdentity(conversation) {
  $("title").textContent = conversation ? conversation.title || "(untitled)" : "New conversation";
  $("title-emoji").textContent = (conversation && conversation.emoji) || "";
  for (const node of [$("title-bar"), $("composer")]) {
    if (conversation && conversation.color) node.dataset.color = conversation.color;
    else delete node.dataset.color;
  }
  document.title = conversation && conversation.title ? conversation.title + " — private_ai" : "private_ai";
}
function setCurrent(id) {
  current = id;
  for (const item of $("conversations").children) item.classList.toggle("current", item.dataset.id === id);
  // Kept in the address so a reload comes back to the same conversation.
  history.replaceState(null, "", id ? "#" + id : location.pathname);
}
function newConversation() {
  follower?.abort();
  follower = null;
  showResearchView(false);
  setCurrent(null);
  showIdentity(null);
  setRunning(false);
  clearTranscript("Start a conversation. Everything is stored on this computer.");
  document.body.classList.remove("menu-open");
  refreshList();
  $("input").focus();
}
async function openConversation(id) {
  follower?.abort();
  follower = null;
  setRunning(false);
  showResearchView(false);
  let conversation;
  try {
    conversation = await api(`/api/conversations/${id}`);
  } catch (error) {
    setCurrent(null);
    showIdentity(null);
    clearTranscript(id === location.hash.slice(1) ? "" : error.message);
    if (!$("transcript").firstChild) clearTranscript("Start a conversation. Everything is stored on this computer.");
    return;
  }
  setCurrent(id);
  showIdentity(conversation);
  clearTranscript();
  for (const block of conversation.blocks) {
    addBlock(block.type, block.text);
    if (block.type !== "text" && block.type !== "thinking") lastBlock = null;
    if (block.type === "user") lastBlock = null;
  }
  closeThinking();
  lastBlock = null;
  if (conversation.provider) await selectProvider(conversation.provider, conversation.model);
  if (conversation.pending !== null) addBlock("user", conversation.pending);
  lastBlock = null;
  document.body.classList.remove("menu-open");
  refreshList();
  $("transcript").scrollTop = $("transcript").scrollHeight;
  if (conversation.running) follow(id);
}
async function deleteConversation(c) {
  if (!confirm(`Permanently delete "${c.title || "(untitled)"}"?`)) return;
  try {
    await api(`/api/conversations/${c.id}`, { method: "DELETE" });
  } catch (error) {
    alert(error.message);
    return;
  }
  if (c.id === current) newConversation();
  else refreshList();
}

// ---- moving an exchange ----

async function chooseTarget(userBlock) {
  if (running || !current) return;
  // The transcript shows every saved message, so the nth question on screen is the nth exchange.
  const exchange = [...document.querySelectorAll("#transcript > .user")].indexOf(userBlock);
  const source = current;
  const state = await api("/api/state");
  const list = $("targets");
  list.replaceChildren();
  const add = (emoji, title, color, target) => {
    const button = el("button");
    button.type = "button";
    if (color) button.dataset.color = color;
    button.append(el("span", "", emoji), el("span", "", title));
    button.onclick = async () => {
      $("move").close();
      try {
        const result = await post(`/api/conversations/${source}/move`, { exchange, target });
        if (result.source_deleted) await openConversation(result.target_id);
        else await openConversation(source);
      } catch (error) {
        addBlock("error", error.message);
        lastBlock = null;
      }
    };
    const item = el("li");
    item.append(button);
    list.append(item);
  };
  for (const c of state.conversations) {
    if (c.id !== source && !c.running) add(c.emoji || "", c.title || "(untitled)", c.color, c.id);
  }
  add("+", "A new conversation", null, null);
  $("move").showModal();
}

// ---- provider and model ----

function showPolicy() {
  const p = providers.find((x) => x.name === $("provider").value);
  const line = $("policy");
  line.replaceChildren();
  if (!p) return;
  if (p.local) {
    line.append(el("span", "badge local", "Local"), " — nothing leaves this machine.");
    return;
  }
  line.append(el("span", "badge cloud", "Cloud"), " — your messages are sent to " + p.name + ". ");
  if (p.policy_note) line.append(p.policy_note + " This is the provider's claim; private_ai cannot verify it. ");
  if (p.policy_url) {
    const link = el("a", "", "Read their policy");
    link.href = p.policy_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    line.append(link);
  }
}
async function loadModels(preferred) {
  const name = $("provider").value;
  const p = providers.find((x) => x.name === name);
  const select = $("model");
  const loading = el("option", "", "loading…");
  loading.value = ""; // sending now uses the provider's default model
  select.replaceChildren(loading);
  let models = [];
  try {
    models = (await api("/api/models?provider=" + encodeURIComponent(name))).models;
  } catch (error) {
    /* the provider may not offer a list; fall back to its default below */
  }
  if ($("provider").value !== name) return; // changed again while loading
  const chosen = preferred || p.model || models[0] || "";
  if (chosen && !models.includes(chosen)) models.unshift(chosen);
  select.replaceChildren();
  for (const m of models) select.append(el("option", "", m));
  select.value = chosen;
  if (!models.length) {
    const none = el("option", "", "(no models found)");
    none.value = "";
    select.append(none);
  }
}
async function selectProvider(name, model) {
  const p = providers.find((x) => x.name === name);
  if (!p || !p.ready) return;
  $("provider").value = name;
  showPolicy();
  await loadModels(model);
}
function fillProviders(state) {
  const select = $("provider");
  select.replaceChildren();
  for (const p of state.providers) {
    const option = el("option", "", p.name + (p.local ? " (local)" : "") + (p.ready ? "" : " — no API key"));
    option.value = p.name;
    option.disabled = !p.ready;
    select.append(option);
  }
  const preferred = state.providers.find((p) => p.name === state.default_provider && p.ready);
  select.value = (preferred || state.providers.find((p) => p.ready) || {}).name || "";
}

// ---- web search ----

function fillSearch(search) {
  const select = $("search");
  select.replaceChildren();
  for (const s of search.services) {
    const option = el("option", "", s.name + (s.local ? " (local)" : "") + (s.ready ? "" : " — no API key"));
    option.value = s.name;
    option.disabled = !s.ready;
    select.append(option);
  }
  select.value = search.using;
  const line = $("search-policy");
  line.replaceChildren();
  const s = search.services.find((x) => x.name === search.using);
  if (!s) return;
  line.append("Web search: you are asked before each query is sent. " + (s.note || "") + " ");
  if (s.url) {
    const link = el("a", "", "Read more");
    link.href = s.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    line.append(link);
  }
}
$("search").onchange = async () => {
  fillSearch(await post("/api/search", { name: $("search").value }));
};

// ---- research ----
// Jobs run on the server and are listed in the side panel under the
// conversations. Opening one puts its report where the transcript would be.

let researchShown = null; // id of the job on screen, or null while a conversation is
let researchTimer = null;
let researchDrawn = ""; // what the job on screen looked like when last drawn

function showResearchView(on) {
  $("research-view").hidden = !on;
  $("transcript").hidden = on;
  $("composer").hidden = on;
  if (!on) researchShown = null;
  for (const item of $("research-list").children) item.classList.toggle("current", on && item.dataset.id === researchShown);
}
async function openResearch(id) {
  follower?.abort();
  follower = null;
  setRunning(false);
  setCurrent(null);
  researchShown = id;
  researchDrawn = "";
  showResearchView(true);
  history.replaceState(null, "", "#research:" + id);
  document.body.classList.remove("menu-open");
  $("research-actions").replaceChildren();
  $("research-report").textContent = "";
  await refreshResearch();
}
async function drawResearch(id) {
  let job;
  try {
    job = await api("/api/research/" + id);
  } catch (error) {
    if (researchShown === id) newConversation();
    return;
  }
  if (researchShown !== id) return;
  showIdentity({ title: job.title || job.topic, emoji: job.emoji, color: job.color });
  $("research-question").textContent = job.title ? job.topic : "";
  const actions = $("research-actions");
  actions.replaceChildren();
  const button = (label, action) => {
    const b = el("button", "", label);
    b.type = "button";
    b.onclick = action;
    actions.append(b);
    return b;
  };
  if (job.status === "running") {
    button("Stop", async () => {
      await post(`/api/research/${id}/stop`);
      refreshResearch();
    });
  } else {
    if (job.report) button("Copy report", () => navigator.clipboard.writeText(job.report));
    if (job.report && job.outline) {
      // The plan is the thinking behind the report: its outline and the reasoning for each section.
      const toggle = button("Show the plan", () => {
        const showingPlan = toggle.textContent === "Show the report";
        $("research-report").textContent = showingPlan ? job.report : job.outline;
        toggle.textContent = showingPlan ? "Show the plan" : "Show the report";
      });
    }
    if (!job.report && job.gathered) {
      // The reading was done and only the writing failed: no searching is repeated.
      button("Write the report again", async () => {
        await post("/api/research", { retry: id, provider: $("provider").value, model: $("model").value });
        refreshResearch();
      });
    }
  }
  const progress = job.log.map((line) => "· " + line).join("\n");
  const failed = job.status === "failed" ? `\n\nThis research failed: ${job.error}` : "";
  $("research-report").textContent = job.report || progress + failed;
}
async function refreshResearch() {
  const data = await api("/api/research");
  $("research-allows").textContent = data.allows;
  const list = $("research-list");
  list.replaceChildren();
  if (!data.jobs.length) list.append(el("li", "none", "No research yet."));
  for (const job of data.jobs) {
    const item = el("li", job.id === researchShown ? "current" : "");
    item.dataset.id = job.id;
    if (job.color) item.dataset.color = job.color;
    item.append(el("span", "emoji", job.emoji || ""));
    const open = el("button", "open");
    open.type = "button";
    open.title = job.topic;
    open.append(el("span", "title", job.title));
    const when = new Date(job.created * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    const line = job.status === "running" ? job.last || "starting…" : job.status === "done" ? when : `${job.status} · ${when}`;
    open.append(el("span", "date " + job.status, line));
    open.onclick = () => openResearch(job.id);
    const remove = el("button", "delete", "×");
    remove.type = "button";
    remove.title = "Delete this research";
    remove.onclick = () => deleteResearch(job);
    item.append(open, remove);
    list.append(item);
  }
  const shown = data.jobs.find((job) => job.id === researchShown);
  if (researchShown && !shown) newConversation();
  // Drawn again only when something changed, so a plan being read is not swapped back to the report.
  const looks = shown ? [shown.status, shown.last, shown.title, shown.color].join("|") : "";
  if (shown && looks !== researchDrawn) {
    researchDrawn = looks;
    await drawResearch(shown.id);
  }
  clearTimeout(researchTimer);
  if (data.jobs.some((job) => job.status === "running")) researchTimer = setTimeout(refreshResearch, 2500);
}
async function deleteResearch(job) {
  if (!confirm(`Permanently delete the research "${job.title}"?`)) return;
  try {
    await api("/api/research/" + job.id, { method: "DELETE" });
  } catch (error) {
    alert(error.message);
    return;
  }
  if (job.id === researchShown) newConversation();
  else refreshResearch();
}
$("research-open").onclick = () => {
  $("research").showModal();
  refreshResearch();
};
$("research-close").onclick = () => $("research").close();
$("research-form").onsubmit = async (event) => {
  event.preventDefault();
  const topic = $("research-topic").value.trim();
  if (!topic) return;
  try {
    const started = await post("/api/research", { topic, provider: $("provider").value, model: $("model").value });
    $("research-topic").value = "";
    $("research").close();
    await openResearch(started.id);
  } catch (error) {
    $("research-allows").textContent = "Could not start: " + error.message;
  }
};

// ---- memory ----

async function showMemory(facts) {
  facts = facts || (await api("/api/memory")).facts;
  const list = $("facts");
  list.replaceChildren();
  if (!facts.length) list.append(el("li", "", "Nothing remembered yet."));
  for (const fact of facts) {
    const item = el("li");
    const remove = el("button", "", "Forget");
    remove.type = "button";
    remove.onclick = async () => showMemory((await post("/api/memory/forget", { fact })).facts);
    item.append(el("span", "", fact), remove);
    list.append(item);
  }
  if (!$("memory").open) $("memory").showModal();
}

// ---- message box size ----
// Grows with the text up to a quarter of the window, then scrolls. Once the
// user drags it to a size of their own, that size is kept until the page is reloaded.

let sizedByUser = false;
let autoHeight = null; // the height last set here, to tell a drag from our own change

function fitInput() {
  if (sizedByUser) return;
  const input = $("input");
  input.style.height = "auto";
  const borders = input.offsetHeight - input.clientHeight;
  const wanted = input.scrollHeight + borders;
  const limit = Math.max(input.offsetHeight, Math.floor(window.innerHeight * 0.25));
  input.style.height = Math.min(wanted, limit) + "px";
  input.style.overflowY = wanted > limit ? "auto" : "hidden";
  autoHeight = input.offsetHeight;
}
$("input").addEventListener("input", fitInput);
window.addEventListener("resize", fitInput);
// A height that did not come from fitInput can only be the user dragging the corner.
new ResizeObserver(() => {
  if (!sizedByUser && autoHeight !== null && Math.abs($("input").offsetHeight - autoHeight) > 2) {
    sizedByUser = true;
    $("input").style.overflowY = "auto";
  }
}).observe($("input"));

// ---- sending ----

async function send() {
  const text = $("input").value.trim();
  if (!text || running) return;
  setRunning(true);
  try {
    const result = await post("/api/chat", {
      conversation_id: current,
      provider: $("provider").value,
      model: $("model").value || null,
      text,
    });
    if (!current) showIdentity({ title: text.slice(0, 60) });
    setCurrent(result.conversation_id);
  } catch (error) {
    setRunning(false);
    addBlock("error", error.message);
    lastBlock = null;
    return;
  }
  $("input").value = "";
  fitInput();
  closeThinking();
  lastBlock = null;
  addBlock("user", text);
  lastBlock = null;
  $("transcript").scrollTop = $("transcript").scrollHeight;
  follow(current);
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  send();
});
$("input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    send();
  }
});
$("stop").onclick = () => current && post(`/api/conversations/${current}/stop`).catch(() => {});
$("new").onclick = newConversation;
$("menu").onclick = () => document.body.classList.toggle("menu-open");
$("memory-open").onclick = () => showMemory();
$("memory-close").onclick = () => $("memory").close();
$("move-close").onclick = () => $("move").close();
$("provider").onchange = () => {
  showPolicy();
  loadModels();
};

(async function start() {
  clearTranscript("Start a conversation. Everything is stored on this computer.");
  showIdentity(null);
  const state = await refreshList();
  fillProviders(state);
  fillSearch(state.search);
  showPolicy();
  fitInput();
  await loadModels();
  await refreshResearch();
  if (location.hash.startsWith("#research:")) await openResearch(location.hash.slice(10));
  else if (location.hash.length > 1) await openConversation(location.hash.slice(1));
  $("input").focus();
})();
