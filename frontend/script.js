const API_BASE_URL = "http://127.0.0.1:8000";

const el = (id) => document.getElementById(id);

const textTab = el("textTab");
const urlTab = el("urlTab");
const textPane = el("textPane");
const urlPane = el("urlPane");

const newsText = el("newsText");
const newsUrl = el("newsUrl");
const newsImage = el("newsImage");
const imageHint = el("imageHint");
const checkBtn = el("checkBtn");
const clearBtn = el("clearBtn");

const loading = el("loading");
const errorBox = el("errorBox");
const resultEmpty = el("resultEmpty");
const resultBox = el("resultBox");
const resultTitle = el("resultTitle");
const confidencePill = el("confidencePill");
const analysisScore = el("analysisScore");
const scoreBar = el("scoreBar");
const reasonText = el("reasonText");
const rawText = el("rawText");
const apiUrlLabel = el("apiUrlLabel");

apiUrlLabel.textContent = `${API_BASE_URL}/analyze`;

let activeTab = "text";

function setTab(tab) {
  activeTab = tab;
  const isText = tab === "text";
  textTab.classList.toggle("active", isText);
  urlTab.classList.toggle("active", !isText);
  textPane.classList.toggle("hidden", !isText);
  urlPane.classList.toggle("hidden", isText);
}

function setLoading(isLoading) {
  loading.classList.toggle("hidden", !isLoading);
  checkBtn.disabled = isLoading;
  clearBtn.disabled = isLoading;
}

function showError(message) {
  errorBox.textContent = message || "";
  errorBox.classList.toggle("hidden", !message);
}

function parseConfidenceToPercent(confidence) {
  if (!confidence) return 60;
  const directNumber = String(confidence).match(/(\d+(\.\d+)?)/);
  if (directNumber) return Math.max(1, Math.min(100, Number(directNumber[1])));

  const value = String(confidence).toLowerCase();
  if (value.includes("high")) return 88;
  if (value.includes("medium")) return 72;
  if (value.includes("low")) return 55;
  return 60;
}

function showResult(result) {
  const label = String(result?.label || "UNKNOWN").toUpperCase();
  const confidencePct = parseConfidenceToPercent(result?.confidence);
  const reason = result?.reason || "No reason provided by model.";
  const raw = result?.raw || "";

  const likelyReal = label === "REAL";
  const score = likelyReal ? confidencePct : Math.max(1, 100 - confidencePct);

  resultEmpty.classList.add("hidden");
  resultBox.classList.remove("hidden");
  resultBox.classList.remove("real", "fake");
  resultBox.classList.add(likelyReal ? "real" : "fake");

  resultTitle.textContent = likelyReal ? "Real" : "Fake";
  confidencePill.textContent = `Confidence: ${confidencePct.toFixed(1)}%`;
  analysisScore.textContent = `${score.toFixed(1)}%`;
  scoreBar.style.width = `${score}%`;
  reasonText.textContent = reason;
  rawText.textContent = raw;
}

function clearAll() {
  newsText.value = "";
  newsUrl.value = "";
  newsImage.value = "";
  imageHint.textContent = "No image selected";
  showError("");
  resultBox.classList.add("hidden");
  resultBox.classList.remove("real", "fake");
  resultEmpty.classList.remove("hidden");
}

textTab.addEventListener("click", () => setTab("text"));
urlTab.addEventListener("click", () => setTab("url"));
clearBtn.addEventListener("click", clearAll);

newsImage.addEventListener("change", () => {
  const file = newsImage.files?.[0];
  imageHint.textContent = file ? `Selected: ${file.name}` : "No image selected";
});

checkBtn.addEventListener("click", async () => {
  showError("");

  const textValue = newsText.value.trim();
  const urlValue = newsUrl.value.trim();
  const file = newsImage.files?.[0] || null;

  let finalText = "";
  if (activeTab === "text") finalText = textValue;
  if (activeTab === "url" && urlValue) {
    finalText = `Please analyze this news URL and likely content credibility: ${urlValue}`;
  }

  if (!finalText && !file) {
    showError("Please provide text, URL, and/or image before analyzing.");
    return;
  }

  try {
    setLoading(true);
    const formData = new FormData();
    if (finalText) formData.append("text", finalText);
    if (file) formData.append("image", file);

    const res = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok || data?.ok === false) {
      throw new Error(data?.detail || data?.error || `Request failed (${res.status})`);
    }

    showResult(data.result || {});
  } catch (err) {
    showError(err?.message || "Something went wrong. Check backend logs.");
  } finally {
    setLoading(false);
  }
});

