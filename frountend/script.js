const analyzeBtn = document.getElementById("analyzeBtn");
const topAnalyzeBtn = document.getElementById("topAnalyzeBtn");
const textInput = document.getElementById("textInput");
const sourceUrl = document.getElementById("sourceUrl");
const pasteBtn = document.getElementById("pasteBtn");
const clearBtn = document.getElementById("clearBtn");
const demoBtn = document.getElementById("demoBtn");
const realDemoBtn = document.getElementById("realDemoBtn");
const scrollResultBtn = document.getElementById("scrollResultBtn");
const charCount = document.getElementById("charCount");
const resultPanel = document.getElementById("resultPanel");
const errorBox = document.getElementById("errorBox");
const historyList = document.getElementById("historyList");
const resultExplanation = document.getElementById("resultExplanation");
const factCheckStatus = document.getElementById("factCheckStatus");
const factCheckVerdict = document.getElementById("factCheckVerdict");
const factCheckSource = document.getElementById("factCheckSource");
const factCheckClaim = document.getElementById("factCheckClaim");
const reasonsList = document.getElementById("reasonsList");

const resultText = document.getElementById("resultText");
const resultBadge = document.getElementById("resultBadge");
const confidenceText = document.getElementById("confidenceText");
const loading = document.getElementById("loading");
const progressFill = document.getElementById("progressFill");

const apiUrlMeta = document.querySelector('meta[name="truthcheck-api-url"]');
const API_URL = apiUrlMeta?.content?.trim() || "http://127.0.0.1:5000/analyze";
const MAX_LENGTH = 5000;
const SAMPLE_TEXT = "Breaking news: A shocking claim says a secret global reset is underway, but no credible source or official record confirms it. The article uses dramatic language, vague attribution, and urgent calls to share immediately.";
const REAL_SAMPLE_TEXT = "The health ministry released an official report with named researchers, published methods, and public data links. Independent outlets verified the same numbers and quoted direct officials with dates and source records.";
let lastAnalysis = null;
let analysisHistory = [];

function resetSessionState() {
    analysisHistory = [];
    lastAnalysis = null;
    renderHistory();
}

function showError(message) {
    if (!message) {
        errorBox.hidden = true;
        errorBox.textContent = "";
        return;
    }

    errorBox.hidden = false;
    errorBox.textContent = message;
}

function setLoading(isLoading) {
    loading.classList.toggle("is-visible", isLoading);
    analyzeBtn.disabled = isLoading;
    topAnalyzeBtn.disabled = isLoading;
    pasteBtn.disabled = isLoading;
}

function scrollToResult() {
    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function getHistoryLabel(result) {
    if (result === "Fake" || result === "Strong Fake") {
        return "fake";
    }

    if (result === "Real" || result === "Strong Real") {
        return "real";
    }

    return "neutral";
}

function renderHistory() {
    if (!analysisHistory.length) {
        historyList.replaceChildren(document.createElement("p"));
        historyList.firstChild.className = "history-empty";
        historyList.firstChild.textContent = "No recent checks yet.";
        return;
    }

    const fragment = document.createDocumentFragment();

    analysisHistory.forEach((entry) => {
        const item = document.createElement("div");
        item.className = "history-item";

        const content = document.createElement("div");

        const title = document.createElement("strong");
        title.textContent = entry.result;

        const preview = document.createElement("span");
        preview.textContent = entry.preview;

        content.append(title, preview);

        const pill = document.createElement("span");
        pill.className = `history-pill ${getHistoryLabel(entry.result)}`;
        pill.textContent = `${entry.confidence}%`;

        item.append(content, pill);
        fragment.appendChild(item);
    });

    historyList.replaceChildren(fragment);
}

function addHistoryEntry(result, confidence, preview) {
    analysisHistory = [
        {
            result,
            confidence: Math.round(Number(confidence) || 0),
            preview
        },
        ...analysisHistory
    ].slice(0, 5);

    renderHistory();
}

function renderReasons(reasons, result) {
    const items = Array.isArray(reasons) && reasons.length ? reasons : ["No additional explanation available."];

    reasonsList.replaceChildren();

    items.forEach((reason) => {
        const item = document.createElement("li");
        item.textContent = reason;
        reasonsList.appendChild(item);
    });

    if (result === "Uncertain") {
        resultExplanation.textContent = "The model is not confident enough to give a stronger verdict, so it is marked as uncertain.";
    } else if (result === "Needs Verification") {
        resultExplanation.textContent = "The model and fact-check signal disagree, so the claim needs manual review.";
    } else if (result === "Strong Fake") {
        resultExplanation.textContent = "The model and fact-check signals both point toward a false claim.";
    } else if (result === "Strong Real") {
        resultExplanation.textContent = "The model and fact-check signals both point toward a credible claim.";
    } else {
        resultExplanation.textContent = "The verdict is based on the model confidence and the text signals it detected.";
    }
}

function renderFactCheck(factCheck) {
    if (!factCheck) {
        factCheckStatus.textContent = "Not checked";
        factCheckVerdict.textContent = "---";
        factCheckSource.textContent = "---";
        factCheckClaim.textContent = "No fact-check lookup was needed for this input.";
        return;
    }

    factCheckStatus.textContent = factCheck.status || "Unknown";
    factCheckVerdict.textContent = factCheck.verdict || "Unknown";
    factCheckSource.textContent = factCheck.source || "Unknown";
    factCheckClaim.textContent = factCheck.claim ? `Claim: ${factCheck.claim}` : "No claim text was returned.";
}

function setResultState(result) {
    resultBadge.className = "result-badge";

    if (result === "Fake" || result === "Strong Fake") {
        resultBadge.classList.add("fake");
        resultBadge.textContent = result;
        resultText.style.color = "#b42318";
    } else if (result === "Real" || result === "Strong Real") {
        resultBadge.classList.add("real");
        resultBadge.textContent = result;
        resultText.style.color = "#0f8a5f";
    } else if (result === "Needs Verification" || result === "Uncertain") {
        resultBadge.classList.add("neutral");
        resultBadge.textContent = result;
        resultText.style.color = "#b45309";
    } else {
        resultBadge.classList.add("neutral");
        resultBadge.textContent = "Pending";
        resultText.style.color = "inherit";
    }
}

function updateCharacterCount() {
    charCount.textContent = `${textInput.value.length} / ${MAX_LENGTH}`;
}

function getSubmissionText() {
    const text = textInput.value.trim();
    const url = sourceUrl.value.trim();

    if (text) {
        return text;
    }

    if (url) {
        return url;
    }

    return "";
}

async function runAnalysis() {
    const text = getSubmissionText();

    if (!text) {
        showError("Please enter text or an article URL.");
        return;
    }

    if (text.length > MAX_LENGTH) {
        showError(`Text too long (max ${MAX_LENGTH} characters)`);
        return;
    }

    showError("");
    setLoading(true);

    try {
        const controller = new AbortController();
        const timeoutId = window.setTimeout(() => controller.abort(), 30000);

        const res = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text }),
            signal: controller.signal
        });

        window.clearTimeout(timeoutId);

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || "Request failed");
        }

        resultText.textContent = data.result;
        confidenceText.textContent = `Confidence: ${data.confidence}%`;
        progressFill.style.width = `${Math.max(0, Math.min(100, Number(data.confidence) || 0))}%`;
        setResultState(data.result);
        renderFactCheck(data.fact_check);
        renderReasons(data.reasons, data.result);
        lastAnalysis = {
            result: data.result,
            confidence: Number(data.confidence) || 0,
            inputLength: (textInput.value.trim() || text).length,
            factCheck: data.fact_check || null
        };
        addHistoryEntry(data.result, data.confidence, text.slice(0, 72));
        scrollToResult();
    } catch (err) {
        console.error(err);
        showError(err.message || "Server not reachable");
        setResultState("Pending");
    } finally {
        setLoading(false);
    }
}

analyzeBtn.addEventListener("click", runAnalysis);
topAnalyzeBtn.addEventListener("click", () => {
    if (getSubmissionText()) {
        runAnalysis();
        return;
    }

    textInput.focus();
    textInput.scrollIntoView({ behavior: "smooth", block: "center" });
});

scrollResultBtn.addEventListener("click", scrollToResult);

demoBtn.addEventListener("click", () => {
    textInput.value = SAMPLE_TEXT;
    updateCharacterCount();
    textInput.focus();
    textInput.scrollIntoView({ behavior: "smooth", block: "center" });
});

realDemoBtn.addEventListener("click", () => {
    textInput.value = REAL_SAMPLE_TEXT;
    updateCharacterCount();
    textInput.focus();
    textInput.scrollIntoView({ behavior: "smooth", block: "center" });
});

clearBtn.addEventListener("click", () => {
    textInput.value = "";
    sourceUrl.value = "";
    resultPanel.dataset.state = "waiting";
    resultText.textContent = "---";
    confidenceText.textContent = "---";
    progressFill.style.width = "0%";
    setResultState("Pending");
    showError("");
    renderFactCheck(null);
    renderReasons(["Waiting for analysis."], "Pending");
    resetSessionState();
    updateCharacterCount();
    textInput.focus();
});

textInput.addEventListener("input", updateCharacterCount);
sourceUrl.addEventListener("input", updateCharacterCount);

pasteBtn.addEventListener("click", async () => {
    try {
        const clipboardText = await navigator.clipboard.readText();
        if (clipboardText) {
            textInput.value = clipboardText;
            updateCharacterCount();
            textInput.focus();
        }
    } catch (error) {
        showError("Clipboard access was blocked by the browser.");
        console.error(error);
    }
});

textInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !analyzeBtn.disabled) {
        runAnalysis();
    }
});

window.addEventListener("pageshow", resetSessionState);

setResultState("Pending");
confidenceText.textContent = "---";
progressFill.style.width = "0%";
renderFactCheck(null);
renderReasons(["Waiting for analysis."], "Pending");
resetSessionState();
updateCharacterCount();