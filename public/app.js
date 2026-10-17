"use strict";

(() => {
    const { t } = window.I18N;

    const MAX_CHARS = 2000;
    // Generous: the first request after a cold start also loads the model.
    const REQUEST_TIMEOUT_MS = 60000;
    const READY_POLL_INTERVAL_MS = 5000;
    const READY_POLL_ATTEMPTS = 24;
    const API_BASE = (document.querySelector('meta[name="api-base"]')?.content ?? "").replace(/\/$/, "");

    const EXAMPLES = [
        { key: "examples.prize", text: "Congratulations! You've won a $1000 gift card. Claim it now by replying with your PIN." },
        { key: "examples.bank", text: "URGENT: Your bank account has been suspended. Verify your details at http://secure-bank-verify.co" },
        { key: "examples.lunch", text: "Hey, are we still on for lunch tomorrow at 12?" },
        { key: "examples.dentist", text: "Reminder: your dentist appointment is on Monday at 9:30am." },
    ];

    const $ = (id) => document.getElementById(id);
    const els = {
        form: $("analyzeForm"),
        message: $("message"),
        counter: $("counter"),
        analyzeButton: $("analyzeButton"),
        clearButton: $("clearButton"),
        retryButton: $("retryButton"),
        examples: $("examples"),
        result: $("result"),
        errorText: $("errorText"),
        verdictText: $("verdictText"),
        verdictHelp: $("verdictHelp"),
        meter: $("meter"),
        scoreValue: $("scoreValue"),
        thresholdValue: $("thresholdValue"),
        latencyValue: $("latencyValue"),
        modelValue: $("modelValue"),
        status: $("status"),
        statusText: $("statusText"),
        langToggle: $("langToggle"),
    };

    // Kept so the current view can be re-rendered when the language changes.
    const view = { result: null, errorKey: null, statusKind: "pending" };
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let tickerFrame = 0;
    let inFlight = false;
    let polling = false;

    class ApiError extends Error {
        constructor(status) {
            super(`HTTP ${status}`);
            this.status = status;
        }
    }

    async function request(path, options = {}) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
        const headers = { Accept: "application/json" };
        if (options.body) headers["Content-Type"] = "application/json";
        try {
            const response = await fetch(`${API_BASE}${path}`, { ...options, headers, signal: controller.signal });
            if (!response.ok) throw new ApiError(response.status);
            return await response.json();
        } finally {
            clearTimeout(timer);
        }
    }

    function errorKeyFor(error) {
        if (error instanceof ApiError) {
            if (error.status === 422) return "error.invalid";
            if (error.status === 503) return "error.warming";
            if (error.status === 429) return "error.rateLimited";
            return "error.server";
        }
        return error.name === "AbortError" ? "error.timeout" : "error.network";
    }

    const formatPercent = (value) =>
        new Intl.NumberFormat(window.I18N.lang, { style: "percent", maximumFractionDigits: 1 }).format(value);

    // Counts the score up from 0, like a number ticker; instant with reduced motion.
    function animateScore(target, animate) {
        cancelAnimationFrame(tickerFrame);
        if (!animate || reducedMotion.matches) {
            els.scoreValue.textContent = formatPercent(target);
            return;
        }
        const duration = 650;
        const start = performance.now();
        const step = (now) => {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - (1 - progress) ** 3;
            els.scoreValue.textContent = formatPercent(target * eased);
            if (progress < 1) tickerFrame = requestAnimationFrame(step);
        };
        tickerFrame = requestAnimationFrame(step);
    }

    function autoResize() {
        els.message.style.height = "auto";
        els.message.style.height = `${els.message.scrollHeight}px`;
    }

    function setState(state) {
        els.result.dataset.state = state;
    }

    function renderResult(result, { animate = true } = {}) {
        view.result = result;
        view.errorKey = null;
        setState(result.is_fraud ? "fraud" : "legit");
        els.verdictText.textContent = t(result.is_fraud ? "verdict.fraud" : "verdict.legit");
        els.verdictHelp.textContent = t(result.is_fraud ? "verdict.fraudHelp" : "verdict.legitHelp");

        const percent = Math.round(result.fraud_score * 1000) / 10;
        els.meter.style.setProperty("--score", `${percent}%`);
        els.meter.style.setProperty("--threshold", `${result.threshold * 100}%`);
        els.meter.setAttribute("aria-valuenow", String(percent));
        els.meter.setAttribute("aria-valuetext", formatPercent(result.fraud_score));

        animateScore(result.fraud_score, animate);
        els.thresholdValue.textContent = formatPercent(result.threshold);
        els.latencyValue.textContent = `${Math.round(result.latency_ms)} ms`;
        els.modelValue.textContent = result.model_version;
    }

    function renderError(key) {
        view.errorKey = key;
        view.result = null;
        els.errorText.textContent = t(key);
        setState("error");
    }

    function setStatus(kind, modelVersion) {
        view.statusKind = kind;
        els.status.className = `status status--${kind}`;
        els.statusText.textContent = t(`status.${kind}`);
        if (modelVersion) els.status.title = modelVersion;
    }

    async function pollReady() {
        if (polling) return;
        polling = true;
        try {
            for (let attempt = 0; attempt < READY_POLL_ATTEMPTS; attempt += 1) {
                try {
                    const ready = await request("/ready");
                    setStatus("ready", ready.model_version);
                    return;
                } catch (error) {
                    if (!(error instanceof ApiError && error.status === 503)) break;
                    setStatus("warming");
                    await new Promise((resolve) => setTimeout(resolve, READY_POLL_INTERVAL_MS));
                }
            }
            setStatus("down");
        } finally {
            polling = false;
        }
    }

    async function analyze() {
        const message = els.message.value.trim();
        if (!message) {
            renderError("error.empty");
            els.message.focus();
            return;
        }
        if (inFlight) return;

        inFlight = true;
        els.form.setAttribute("aria-busy", "true");
        els.analyzeButton.disabled = true;
        els.analyzeButton.setAttribute("aria-busy", "true");
        setState("loading");
        try {
            const result = await request("/v1/predict", {
                method: "POST",
                body: JSON.stringify({ message }),
            });
            renderResult(result);
            setStatus("ready", result.model_version);
        } catch (error) {
            renderError(errorKeyFor(error));
            if (error instanceof ApiError && error.status === 503) pollReady();
        } finally {
            inFlight = false;
            els.form.removeAttribute("aria-busy");
            els.analyzeButton.disabled = false;
            els.analyzeButton.removeAttribute("aria-busy");
        }
    }

    function updateCounter() {
        const length = els.message.value.length;
        els.counter.textContent = `${length} / ${MAX_CHARS}`;
        els.counter.classList.toggle("over-limit", length >= MAX_CHARS);
    }

    function renderExamples() {
        els.examples.replaceChildren(
            ...EXAMPLES.map(({ key, text }) => {
                const item = document.createElement("li");
                const button = document.createElement("button");
                button.type = "button";
                button.className = "chip";
                button.textContent = t(key);
                button.title = text;
                button.addEventListener("click", () => {
                    els.message.value = text;
                    updateCounter();
                    autoResize();
                    analyze();
                });
                item.append(button);
                return item;
            }),
        );
    }

    function renderLanguageToggle() {
        const languages = window.I18N.languages();
        els.langToggle.hidden = languages.length < 2;
        const next = languages[(languages.indexOf(window.I18N.lang) + 1) % languages.length];
        els.langToggle.textContent = next.toUpperCase();
        els.langToggle.setAttribute("aria-label", window.I18N.messages[next].name);
        els.langToggle.dataset.next = next;
    }

    function rerender() {
        window.I18N.apply();
        renderExamples();
        renderLanguageToggle();
        setStatus(view.statusKind);
        if (view.result) renderResult(view.result, { animate: false });
        if (view.errorKey) renderError(view.errorKey);
    }

    function setLanguage(lang) {
        window.I18N.setLang(lang);
        rerender();
    }

    els.form.addEventListener("submit", (event) => {
        event.preventDefault();
        analyze();
    });
    els.message.addEventListener("input", () => {
        updateCounter();
        autoResize();
    });
    els.message.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            els.form.requestSubmit();
        }
    });
    els.clearButton.addEventListener("click", () => {
        els.message.value = "";
        updateCounter();
        autoResize();
        view.result = null;
        view.errorKey = null;
        setState("idle");
        els.message.focus();
    });
    els.retryButton.addEventListener("click", analyze);
    els.langToggle.addEventListener("click", () => setLanguage(els.langToggle.dataset.next));

    window.SmsFraudApp = { setLanguage };
    rerender();
    updateCounter();
    pollReady();
})();
