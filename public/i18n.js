"use strict";

// Minimal translation layer: elements carry data-i18n / data-i18n-placeholder keys.
window.I18N = (() => {
    const messages = {
        en: {
            name: "English",
            skip: "Skip to the message field",
            title: "SMS Fraud Detector",
            tagline: "Paste a text message to estimate the risk of fraud.",
            "status.pending": "Checking model…",
            "status.checking": "Checking model…",
            "status.ready": "Model ready",
            "status.warming": "Model warming up…",
            "status.down": "Service unavailable",
            "analyze.title": "Analyze a message",
            "analyze.label": "SMS message (English)",
            "analyze.placeholder": "e.g. Your parcel is on hold, pay the fee at…",
            "analyze.hint": "Ctrl + Enter to analyze",
            "analyze.submit": "Analyze",
            "analyze.clear": "Clear",
            "examples.title": "Try an example",
            "examples.prize": "Prize claim",
            "examples.bank": "Bank alert",
            "examples.lunch": "Lunch plan",
            "examples.dentist": "Appointment reminder",
            "result.title": "Result",
            "result.idle": "The verdict will appear here.",
            "result.loading": "Analyzing…",
            "result.retry": "Try again",
            "result.score": "Fraud score",
            "result.threshold": "Alert threshold",
            "result.latency": "Analysis time",
            "result.model": "Model",
            "verdict.fraud": "Likely fraud",
            "verdict.fraudHelp": "Do not click links, call numbers or reply with personal information.",
            "verdict.legit": "Looks legitimate",
            "verdict.legitHelp": "No fraud pattern detected. Stay careful with unexpected requests.",
            "error.empty": "Please enter a message to analyze.",
            "error.invalid": "The message must be between 1 and 2000 characters.",
            "error.warming": "The model is starting up. Try again in a few seconds.",
            "error.rateLimited": "Too many requests. Please wait a moment.",
            "error.timeout": "The analysis took too long. Please try again.",
            "error.server": "The analysis service returned an error.",
            "error.network": "Could not reach the analysis service.",
            "notice.title": "About this demo",
            "notice.body": "This is a proof of concept. The model (DistilBERT fine-tuned with LoRA on English SMS) can be wrong in both directions: never rely on it alone, and never share codes, passwords or bank details by SMS.",
            "footer.source": "Source code",
            "footer.api": "API documentation",
        },
    };

    let lang = "en";

    function t(key, vars = {}) {
        const template = messages[lang]?.[key] ?? messages.en[key] ?? key;
        return template.replace(/\{(\w+)\}/g, (_, name) => String(vars[name] ?? ""));
    }

    function apply(root = document) {
        root.querySelectorAll("[data-i18n]").forEach((el) => {
            el.textContent = t(el.dataset.i18n);
        });
        root.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
            el.placeholder = t(el.dataset.i18nPlaceholder);
        });
        document.documentElement.lang = lang;
    }

    return {
        messages,
        t,
        apply,
        get lang() {
            return lang;
        },
        setLang(next) {
            if (messages[next]) {
                lang = next;
                apply();
            }
        },
        languages: () => Object.keys(messages),
    };
})();
