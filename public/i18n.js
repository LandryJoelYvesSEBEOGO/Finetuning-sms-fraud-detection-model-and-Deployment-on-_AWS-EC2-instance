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
            "hero.badge": "Proof of concept · DistilBERT + LoRA",
            "hero.title1": "Is this SMS",
            "hero.title2": "a scam?",
            "steps.title": "How it works",
            "steps.1.title": "Paste",
            "steps.1.body": "Paste an English SMS. Messages are analyzed on the fly, never stored or logged.",
            "steps.2.title": "Analyze",
            "steps.2.body": "DistilBERT fine-tuned with LoRA reads the message, served with ONNX Runtime in milliseconds.",
            "steps.3.title": "Decide",
            "steps.3.body": "You get a fraud score from 0 to 100%, compared with an alert threshold.",
            "footer.note": "Proof of concept, for demonstration only.",
            "footer.source": "Source code",
            "footer.api": "API documentation",
        },
        fr: {
            name: "Français",
            skip: "Aller au champ du message",
            title: "Détecteur de fraude SMS",
            tagline: "Collez un SMS pour estimer le risque de fraude.",
            "status.pending": "Vérification du modèle…",
            "status.checking": "Vérification du modèle…",
            "status.ready": "Modèle prêt",
            "status.warming": "Démarrage du modèle…",
            "status.down": "Service indisponible",
            "analyze.title": "Analyser un message",
            "analyze.label": "Message SMS (en anglais)",
            "analyze.placeholder": "ex. Your parcel is on hold, pay the fee at…",
            "analyze.hint": "Ctrl + Entrée pour analyser",
            "analyze.submit": "Analyser",
            "analyze.clear": "Effacer",
            "examples.title": "Essayer un exemple",
            "examples.prize": "Faux gain",
            "examples.bank": "Alerte bancaire",
            "examples.lunch": "Déjeuner",
            "examples.dentist": "Rappel de rendez-vous",
            "result.title": "Résultat",
            "result.idle": "Le verdict s'affichera ici.",
            "result.loading": "Analyse en cours…",
            "result.retry": "Réessayer",
            "result.score": "Score de fraude",
            "result.threshold": "Seuil d'alerte",
            "result.latency": "Durée d'analyse",
            "result.model": "Modèle",
            "verdict.fraud": "Fraude probable",
            "verdict.fraudHelp": "Ne cliquez sur aucun lien, n'appelez aucun numéro et ne communiquez aucune information personnelle.",
            "verdict.legit": "Message légitime",
            "verdict.legitHelp": "Aucun signe de fraude détecté. Restez vigilant face aux demandes inattendues.",
            "error.empty": "Veuillez saisir un message à analyser.",
            "error.invalid": "Le message doit contenir entre 1 et 2000 caractères.",
            "error.warming": "Le modèle démarre. Réessayez dans quelques secondes.",
            "error.rateLimited": "Trop de requêtes. Merci de patienter un instant.",
            "error.timeout": "L'analyse a pris trop de temps. Veuillez réessayer.",
            "error.server": "Le service d'analyse a renvoyé une erreur.",
            "error.network": "Impossible de joindre le service d'analyse.",
            "notice.title": "À propos de cette démo",
            "notice.body": "Ceci est une preuve de concept. Le modèle (DistilBERT affiné avec LoRA sur des SMS en anglais) peut se tromper dans les deux sens : ne vous y fiez jamais seul, et ne communiquez jamais de code, de mot de passe ou de coordonnées bancaires par SMS.",
            "hero.badge": "Preuve de concept · DistilBERT + LoRA",
            "hero.title1": "Ce SMS est-il",
            "hero.title2": "une arnaque ?",
            "steps.title": "Comment ça marche",
            "steps.1.title": "Collez",
            "steps.1.body": "Collez un SMS en anglais. Il est analysé à la volée, jamais stocké ni journalisé.",
            "steps.2.title": "Analyse",
            "steps.2.body": "DistilBERT affiné avec LoRA lit le message, servi par ONNX Runtime en quelques millisecondes.",
            "steps.3.title": "Décision",
            "steps.3.body": "Vous obtenez un score de fraude de 0 à 100 %, comparé à un seuil d'alerte.",
            "footer.note": "Preuve de concept, à des fins de démonstration uniquement.",
            "footer.source": "Code source",
            "footer.api": "Documentation de l'API",
        },
    };

    const STORAGE_KEY = "sms-fraud-lang";

    function initialLanguage() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved && messages[saved]) return saved;
        } catch {
            // Storage can be unavailable (private mode, blocked cookies).
        }
        const preferred = (navigator.languages ?? [navigator.language ?? "en"]).map((l) => l.slice(0, 2));
        return preferred.find((l) => messages[l]) ?? "en";
    }

    let lang = initialLanguage();

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
        document.title = t("title");
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
                try {
                    localStorage.setItem(STORAGE_KEY, next);
                } catch {
                    // Not persisted: the choice still applies to this page.
                }
                apply();
            }
        },
        languages: () => Object.keys(messages),
    };
})();
