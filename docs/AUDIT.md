# Audit technique — SMS Fraud Detection

_Date : 2026-10-14 · Périmètre : intégralité du dépôt (commit `19c9585`) + modèle Hugging Face `MrNWARMeilleur/sms_fraud_detection`_

## 1. Synthèse

| Domaine | État | Verdict |
|---|---|---|
| Modèle ML | Adaptateur LoRA DistilBERT publié sur HF, **aucun code d'entraînement ni métrique dans le repo** | 🔴 Non reproductible |
| Backend (FastAPI) | 1 endpoint, ~40 lignes, fonctionnel sur le chemin nominal | 🟠 Fragile |
| Frontend | Page HTML statique servie par un serveur Python de dev | 🟠 Minimal |
| Infra / Docker | 3 conteneurs (api, web, nginx), sans TLS, Python 3.8 EOL | 🟠 Démo uniquement |
| Sécurité | CORS `*` + credentials, pas d'auth, pas de rate limit, `/docs` public | 🔴 Non exposable |
| Qualité / Tests / CI | Aucun test automatisé, aucune CI, aucun linter | 🔴 Absent |
| Documentation | README ambitieux mais **en grande partie inexact** | 🟠 À réécrire |

**En bref** : le projet est une démo qui marche sur le chemin nominal. Le cœur du « POC », c'est-à-dire le fine-tuning et la preuve que le modèle détecte quelque chose, n'est pas dans le dépôt. Il faut le reconstruire en priorité, puis durcir l'API et le déploiement.

### Inventaire

```
app.py                  API FastAPI (chargement modèle + /predict)
test.py                 script manuel (pas un test)
requirements.txt        dépendances API (épinglées, 2023)
Dockerfile              image API multi-stage, python:3.8-slim
docker-compose.yml      api + web + nginx
nginx.conf              reverse proxy HTTP :80
interface-web/          index.html, styles.css, server.py (http.server), Dockerfile
.env                    AWS_ACCOUNT_ID / AWS_REGION — placeholders, non utilisés
*.png / *.jpg           illustrations README (3,4 Mo)
```

---

## 2. Modèle ML

### Ce qui a été vérifié sur Hugging Face
- `adapter_config.json` : LoRA `r=8`, `alpha=16`, `dropout=0.1`, cibles `q_lin`/`v_lin`, `task_type=SEQ_CLS`, `modules_to_save=["classifier","score"]`.
- La model card indique **PEFT 0.14.0** et `training_args.bin` référence `SaveStrategy`/`AcceleratorConfig`, ce qui suppose un entraînement avec transformers ≥ 4.4x, alors que l'API tourne avec `transformers==4.30.2` / `peft==0.4.0`.
  - J'ai vérifié que PEFT 0.4.0 sait charger le `safetensors` et ignore les clés de config inconnues : le chargement fonctionne, mais l'écart de versions entre l'entraînement et l'inférence n'est ni voulu ni testé.
- La taille de l'adaptateur (2,96 Mo) correspond à LoRA + `pre_classifier` + `classifier` (≈ 740k params fp32). Comme PEFT matche par suffixe, `pre_classifier` est bien sauvegardé : **la tête de classification est complète**, ce n'est pas un bug.
- Je n'ai **pas pu exécuter le modèle** dans cet environnement : l'accès réseau à `huggingface.co` est bloqué. La qualité réelle des prédictions n'est donc pas mesurée.

### Problèmes
| # | Sévérité | Constat |
|---|---|---|
| M1 | 🔴 | **Aucun code d'entraînement, dataset, split ni métrique** dans le repo, malgré le nom « Finetuning… ». La model card HF est le template vide. Impossible de prouver ou de reproduire la performance. |
| M2 | 🔴 | **Spam ≠ fraude.** L'exemple de `test.py` (« For sale - arsenal dartboard… ») provient du dataset *UCI SMS Spam Collection*. Le modèle est très probablement un classifieur spam/ham, présenté comme « fraude ». |
| M3 | 🔴 | **Langue** : `distilbert-base-uncased` est anglophone, alors que l'UI est en français. Des SMS d'arnaque français (colis, CPF, Ameli, banque…) seront mal classés. |
| M4 | 🟠 | Labels codés en dur (`LABEL_0` = non-fraude), sans `id2label`. Le score de confiance est jeté. |
| M5 | 🟠 | Révision HF non épinglée (`revision=`) : toute modification du repo HF change la prod silencieusement. |
| M6 | 🟡 | Modèle téléchargé **au démarrage du conteneur** (≈ 270 Mo), sans cache persistant. Démarrage lent, et le conteneur ne démarre plus si HF est indisponible. |
| M7 | 🟡 | `training_args.bin` est un pickle (ne jamais le charger côté serveur). |

---

## 3. Backend — `app.py`

| # | Sévérité | Ligne | Constat |
|---|---|---|---|
| B1 | 🔴 | `app.py:31` | Le pipeline est appelé **sans `truncation=True`** : un message de plus de 512 tokens provoque une erreur 500. |
| B2 | 🔴 | `app.py:30` | `async def` + inférence CPU synchrone : **la boucle d'événements est bloquée**. Les requêtes sont sérialisées et le healthcheck peut tomber sous charge. Il faut utiliser `def` (threadpool) ou un `run_in_threadpool` explicite. |
| B3 | 🔴 | `app.py:39` | `uvicorn.run(app, reload=True)` avec un objet (et non une chaîne d'import) : uvicorn 0.22 fait `sys.exit(1)`. **`python app.py` ne démarre pas.** |
| B4 | 🟠 | `app.py:11-18` | CORS `allow_origins=["*"]` + `allow_credentials=True` : combinaison invalide selon la spec et inutile, puisque nginx sert tout sur la même origine. |
| B5 | 🟠 | `app.py:26` | Aucune validation : chaîne vide acceptée, pas de longueur max, pas de limite de taille de corps. |
| B6 | 🟠 | `app.py:36` | La réponse est une **chaîne brute**, sans schéma, label ni score. C'est inexploitable par un client tiers et non versionné. |
| B7 | 🟠 | — | Pas d'endpoint `/health` ni `/ready` : le healthcheck interroge `/docs`, qui est en plus exposé publiquement. |
| B8 | 🟠 | — | Pas d'auth (clé API), pas de rate limiting. Sur un t2.medium, l'API est facile à saturer. |
| B9 | 🟡 | — | Pas de logs structurés, de métriques (latence, distribution des prédictions), ni de request-id. |
| B10 | 🟡 | `app.py:20-24` | Chargement du modèle au niveau module (effet de bord à l'import, ce qui complique les tests). Il vaut mieux le faire dans un `lifespan`. |
| B11 | 🟡 | `app.py:20` | Le tokenizer est pris sur le modèle de base, alors que le repo HF fournit le sien. C'est équivalent aujourd'hui, mais fragile. |

### Dépendances (`requirements.txt`)
- Stack 2023 : torch 2.0.1, transformers 4.30.2, fastapi 0.95.2 (pydantic v1, non épinglé), peft 0.4.0.
- `huggingface_hub>=0.14.1` est **non borné** alors que le reste est épinglé : risque de casse au rebuild.
- Pas de lockfile, pas de séparation dev/prod.

---

## 4. Frontend — `interface-web/`

| # | Sévérité | Constat |
|---|---|---|
| F1 | 🟠 | `index.html:78` lit `response.text()` sur une réponse JSON : l'utilisateur voit **les guillemets** (`"This message is a fraud"`). Le résultat s'affiche en anglais dans une UI française. |
| F2 | 🟠 | Aucun retour visuel : pas de code couleur fraude/légitime, pas de score, bouton non désactivé pendant l'analyse (double soumission possible). |
| F3 | 🟠 | `server.py` utilise `http.server.HTTPServer`, un serveur de dev **mono-thread**. Il sert tout le dossier, y compris `GET /server.py`. Le templating `${API_URL}` ne s'applique qu'à `/` : `/index.html` reçoit le littéral et l'appel API casse. |
| F4 | 🟡 | Conteneur `web` superflu : nginx peut servir directement les fichiers statiques. |
| F5 | 🟡 | Accessibilité : `textarea` sans `<label>`, pas de `aria-live` sur le résultat. |
| F6 | 🟡 | `console.log` des headers en production, footer « © 2024 », pas d'exemples ni de raccourci clavier. |
| ✅ | — | Le résultat est inséré via `textContent` : pas de XSS. |

---

## 5. Infrastructure & déploiement

| # | Sévérité | Fichier | Constat |
|---|---|---|---|
| I1 | 🔴 | `nginx.conf` | **HTTP uniquement**, pas de TLS. Le README propose `certbot --nginx` sur l'hôte, ce qui entre en conflit avec le nginx conteneurisé qui occupe déjà le port 80. |
| I2 | 🟠 | `docker-compose.yml:8-9,26-27` | Bind mounts `.:/app` et `./interface-web:/app` : le code de l'image est écrasé par celui de l'hôte. C'est pratique en dev, mais dangereux en prod (le `.env` est monté, et une modification sur l'hôte change la prod). |
| I3 | 🟠 | `Dockerfile` | `python:3.8-slim` : **Python 3.8 est EOL depuis oct. 2024**. Le builder installe gcc/g++/git inutilement (des wheels existent), puis copie tout `/usr/local/bin`. |
| I4 | 🟠 | `nginx.conf` | Pas de `client_max_body_size`, de `limit_req`, de headers de sécurité, de gzip ni de timeouts. |
| I5 | 🟠 | — | **Pas d'IaC** (Terraform/CDK), pas de CI/CD, pas de registry (ECR). Le déploiement se fait par `git clone` + build manuel sur EC2. |
| I6 | 🟡 | `Dockerfile:50` | `start-period=5s`, alors que le chargement du modèle prend bien plus. Compose le surcharge (120s), mais l'image seule est incohérente. |
| I7 | 🟡 | `.env` | Fichier versionné (placeholders uniquement, **aucun secret réel**, vérifié sur tout l'historique), non utilisé. `.gitignore` ne contient que `.qodo`. |
| I8 | 🟡 | `.dockerignore` | Contient des restes d'un autre projet (`Resume_AI_intern.pdf`, `Structured_OCR.ipynb`, `mlruns/`). |
| I9 | 🟡 | `docker-compose.yml:1` | `version:` est obsolète en Compose v2. |

---

## 6. Qualité, tests, documentation

- **Tests** : `test.py` est un script manuel qui suppose l'API lancée. Il n'y a ni pytest, ni test de contrat, ni test du modèle sur un jeu de référence.
- **CI** : aucune (pas de `.github/workflows`).
- **Lint / format / typage** : rien de configuré.
- **README : affirmations non vérifiables ou fausses**
  - les fichiers `docker-compose.prod.yml`, `.env.example` et `LICENSE` sont mentionnés mais **n'existent pas** ;
  - « Multilingual support (EN/FR) », « CORS configured for production », « 99.9% uptime », « < 500ms », « 100 simultaneous requests » : rien de cela n'est implémenté ni mesuré ;
  - VPC / NAT Gateway / CloudWatch : aucune trace dans le repo ;
  - « 60% des fraudes financières commencent par un SMS » : chiffre sans source ;
  - le crontab `docker-compose pull` ne sert à rien, car les images sont buildées localement ;
  - contacts placeholders, URL de clone fictive.
- Nom de fichier `project_Pipepline.jpg` (faute de frappe, 3,4 Mo dans l'historique git).

---

## 7. Feuille de route vers un POC déployable

### Décisions de cadrage (validées)
- **Cible** : détection de **fraude** par SMS (pas de spam générique).
- **Langue** : anglais (textes et dataset).
- **Hébergement** : **Vercel** (frontend statique + fonction Python serverless).

### Conséquence d'architecture : Vercel impose une inférence légère
Les fonctions Python Vercel ont une taille de bundle limitée, et PyTorch (plusieurs centaines de Mo) n'y tient pas.
L'inférence en production passera donc par **ONNX Runtime** (modèle LoRA fusionné, exporté en ONNX fp32 sans modification des poids, ≈ 265 Mo, téléchargé au démarrage dans `/tmp`).
PyTorch reste utilisé pour l'entraînement et l'export.

```
Vercel
├── frontend statique (HTML/CSS/JS)    → /
└── fonction Python FastAPI            → /api/v1/predict
        └── ONNX Runtime + tokenizers  ← modèle fp32 (release GitHub, checksum vérifié)
```

### Phases (une pull request par phase, petits commits)

| Phase | Contenu | Corrige |
|---|---|---|
| ✅ **1. Fiabilisation backend** | Package `sms_fraud/`, contrat d'API `/v1` (label + score), validation, troncature, inférence non bloquante, `/health` et `/ready`, config par variables d'environnement, CORS restreint, stack Python 3.12 alignée sur l'entraînement, tests pytest, lint ruff, CI GitHub Actions | B1–B11, M4, M5, F1, I3, I6–I8 |
| ~~2. ML reproductible~~ | **Hors périmètre** : le modèle publié est conservé tel quel (décision du 2026-10-17), le dataset ayant déjà été indexé par ailleurs | — |
| **3. Inférence ONNX** | Fusion LoRA, export ONNX **fp32** (poids inchangés, pas de quantification), backend `onnx` de l'API, vérification de parité avec PyTorch, publication en release GitHub | Compatibilité Vercel, M6 |
| **4. Frontend** | Nouvelle UI : verdict, jauge de confiance, exemples, états de chargement et d'erreur, accessibilité | F2–F6 |
| **5. Déploiement Vercel** | `vercel.json`, fonction Python, déploiements de preview par PR, suppression de Docker/nginx/EC2, README fidèle | I1, I2, I4, I5, I9, doc |
