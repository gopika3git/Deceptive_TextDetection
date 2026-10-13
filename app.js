/**
 * app.js
 * VeritasAI — General Text Deception Detection & Personal Stylometry Frontend.
 * Orchestrates Engine A (General Classifier), Engine B (Personal Stylometry),
 * Dual-Engine Synthesis, Profile CRUD, and Dynamic Model Metrics.
 */

// Sample Texts for Demonstrations
const SAMPLES = {
  general: {
    hotel_deceptive: "I recently stayed at this hotel for a weekend getaway and it was absolutely magnificent! The staff was the friendliest I have ever encountered, catering to our every single need with a warm smile. Our room was extraordinarily clean, luxurious, and spacious with breathtaking views. The location is perfection, walking distance to everything Chicago has to offer. I will definitely recommend this breathtaking hotel to all my friends and family and will certainly return soon!",
    hotel_truthful: "My husband and I stayed here for three nights while attending a conference downtown. The check-in line was a bit long around 4pm, but the receptionist apologized and offered two bottles of water. The room was on the 14th floor facing the courtyard; it was quiet and the bed was comfortable, though the bathroom door squeaked slightly. We ate breakfast at the downstairs cafe once—the coffee was strong and the oatmeal was good. Convenient location right near the red line subway.",
    email_sample: "Following up on the vendor contract renegotiation, the newly proposed terms appear fully acceptable with one minor exception in section four regarding third-party warranty liabilities. I have marked up the clauses and returned the redlined draft to outside legal counsel for formal signoff."
  },
  personal: {
    matching: "Good afternoon team. I have completed the preliminary assessment of the data pipeline migration. Initial benchmark runs demonstrate consistent throughput across all operational workloads. Please review the attached draft and provide any feedback ahead of our Friday team review.",
    divergent: "HEY MAN!! WHERE ARE U?? COME QUICK ASAP!! NEED HELP IMMEDIATELY ON THIS BUG!!!"
  }
};

let domainCatalog = {};
let activeProfiles = [];
let selectedProfileId = null;
let currentGeneralPrediction = null;
let userFeedbackVote = null;

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  fetchDomainCatalog();
  fetchProfiles();
  fetchLiveMetrics();
});

function setupEventListeners() {
  // Character & word counters
  const genInput = document.getElementById("general-text-input");
  if (genInput) genInput.addEventListener("input", () => updateCharCount("general-text-input", "general-char-count"));

  const persInput = document.getElementById("personal-compare-input");
  if (persInput) persInput.addEventListener("input", () => updateCharCount("personal-compare-input", "personal-char-count"));

  const combInput = document.getElementById("combined-text-input");
  if (combInput) combInput.addEventListener("input", () => updateCharCount("combined-text-input", "combined-char-count"));

  const wSampleInput = document.getElementById("w-sample-text");
  if (wSampleInput) wSampleInput.addEventListener("input", () => updateCharCount("w-sample-text", "w-sample-count"));
}

function updateCharCount(inputId, countId) {
  const el = document.getElementById(inputId);
  const countEl = document.getElementById(countId);
  if (!el || !countEl) return;
  const text = el.value;
  const chars = text.length;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  countEl.innerText = `${chars} characters | ${words} words`;
}

// Tab Switching
function switchTab(tabKey) {
  document.querySelectorAll(".nav-tab").forEach(tab => tab.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

  if (tabKey === "engine-a") {
    document.getElementById("tab-engine-a").classList.add("active");
    document.getElementById("section-engine-a").classList.add("active");
  } else if (tabKey === "engine-b") {
    document.getElementById("tab-engine-b").classList.add("active");
    document.getElementById("section-engine-b").classList.add("active");
  } else if (tabKey === "combined") {
    document.getElementById("tab-combined").classList.add("active");
    document.getElementById("section-combined").classList.add("active");
  } else if (tabKey === "metrics") {
    document.getElementById("tab-metrics").classList.add("active");
    document.getElementById("section-metrics").classList.add("active");
    fetchLiveMetrics();
  }
}


// ==================== ENGINE A: GENERAL CLASSIFICATION ====================

async function fetchDomainCatalog() {
  try {
    const res = await fetch("/api/domains");
    if (!res.ok) return;
    const data = await res.json();
    domainCatalog = data.catalog || {};
    onDomainChange();
  } catch (err) {
    console.warn("Could not load domains catalog:", err);
  }
}

function onDomainChange() {
  const select = document.getElementById("domain-select");
  const pill = document.getElementById("domain-validation-pill");
  const caveatBox = document.getElementById("domain-caveat-box");
  if (!select || !pill || !caveatBox) return;

  const domain = select.value;
  const meta = domainCatalog[domain] || {};
  const status = meta.validation_status || "Supported / General text transfer";
  const isValidated = status.includes("Validated");

  if (isValidated) {
    pill.className = "pill-badge pill-pass";
    pill.innerText = "Validated Benchmark";
    caveatBox.className = "notice-box notice-info";
    if (domain === "hotel_review") {
      caveatBox.innerHTML = `<strong>Validated:</strong> Evaluated on hotel reviews with verified 88.75% test accuracy (Ott et al. gold standard). Predictions reflect benchmark deception markers.`;
    } else if (domain === "personal_statement") {
      caveatBox.innerHTML = `<strong>Validated:</strong> Evaluated on personal statements with verified 62.55% test accuracy across 103 held-out contributors (Pérez-Rosas & Mihalcea 2015). Predictions reflect subtle first-person cues, not universal lie detection.`;
    } else {
      caveatBox.innerHTML = `<strong>Validated:</strong> ${status}. Predictions reflect benchmark training resemblance.`;
    }
  } else {
    pill.className = "pill-badge pill-info";
    pill.innerText = "General Transfer (Unvalidated)";
    caveatBox.className = "notice-box notice-warning";
    caveatBox.innerHTML = `<strong>Notice:</strong> This domain utilizes lexical and n-gram transfer. Domain-specific ground truth has not yet been validated.`;
  }
}

function loadSampleGeneral(sampleType) {
  const textarea = document.getElementById("general-text-input");
  if (!textarea) return;
  if (SAMPLES.general[sampleType]) {
    textarea.value = SAMPLES.general[sampleType];
    updateCharCount("general-text-input", "general-char-count");
  }
}

function clearGeneralText() {
  const textarea = document.getElementById("general-text-input");
  if (textarea) textarea.value = "";
  updateCharCount("general-text-input", "general-char-count");
  document.getElementById("general-placeholder").classList.remove("hidden");
  document.getElementById("general-results-content").classList.add("hidden");
}

async function analyzeGeneralText() {
  const textarea = document.getElementById("general-text-input");
  const text = textarea ? textarea.value.trim() : "";
  if (!text) {
    alert("Please enter text to classify.");
    return;
  }

  const domainSelect = document.getElementById("domain-select");
  const domain = domainSelect ? domainSelect.value : "general_text";

  const btn = document.getElementById("analyze-general-btn");
  const origBtnContent = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span>Analyzing Text...</span>`;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, domain: domain, top_k: 8 })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Analysis failed");
    }

    const data = await res.json();
    currentGeneralPrediction = data;
    renderGeneralResults(data);
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = origBtnContent;
  }
}

function renderGeneralResults(data) {
  document.getElementById("general-placeholder").classList.add("hidden");
  const resultsContent = document.getElementById("general-results-content");
  resultsContent.classList.remove("hidden");

  // Verdict Banner
  const banner = document.getElementById("general-verdict-banner");
  const badge = document.getElementById("general-verdict-badge");
  const scoreVal = document.getElementById("general-score-val");
  const confVal = document.getElementById("general-confidence-val");
  const probVal = document.getElementById("general-prob-val");

  const isDeceptive = data.label === "Deceptive";
  banner.className = `verdict-banner ${isDeceptive ? "verdict-deceptive" : "verdict-truthful"}`;
  badge.innerText = data.label;

  const formattedScore = (data.decision_score > 0 ? "+" : "") + data.decision_score.toFixed(4);
  scoreVal.innerText = formattedScore;
  confVal.innerText = `(${data.confidence_label})`;

  if (data.calibrated_probability !== null && data.calibrated_probability !== undefined) {
    probVal.innerText = `${(data.calibrated_probability * 100).toFixed(1)}%`;
  } else {
    probVal.innerText = "N/A";
  }

  // Domain Caveat
  const caveatBox = document.getElementById("general-result-caveat");
  caveatBox.innerText = data.domain_caveat || "";

  // Render Token Heatmap
  const highlightBox = document.getElementById("general-highlight-box");
  highlightBox.innerHTML = "";
  if (data.word_highlights && data.word_highlights.length > 0) {
    data.word_highlights.forEach(token => {
      const span = document.createElement("span");
      span.innerText = token.word + " ";
      span.className = `token-highlight cat-${token.category}`;
      if (token.category !== "neutral") {
        span.title = `${token.category.toUpperCase()} cue (weight: ${token.impact.toFixed(4)})`;
      }
      highlightBox.appendChild(span);
    });
  }

  // Render Feature Lists
  renderCuesList("general-deceptive-cues", data.top_deceptive_features, "score-pos");
  renderCuesList("general-truthful-cues", data.top_truthful_features, "score-neg");

  // Reset Feedback Drawer
  document.getElementById("general-feedback-drawer").classList.add("hidden");
  document.getElementById("general-feedback-msg").classList.add("hidden");
}

function renderCuesList(elementId, cues, scoreClass) {
  const ul = document.getElementById(elementId);
  if (!ul) return;
  ul.innerHTML = "";
  if (!cues || cues.length === 0) {
    ul.innerHTML = `<li class="feature-item" style="color:var(--text-muted);">No prominent cues</li>`;
    return;
  }
  cues.forEach(cue => {
    const li = document.createElement("li");
    li.className = "feature-item";
    const sign = cue.contribution > 0 ? "+" : "";
    li.innerHTML = `
      <span class="feature-name" title="${cue.feature}">"${cue.feature}"</span>
      <span class="feature-score ${scoreClass}">${sign}${cue.contribution.toFixed(4)}</span>
    `;
    ul.appendChild(li);
  });
}

function openFeedbackModal(agrees) {
  userFeedbackVote = agrees;
  const drawer = document.getElementById("general-feedback-drawer");
  const title = document.getElementById("general-feedback-title");
  if (!drawer || !title || !currentGeneralPrediction) return;
  drawer.classList.remove("hidden");
  title.innerText = agrees
    ? `You agree this text is ${currentGeneralPrediction.label}. Add comments:`
    : `You disagree (you perceive it as ${currentGeneralPrediction.label === "Truthful" ? "Deceptive" : "Truthful"}). Add details:`;
}

async function sendGeneralFeedback() {
  if (!currentGeneralPrediction) return;
  const text = document.getElementById("general-text-input").value.trim();
  const comment = document.getElementById("general-feedback-comment").value.trim();
  const domain = document.getElementById("domain-select").value;

  const userLabel = userFeedbackVote
    ? currentGeneralPrediction.label
    : (currentGeneralPrediction.label === "Truthful" ? "Deceptive" : "Truthful");

  try {
    const res = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        review_text: text,
        domain: domain,
        predicted_label: currentGeneralPrediction.label,
        user_label: userLabel,
        comments: comment
      })
    });
    if (!res.ok) throw new Error("Feedback submission failed");
    const msg = document.getElementById("general-feedback-msg");
    msg.classList.remove("hidden");
    msg.innerText = "✓ Thank you! Feedback has been safely recorded for audit.";
    document.getElementById("general-feedback-drawer").classList.add("hidden");
    document.getElementById("general-feedback-comment").value = "";
  } catch (err) {
    alert("Feedback error: " + err.message);
  }
}


// ==================== ENGINE B: PERSONAL STYLOMETRY ====================

async function fetchProfiles() {
  try {
    const res = await fetch("/api/profiles");
    if (!res.ok) return;
    const data = await res.json();
    activeProfiles = data.profiles || [];
    renderProfileSelects();
    if (activeProfiles.length > 0 && !selectedProfileId) {
      selectedProfileId = activeProfiles[0].id;
      loadActiveProfileDetails(selectedProfileId);
    }
  } catch (err) {
    console.warn("Could not load profiles:", err);
  }
}

function renderProfileSelects() {
  const profileSelect = document.getElementById("profile-select");
  const combinedSelect = document.getElementById("combined-profile-select");
  if (!profileSelect) return;

  profileSelect.innerHTML = "";
  if (combinedSelect) combinedSelect.innerHTML = `<option value="">(None - General classification only)</option>`;

  activeProfiles.forEach(p => {
    const isDemo = p.is_demo || p.id === "demo_analyst";
    const tag = isDemo ? "[DEMO SUBJECT]" : "[TARGET AUTHOR]";
    const readyTag = p.readiness_status || p.reliability_rating;

    const opt = document.createElement("option");
    opt.value = p.id;
    opt.innerText = isDemo
      ? `${tag} ${p.name} (${p.sample_count} samples) — Demonstration Only`
      : `${tag} ${p.name} (${p.sample_count} samples, ${readyTag})`;
    profileSelect.appendChild(opt);

    if (combinedSelect) {
      const opt2 = document.createElement("option");
      opt2.value = p.id;
      opt2.innerText = isDemo
        ? `${tag} ${p.name} (Demonstration Subject)`
        : `${tag} ${p.name} (${p.sample_count} samples)`;
      combinedSelect.appendChild(opt2);
    }
  });

  if (selectedProfileId) {
    profileSelect.value = selectedProfileId;
    if (combinedSelect) combinedSelect.value = selectedProfileId;
  }
}

async function onProfileChange() {
  const select = document.getElementById("profile-select");
  if (!select) return;
  selectedProfileId = select.value;
  const combSelect = document.getElementById("combined-profile-select");
  if (combSelect) combSelect.value = selectedProfileId;
  await loadActiveProfileDetails(selectedProfileId);
}

async function onCombinedProfileChange() {
  const combSelect = document.getElementById("combined-profile-select");
  if (!combSelect) return;
  const pid = combSelect.value;
  if (!pid) {
    const cInd = document.getElementById("combined-active-profile-indicator");
    if (cInd) cInd.innerHTML = `<span style="color:var(--text-muted);">ℹ️ <strong>General Classification Only:</strong> No target author baseline selected for Engine B.</span>`;
    return;
  }
  try {
    const res = await fetch(`/api/profiles/${pid}`);
    if (res.ok) {
      const data = await res.json();
      updateActiveProfileIndicators(data.profile);
    }
  } catch (err) {
    console.warn("Could not load combined profile details:", err);
  }
}

async function loadActiveProfileDetails(profileId) {
  if (!profileId) {
    updateActiveProfileIndicators(null);
    return;
  }
  try {
    const res = await fetch(`/api/profiles/${profileId}`);
    if (!res.ok) return;
    const data = await res.json();
    const prof = data.profile;
    renderActiveProfileOverview(prof);
  } catch (err) {
    console.warn("Could not load profile details:", err);
  }
}

function updateActiveProfileIndicators(prof) {
  const pInd = document.getElementById("personal-active-profile-indicator");
  const cInd = document.getElementById("combined-active-profile-indicator");
  if (!prof) {
    if (pInd) {
      pInd.innerHTML = `<span style="color:var(--text-muted);">No target author profile selected. Create or select a profile above.</span>`;
    }
    if (cInd) {
      cInd.innerHTML = `<span style="color:var(--text-muted);">ℹ️ <strong>General Classification Only:</strong> No target author baseline selected. To evaluate author-specific stylistic divergence, select a profile above.</span>`;
    }
    return;
  }

  const isDemo = prof.is_demo || prof.id === "demo_analyst";
  const baseline = prof.baseline || {};
  const isReady = (baseline.is_operationally_ready !== undefined)
    ? baseline.is_operationally_ready
    : ((prof.samples || []).length >= 3 && (baseline.total_word_count || 0) >= 200);
  const readinessText = isReady ? "Operationally Ready for Comparison" : "Needs More Writing Samples";
  const badgeClass = isReady ? "pill-pass" : "pill-warning";
  const typeTag = isDemo ? "⚠️ DEMO SUBJECT" : "🎯 TARGET AUTHOR";

  const pContent = `
    <div class="indicator-meta">
      <span class="pill-badge ${isDemo ? 'pill-warning' : 'pill-info'}">${typeTag}</span>
      <strong>${prof.name}</strong>
      <span style="color:var(--text-secondary);">(${prof.samples ? prof.samples.length : 0} samples, ${baseline.total_word_count || 0} words)</span>
    </div>
    <div style="display:flex; align-items:center; gap:0.5rem;">
      <span class="pill-badge ${badgeClass}">${readinessText}</span>
    </div>
  `;

  if (pInd) pInd.innerHTML = pContent;

  const combSelect = document.getElementById("combined-profile-select");
  if (cInd) {
    if (combSelect && combSelect.value === prof.id) {
      cInd.innerHTML = `
        <div class="indicator-meta">
          <span class="pill-badge ${isDemo ? 'pill-warning' : 'pill-info'}">${typeTag}</span>
          <span>Engine B Target: <strong>${prof.name}</strong> (${prof.samples ? prof.samples.length : 0} samples)</span>
        </div>
        <span class="pill-badge ${badgeClass}">${readinessText}</span>
      `;
    } else if (combSelect && !combSelect.value) {
      cInd.innerHTML = `<span style="color:var(--text-muted);">ℹ️ <strong>General Classification Only:</strong> No target author baseline selected for Engine B.</span>`;
    }
  }
}

function renderActiveProfileOverview(prof) {
  document.getElementById("active-profile-name").innerText = prof.name;
  document.getElementById("active-profile-desc").innerText = prof.description || "Target Author Profile";

  const isDemo = prof.is_demo || prof.id === "demo_analyst";
  const demoBanner = document.getElementById("demo-profile-banner");
  if (demoBanner) {
    if (isDemo) {
      demoBanner.classList.remove("hidden");
    } else {
      demoBanner.classList.add("hidden");
    }
  }

  const baseline = prof.baseline || {};
  document.getElementById("active-profile-samples").innerText = baseline.sample_count || 0;
  document.getElementById("active-profile-words").innerText = baseline.total_word_count || 0;

  const stats = baseline.feature_stats || {};
  const avgSent = stats["avg_sentence_length"] ? stats["avg_sentence_length"].mean.toFixed(1) + " w" : "--";
  const avgTtr = stats["vocabulary_diversity_ttr"] ? stats["vocabulary_diversity_ttr"].mean.toFixed(2) : "--";
  document.getElementById("active-profile-sent-len").innerText = avgSent;
  document.getElementById("active-profile-ttr").innerText = avgTtr;

  const badge = document.getElementById("active-profile-badge");
  const relRating = baseline.reliability_rating || "Unknown";
  badge.innerText = relRating;
  if (relRating.includes("High")) {
    badge.className = "pill-badge pill-pass";
  } else if (relRating.includes("Moderate")) {
    badge.className = "pill-badge pill-warning";
  } else {
    badge.className = "pill-badge pill-fail";
  }

  // Warnings
  const warnContainer = document.getElementById("profile-warnings-container");
  warnContainer.innerHTML = "";
  if (baseline.sufficiency_warnings && baseline.sufficiency_warnings.length > 0) {
    baseline.sufficiency_warnings.forEach(w => {
      const box = document.createElement("div");
      box.className = "notice-box notice-warning";
      box.style.marginTop = "0.5rem";
      box.innerText = `⚠️ ${w}`;
      warnContainer.appendChild(box);
    });
  }

  // Update indicators
  updateActiveProfileIndicators(prof);

  // Populate timeline / samples drawer
  renderSampleList(prof.samples || []);
}

function renderSampleList(samples) {
  const container = document.getElementById("profile-sample-list");
  if (!container) return;
  container.innerHTML = "";
  if (samples.length === 0) {
    container.innerHTML = `<p style="font-size:0.8rem; color:var(--text-muted);">No historical samples added yet.</p>`;
    return;
  }

  samples.forEach(s => {
    const item = document.createElement("div");
    item.className = "sample-item-card";
    const dateStr = s.date || "Unknown Date";
    const contextStr = s.context || "General";
    item.innerHTML = `
      <div style="flex:1;">
        <div class="sample-meta">📅 ${dateStr} • 🏷️ ${contextStr} ${s.source_description ? '• ' + s.source_description : ''}</div>
        <p class="sample-item-text">"${s.text}"</p>
      </div>
      <button class="btn btn-danger-ghost btn-sm" onclick="deleteSample('${s.id}')" title="Delete sample">&times;</button>
    `;
    container.appendChild(item);
  });
}

function toggleTimelineView() {
  const drawer = document.getElementById("profile-timeline-drawer");
  if (drawer) drawer.classList.toggle("hidden");
}

function loadSamplePersonal(type) {
  const textarea = document.getElementById("personal-compare-input");
  if (!textarea) return;
  if (SAMPLES.personal[type]) {
    textarea.value = SAMPLES.personal[type];
    updateCharCount("personal-compare-input", "personal-char-count");
  }
}

function clearPersonalText() {
  const textarea = document.getElementById("personal-compare-input");
  if (textarea) textarea.value = "";
  updateCharCount("personal-compare-input", "personal-char-count");
  document.getElementById("personal-placeholder").classList.remove("hidden");
  document.getElementById("personal-results-content").classList.add("hidden");
}

async function runPersonalComparison() {
  if (!selectedProfileId) {
    alert("Please select or create a personal writing profile first.");
    return;
  }
  const text = document.getElementById("personal-compare-input").value.trim();
  if (!text) {
    alert("Please enter the author's current text to compare.");
    return;
  }

  const btn = document.getElementById("compare-personal-btn");
  const origBtnContent = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span>Comparing Against Baseline...</span>`;

  try {
    const res = await fetch("/api/compare-personal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        profile_id: selectedProfileId
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Comparison failed");
    }

    const data = await res.json();
    renderPersonalComparisonResults(data.comparison);
  } catch (err) {
    alert("Comparison error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = origBtnContent;
  }
}

function renderPersonalComparisonResults(comp) {
  document.getElementById("personal-placeholder").classList.add("hidden");
  const resultsContent = document.getElementById("personal-results-content");
  resultsContent.classList.remove("hidden");

  // Alignment Meter
  const alignVal = comp.alignment_index !== null ? comp.alignment_index : 100.0;
  document.getElementById("alignment-index-val").innerText = `${alignVal}%`;
  document.getElementById("alignment-assessment-title").innerText = comp.overall_assessment;
  document.getElementById("alignment-summary-text").innerText = comp.summary_statement;

  const badge = document.getElementById("alignment-badge");
  badge.innerText = comp.overall_assessment;
  if (comp.assessment_color === "success") {
    badge.className = "pill-badge pill-pass";
  } else if (comp.assessment_color === "warning") {
    badge.className = "pill-badge pill-warning";
  } else {
    badge.className = "pill-badge pill-fail";
  }

  // Shifted Features Table
  const tbody = document.getElementById("shifted-features-tbody");
  tbody.innerHTML = "";
  if (!comp.top_shifted_features || comp.top_shifted_features.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:1rem;">All features within baseline range (|Z| &lt; 1.0).</td></tr>`;
  } else {
    comp.top_shifted_features.forEach(feat => {
      const tr = document.createElement("tr");
      const zClass = feat.shift_severity === "high" ? "badge-shift-high" : "badge-shift-mod";
      const sign = feat.z_score > 0 ? "+" : "";
      const relSign = feat.relative_change_pct > 0 ? "+" : "";
      tr.innerHTML = `
        <td><strong>${feat.label}</strong><br><span style="font-size:0.7rem; color:var(--text-muted);">${feat.description}</span></td>
        <td>${feat.baseline_mean} &plusmn; ${feat.baseline_std}</td>
        <td><strong>${feat.current_value}</strong></td>
        <td><span class="${zClass}">${sign}${feat.z_score.toFixed(2)} &sigma;</span></td>
        <td>${relSign}${feat.relative_change_pct.toFixed(1)}%</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Stable Features Chips
  const stableChips = document.getElementById("stable-features-chips");
  stableChips.innerHTML = "";
  if (!comp.stable_features || comp.stable_features.length === 0) {
    stableChips.innerHTML = `<span style="font-size:0.75rem; color:var(--text-muted);">No stable features recorded.</span>`;
  } else {
    comp.stable_features.forEach(f => {
      const chip = document.createElement("span");
      chip.className = "stable-chip";
      chip.innerText = `✓ ${f.label} (${f.current_value})`;
      stableChips.appendChild(chip);
    });
  }

  // Alternative Explanations List
  const altList = document.getElementById("alt-explanations-list");
  altList.innerHTML = "";
  if (comp.alternative_explanations) {
    comp.alternative_explanations.forEach(exp => {
      const li = document.createElement("li");
      li.innerText = exp;
      altList.appendChild(li);
    });
  }
}

// Modal Handlers
function openNewProfileModal() {
  document.getElementById("modal-new-profile").classList.remove("hidden");
}
function closeNewProfileModal() {
  document.getElementById("modal-new-profile").classList.add("hidden");
}

async function submitNewProfile() {
  const name = document.getElementById("new-profile-name").value.trim();
  const desc = document.getElementById("new-profile-desc").value.trim();
  const notes = document.getElementById("new-profile-notes").value.trim();
  const consent = document.getElementById("new-profile-consent").checked;

  if (!name) {
    alert("Please enter a profile name.");
    return;
  }
  if (!consent) {
    alert("You must confirm authorization to analyze and store writing samples.");
    return;
  }

  try {
    const res = await fetch("/api/profiles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: name,
        description: desc,
        notes: notes,
        user_authorization_confirmed: true
      })
    });
    if (!res.ok) throw new Error("Failed to create profile");
    const data = await res.json();
    closeNewProfileModal();
    await fetchProfiles();
    selectedProfileId = data.profile.id;
    document.getElementById("profile-select").value = selectedProfileId;
    loadActiveProfileDetails(selectedProfileId);
  } catch (err) {
    alert("Error: " + err.message);
  }
}

function openAddSampleModal() {
  document.getElementById("modal-add-sample").classList.remove("hidden");
}
function closeAddSampleModal() {
  document.getElementById("modal-add-sample").classList.add("hidden");
}

async function submitAddSample() {
  if (!selectedProfileId) return;
  const text = document.getElementById("sample-text").value.trim();
  const date = document.getElementById("sample-date").value;
  const context = document.getElementById("sample-context").value;
  const desc = document.getElementById("sample-desc").value.trim();
  const rep = document.getElementById("sample-rep").checked;

  if (!text) {
    alert("Please enter historical sample text.");
    return;
  }

  try {
    const res = await fetch(`/api/profiles/${selectedProfileId}/samples`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        date: date || null,
        context: context,
        source_description: desc,
        is_representative: rep
      })
    });
    if (!res.ok) throw new Error("Failed to add sample");
    closeAddSampleModal();
    document.getElementById("sample-text").value = "";
    await loadActiveProfileDetails(selectedProfileId);
    await fetchProfiles();
  } catch (err) {
    alert("Error: " + err.message);
  }
}

async function deleteSample(sampleId) {
  if (!confirm("Are you sure you want to delete this historical sample?")) return;
  try {
    const res = await fetch(`/api/profiles/${selectedProfileId}/samples/${sampleId}`, {
      method: "DELETE"
    });
    if (!res.ok) throw new Error("Failed to delete sample");
    await loadActiveProfileDetails(selectedProfileId);
    await fetchProfiles();
  } catch (err) {
    alert("Error: " + err.message);
  }
}

async function deleteActiveProfile() {
  if (!selectedProfileId) return;
  if (!confirm("Are you sure you want to permanently delete this profile and all its historical text?")) return;
  try {
    const res = await fetch(`/api/profiles/${selectedProfileId}`, {
      method: "DELETE"
    });
    if (!res.ok) throw new Error("Failed to delete profile");
    selectedProfileId = null;
    await fetchProfiles();
    clearPersonalText();
  } catch (err) {
    alert("Error: " + err.message);
  }
}


// ==================== GUIDED ONBOARDING WIZARD ====================

let wizardCurrentStep = 1;
let wizardCreatedProfile = null;

function openOnboardingWizard() {
  wizardCurrentStep = 1;
  wizardCreatedProfile = null;

  // Reset inputs
  const nameInp = document.getElementById("w-profile-name");
  const descInp = document.getElementById("w-profile-desc");
  const notesInp = document.getElementById("w-profile-notes");
  const consentInp = document.getElementById("w-profile-consent");
  const sampleInp = document.getElementById("w-sample-text");
  const sampleDescInp = document.getElementById("w-sample-desc");

  if (nameInp) nameInp.value = "";
  if (descInp) descInp.value = "";
  if (notesInp) notesInp.value = "";
  if (consentInp) consentInp.checked = false;
  if (sampleInp) sampleInp.value = "";
  if (sampleDescInp) sampleDescInp.value = "";

  const errName = document.getElementById("w-err-name");
  const errConsent = document.getElementById("w-err-consent");
  const errSample = document.getElementById("w-err-sample");
  if (errName) errName.classList.add("hidden");
  if (errConsent) errConsent.classList.add("hidden");
  if (errSample) errSample.classList.add("hidden");

  renderWizardStep(1);
  const wizardModal = document.getElementById("modal-onboarding-wizard");
  if (wizardModal) wizardModal.classList.remove("hidden");
}

function closeOnboardingWizard() {
  const wizardModal = document.getElementById("modal-onboarding-wizard");
  if (wizardModal) wizardModal.classList.add("hidden");
  if (wizardCreatedProfile) {
    selectedProfileId = wizardCreatedProfile.id;
    fetchProfiles().then(() => {
      const select = document.getElementById("profile-select");
      if (select) select.value = selectedProfileId;
      const combSelect = document.getElementById("combined-profile-select");
      if (combSelect) combSelect.value = selectedProfileId;
      loadActiveProfileDetails(selectedProfileId);
    });
  }
}

function renderWizardStep(step) {
  wizardCurrentStep = step;

  // Update step indicators
  for (let s = 1; s <= 4; s++) {
    const el = document.getElementById(`wizard-step-${s}`);
    const ind = document.getElementById(`w-step-ind-${s}`);
    if (el) el.classList.add("hidden");
    if (ind) {
      ind.classList.remove("active", "completed");
      if (s < step) ind.classList.add("completed");
      else if (s === step) ind.classList.add("active");
    }
  }

  const currentEl = document.getElementById(`wizard-step-${step}`);
  if (currentEl) currentEl.classList.remove("hidden");

  const btnBack = document.getElementById("w-btn-back");
  const btnNext = document.getElementById("w-btn-next");
  const btnCancel = document.getElementById("w-btn-cancel");

  if (step === 1) {
    if (btnBack) btnBack.classList.add("hidden");
    if (btnNext) {
      btnNext.classList.remove("hidden");
      btnNext.innerHTML = `<span>Next: Add Samples &rarr;</span>`;
    }
    if (btnCancel) {
      btnCancel.classList.remove("hidden");
      btnCancel.innerText = "Cancel";
    }
  } else if (step === 2) {
    if (btnBack) btnBack.classList.remove("hidden");
    if (btnNext) {
      btnNext.classList.remove("hidden");
      btnNext.innerHTML = `<span>Next: Review Readiness &rarr;</span>`;
    }
    if (btnCancel) {
      btnCancel.classList.remove("hidden");
      btnCancel.innerText = "Save & Exit";
    }
  } else if (step === 3) {
    if (btnBack) btnBack.classList.remove("hidden");
    if (btnNext) {
      btnNext.classList.remove("hidden");
      btnNext.innerHTML = `<span>Complete &amp; Activate &rarr;</span>`;
    }
    if (btnCancel) {
      btnCancel.classList.remove("hidden");
      btnCancel.innerText = "Close";
    }
  } else if (step === 4) {
    if (btnBack) btnBack.classList.add("hidden");
    if (btnNext) btnNext.classList.add("hidden");
    if (btnCancel) {
      btnCancel.classList.remove("hidden");
      btnCancel.innerText = "Finish";
    }
  }
}

async function wizardGoNext() {
  if (wizardCurrentStep === 1) {
    const name = document.getElementById("w-profile-name").value.trim();
    const consent = document.getElementById("w-profile-consent").checked;
    let valid = true;

    const errName = document.getElementById("w-err-name");
    const errConsent = document.getElementById("w-err-consent");

    if (!name) {
      if (errName) errName.classList.remove("hidden");
      valid = false;
    } else {
      if (errName) errName.classList.add("hidden");
    }

    if (!consent) {
      if (errConsent) errConsent.classList.remove("hidden");
      valid = false;
    } else {
      if (errConsent) errConsent.classList.add("hidden");
    }

    if (!valid) return;

    if (!wizardCreatedProfile) {
      const desc = document.getElementById("w-profile-desc").value.trim();
      const notes = document.getElementById("w-profile-notes").value.trim();
      try {
        const res = await fetch("/api/profiles", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: name,
            description: desc,
            notes: notes,
            user_authorization_confirmed: true
          })
        });
        if (!res.ok) {
          const err = await res.json();
          alert(err.detail || "Failed to create target author profile");
          return;
        }
        const data = await res.json();
        wizardCreatedProfile = data.profile;
        await fetchProfiles();
      } catch (err) {
        alert("Error creating target profile: " + err.message);
        return;
      }
    }
    updateWizardStep2UI();
    renderWizardStep(2);
  } else if (wizardCurrentStep === 2) {
    if (!wizardCreatedProfile) return;
    await updateWizardStep3UI();
    renderWizardStep(3);
  } else if (wizardCurrentStep === 3) {
    if (!wizardCreatedProfile) return;
    selectedProfileId = wizardCreatedProfile.id;
    const select = document.getElementById("profile-select");
    if (select) select.value = selectedProfileId;
    const combSelect = document.getElementById("combined-profile-select");
    if (combSelect) combSelect.value = selectedProfileId;
    await loadActiveProfileDetails(selectedProfileId);

    const sName = document.getElementById("w-success-name");
    const sStats = document.getElementById("w-success-stats");
    if (sName) sName.innerText = wizardCreatedProfile.name;
    const baseline = wizardCreatedProfile.baseline || {};
    const sc = wizardCreatedProfile.samples ? wizardCreatedProfile.samples.length : 0;
    const tw = baseline.total_word_count || 0;
    const isReady = baseline.is_operationally_ready;
    if (sStats) {
      sStats.innerText = `${sc} sample${sc !== 1 ? 's' : ''} • ${tw} words • ${isReady ? 'Operationally Ready' : 'Incomplete Baseline'}`;
    }

    renderWizardStep(4);
  }
}

function wizardGoBack() {
  if (wizardCurrentStep > 1) {
    renderWizardStep(wizardCurrentStep - 1);
  }
}

async function submitOnboardingSample() {
  if (!wizardCreatedProfile) return;
  const textarea = document.getElementById("w-sample-text");
  const text = textarea ? textarea.value.trim() : "";
  const words = text ? text.split(/\s+/).length : 0;

  const errSample = document.getElementById("w-err-sample");
  if (words < 5) {
    if (errSample) errSample.classList.remove("hidden");
    return;
  }
  if (errSample) errSample.classList.add("hidden");

  const context = document.getElementById("w-sample-context").value;
  const desc = document.getElementById("w-sample-desc").value.trim();
  const btn = document.getElementById("w-add-sample-btn");
  if (btn) btn.disabled = true;

  try {
    const res = await fetch(`/api/profiles/${wizardCreatedProfile.id}/samples`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        context: context,
        source_description: desc,
        is_representative: true
      })
    });
    if (!res.ok) {
      const err = await res.json();
      alert(err.detail || "Failed to add sample");
      return;
    }
    const data = await res.json();
    wizardCreatedProfile = data.profile;
    textarea.value = "";
    document.getElementById("w-sample-desc").value = "";
    document.getElementById("w-sample-count").innerText = "0 characters | 0 words";

    updateWizardStep2UI();
  } catch (err) {
    alert("Error adding sample: " + err.message);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function deleteOnboardingSample(sampleId) {
  if (!wizardCreatedProfile) return;
  try {
    const res = await fetch(`/api/profiles/${wizardCreatedProfile.id}/samples/${sampleId}`, {
      method: "DELETE"
    });
    if (!res.ok) throw new Error("Failed to delete sample");
    const data = await res.json();
    wizardCreatedProfile = data.profile;
    updateWizardStep2UI();
  } catch (err) {
    alert("Error deleting sample: " + err.message);
  }
}

function updateWizardStep2UI() {
  if (!wizardCreatedProfile) return;
  const samples = wizardCreatedProfile.samples || [];
  const baseline = wizardCreatedProfile.baseline || {};
  const sc = samples.length;
  const tw = baseline.total_word_count || 0;

  const statS = document.getElementById("w-stat-samples");
  const statW = document.getElementById("w-stat-words");
  if (statS) statS.innerText = sc;
  if (statW) statW.innerText = tw;

  const pctSamples = Math.min(100, Math.round((sc / 3) * 100));
  const pctWords = Math.min(100, Math.round((tw / 200) * 100));

  const pctSEl = document.getElementById("w-pct-samples");
  const pctWEl = document.getElementById("w-pct-words");
  if (pctSEl) pctSEl.innerText = `${pctSamples}%`;
  if (pctWEl) pctWEl.innerText = `${pctWords}%`;

  const barS = document.getElementById("w-bar-samples");
  const barW = document.getElementById("w-bar-words");
  if (barS) barS.style.width = `${pctSamples}%`;
  if (barW) barW.style.width = `${pctWords}%`;

  const badge = document.getElementById("w-readiness-badge");
  const hint = document.getElementById("w-readiness-hint");
  const isReady = (sc >= 3 && tw >= 200);

  if (badge && hint) {
    if (isReady) {
      badge.className = "pill-badge pill-pass";
      badge.innerText = "Operationally Ready";
      hint.innerHTML = `<span style="color:var(--color-truthful);">✅ Minimum baseline reached (${sc} samples, ${tw} words). The engine can compute initial variance for comparison.</span>`;
    } else {
      badge.className = "pill-badge pill-warning";
      badge.innerText = "Needs More Writing Samples";
      const missingSamples = Math.max(0, 3 - sc);
      const missingWords = Math.max(0, 200 - tw);
      hint.innerText = `Add at least ${missingSamples} more sample(s) and ${missingWords} more words to establish operational baseline variance.`;
    }
  }

  const container = document.getElementById("w-samples-container");
  const itemsCount = document.getElementById("w-sample-items-count");
  if (itemsCount) itemsCount.innerText = sc;
  if (container) {
    if (sc === 0) {
      container.innerHTML = `<p style="font-size:0.8rem; color:var(--text-muted); text-align:center; padding:1rem;">No samples added yet. Paste a sample above and click "+ Add Sample".</p>`;
    } else {
      container.innerHTML = "";
      samples.forEach(s => {
        const words = s.text.trim().split(/\s+/).length;
        const div = document.createElement("div");
        div.className = "sample-item-card";
        div.style.marginBottom = "0.4rem";
        div.style.padding = "0.5rem 0.75rem";
        div.innerHTML = `
          <div style="flex:1;">
            <div class="sample-meta" style="font-size:0.75rem;">🏷️ ${s.context} • ${words} words ${s.source_description ? '• ' + s.source_description : ''}</div>
            <p class="sample-item-text" style="font-size:0.8rem; margin-top:2px;">"${s.text.substring(0, 100)}${s.text.length > 100 ? '...' : ''}"</p>
          </div>
          <button class="btn btn-danger-ghost btn-sm" onclick="deleteOnboardingSample('${s.id}')" title="Delete sample">&times;</button>
        `;
        container.appendChild(div);
      });
    }
  }
}

async function updateWizardStep3UI() {
  if (!wizardCreatedProfile) return;
  try {
    const res = await fetch(`/api/profiles/${wizardCreatedProfile.id}/readiness`);
    if (!res.ok) return;
    const readyData = await res.json();

    const rName = document.getElementById("w-review-name");
    const rDesc = document.getElementById("w-review-desc");
    if (rName) rName.innerText = wizardCreatedProfile.name;
    if (rDesc) rDesc.innerText = wizardCreatedProfile.description || "Target Author Profile";

    const badge = document.getElementById("w-review-status-badge");
    if (badge) {
      badge.innerText = readyData.readiness_status;
      badge.className = `pill-badge ${readyData.is_operationally_ready ? 'pill-pass' : 'pill-warning'}`;
    }

    const rSamples = document.getElementById("w-review-samples");
    const rWords = document.getElementById("w-review-words");
    if (rSamples) rSamples.innerText = readyData.sample_count;
    if (rWords) rWords.innerText = readyData.total_words;

    const stats = (wizardCreatedProfile.baseline || {}).feature_stats || {};
    const sentLen = stats["avg_sentence_length"] ? stats["avg_sentence_length"].mean.toFixed(1) + " w" : "--";
    const ttr = stats["vocabulary_diversity_ttr"] ? stats["vocabulary_diversity_ttr"].mean.toFixed(2) : "--";
    const rSent = document.getElementById("w-review-sent");
    const rTtr = document.getElementById("w-review-ttr");
    if (rSent) rSent.innerText = sentLen;
    if (rTtr) rTtr.innerText = ttr;

    const detailBox = document.getElementById("w-readiness-detail-box");
    if (detailBox) {
      detailBox.className = `notice-box ${readyData.is_operationally_ready ? 'notice-info' : 'notice-warning'}`;
      detailBox.innerHTML = `<strong>Operational Status:</strong> ${readyData.readiness_detail}`;
    }
  } catch (err) {
    console.warn("Could not fetch readiness detail:", err);
  }
}

async function finishOnboardingToTab(targetTab) {
  const wizardModal = document.getElementById("modal-onboarding-wizard");
  if (wizardModal) wizardModal.classList.add("hidden");
  if (wizardCreatedProfile) {
    selectedProfileId = wizardCreatedProfile.id;
    await fetchProfiles();
    const select = document.getElementById("profile-select");
    if (select) select.value = selectedProfileId;
    const combSelect = document.getElementById("combined-profile-select");
    if (combSelect) combSelect.value = selectedProfileId;
    await loadActiveProfileDetails(selectedProfileId);
  }
  switchTab(targetTab);
  if (targetTab === "engine-b") {
    const input = document.getElementById("personal-compare-input");
    if (input) input.focus();
  } else if (targetTab === "combined") {
    const input = document.getElementById("combined-text-input");
    if (input) input.focus();
  }
}


// ==================== TAB 3: COMBINED DUAL-ENGINE SYNTHESIS ====================

async function runCombinedAnalysis() {
  const text = document.getElementById("combined-text-input").value.trim();
  if (!text) {
    alert("Please enter text for dual-engine synthesis.");
    return;
  }

  const domain = document.getElementById("combined-domain-select").value;
  const profileId = document.getElementById("combined-profile-select").value || null;

  const btn = document.getElementById("run-combined-btn");
  const origBtnContent = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span>Synthesizing Dual Engines...</span>`;

  try {
    const res = await fetch("/api/analyze-combined", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        domain: domain,
        profile_id: profileId
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Combined analysis failed");
    }

    const data = await res.json();
    renderCombinedResults(data);
  } catch (err) {
    alert("Combined analysis error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = origBtnContent;
  }
}

function renderCombinedResults(data) {
  const view = document.getElementById("combined-results-view");
  view.classList.remove("hidden");

  const synth = data.combined_synthesis || {};
  document.getElementById("combined-concordance-title").innerText = synth.concordance_type || "Analysis Completed";
  document.getElementById("combined-narrative-text").innerText = synth.synthesis_narrative || "";

  // Engine A card
  const a = data.engine_a_general_classification || {};
  document.getElementById("comb-a-class").innerText = a.label || "Unknown";
  document.getElementById("comb-a-score").innerText = (a.decision_score > 0 ? "+" : "") + (a.decision_score ? a.decision_score.toFixed(4) : "0.00");
  document.getElementById("comb-a-prob").innerText = a.calibrated_probability !== null ? `${(a.calibrated_probability * 100).toFixed(1)}%` : "N/A";
  document.getElementById("comb-a-domain").innerText = a.domain_status || a.domain || "";
  document.getElementById("comb-a-caveat").innerText = a.domain_caveat || "";

  // Engine B card
  const b = data.engine_b_personal_stylometry;
  if (b && !b.error) {
    document.getElementById("comb-b-alignment").innerText = `${b.alignment_index}%`;
    document.getElementById("comb-b-assessment").innerText = b.overall_assessment || "Evaluated";
    document.getElementById("comb-b-samples").innerText = `${b.baseline_summary.sample_count} historical samples`;
    document.getElementById("comb-b-reliability").innerText = b.baseline_summary.reliability_rating || "Unknown";
    document.getElementById("comb-b-summary").innerText = b.summary_statement || "";
  } else {
    document.getElementById("comb-b-alignment").innerText = "Not Evaluated";
    document.getElementById("comb-b-assessment").innerText = b && b.error ? "Baseline Incomplete" : "No Profile Selected";
    document.getElementById("comb-b-samples").innerText = "--";
    document.getElementById("comb-b-reliability").innerText = "--";
    document.getElementById("comb-b-summary").innerText = b && b.error
      ? b.error
      : "Select a personal writing profile above to evaluate stylometric deviation.";
  }
}


// ==================== TAB 4: METRICS & PRIVACY ====================

async function fetchLiveMetrics() {
  try {
    const res = await fetch("/metrics");
    if (!res.ok) return;
    const json = await res.json();
    const metrics = json.metrics;
    if (!metrics) return;

    const candidates = metrics.models_comparison;
    if (candidates && candidates.calibrated_combined_svm) {
      const cal = candidates.calibrated_combined_svm;
      document.getElementById("metric-cal-acc-val").innerText = (cal.test_accuracy * 100).toFixed(2) + "%";
      document.getElementById("metric-prec-val").innerText = (cal.test_precision * 100).toFixed(2) + "%";
      document.getElementById("metric-rec-val").innerText = (cal.test_recall * 100).toFixed(2) + "%";
    }
    if (candidates && candidates.combined_tfidf_linear_svm) {
      const comb = candidates.combined_tfidf_linear_svm;
      document.getElementById("metric-acc-val").innerText = (comb.test_accuracy * 100).toFixed(2) + "%";
    }
  } catch (err) {
    console.warn("Could not fetch metrics:", err);
  }
}
