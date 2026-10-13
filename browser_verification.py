"""
browser_verification.py
End-to-End Browser Automation Suite for VeritasAI Text Deception Detection.

Launches the local FastAPI application and uses an installed Chrome or Edge browser
via native Chrome DevTools Protocol (CDP) over WebSockets to test all end-to-end
user workflows directly in a live browser without third-party web driver packages.

Workflows Verified:
1. Homepage rendering, headers, tabs, model status badge
2. Hotel Reviews classification (Gold Standard baseline, 88.75% test acc, Platt calibrated probability)
3. Personal Statements classification (Dedicated model, 62.55% test acc, 103 held-out contributors, non-universal caveat)
4. Combined Dual-Engine Synthesis (With and without historical writing profile, methodological rule alert)
5. Engine B Personal Stylometry (Consistent vs divergent sample deviation, Z-scores, stable habits, alternative explanations)
6. Input validation, empty inputs, modal alerts and error handling
7. Metrics & Privacy governance display (Leakage audit, zero-shot transfer failure reporting, ethical disclaimers)
"""

import subprocess
import time
import urllib.request
import json
import asyncio
import os
import shutil
import sys
import socket
import websockets

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(PROJECT_DIR, "chrome_test_profile_e2e")

def find_browser_executable() -> str:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Neither Google Chrome nor Microsoft Edge was found in standard locations.")

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

class BrowserController:
    def __init__(self, browser_ws_url):
        self.ws_url = browser_ws_url
        self.cur_id = 1
        self.session_id = None
        self.ws = None
        self._waiters = {}
        self.dialogs = []

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url)
        asyncio.create_task(self._reader())

    async def _reader(self):
        try:
            async for raw in self.ws:
                data = json.loads(raw)
                if "id" in data:
                    req_id = data["id"]
                    fut = self._waiters.get(req_id)
                    if fut and not fut.done():
                        fut.set_result(data)
                else:
                    if data.get("method") == "Page.javascriptDialogOpening":
                        msg = data.get("params", {}).get("message", "")
                        self.dialogs.append(msg)
                        target_session = data.get("sessionId") or self.session_id
                        asyncio.create_task(self.send("Page.handleJavaScriptDialog", {"accept": True}, session=target_session))
        except Exception:
            pass

    async def send(self, method, params=None, session=None, timeout=15.0):
        cid = self.cur_id
        self.cur_id += 1
        fut = asyncio.get_event_loop().create_future()
        self._waiters[cid] = fut
        payload = {"id": cid, "method": method}
        if params:
            payload["params"] = params
        target_s = session if session is not None else self.session_id
        if target_s:
            payload["sessionId"] = target_s
        await self.ws.send(json.dumps(payload))
        res = await asyncio.wait_for(fut, timeout=timeout)
        self._waiters.pop(cid, None)
        if "error" in res:
            raise RuntimeError(f"CDP Error ({method}): {res['error']}")
        return res.get("result", {})

    async def attach_page(self, url):
        res = await self.send("Target.createTarget", {"url": "about:blank"})
        target_id = res["targetId"]
        att = await self.send("Target.attachToTarget", {"targetId": target_id, "flatten": True})
        self.session_id = att["sessionId"]
        await self.send("Page.enable")
        await self.send("Runtime.enable")
        await self.send("Page.navigate", {"url": url})

    async def evaluate(self, expr, await_promise=True):
        res = await self.send("Runtime.evaluate", {
            "expression": expr,
            "awaitPromise": await_promise,
            "returnByValue": True
        })
        if "exceptionDetails" in res:
            err = res["exceptionDetails"].get("exception", {}).get("description")
            raise RuntimeError(f"Browser JS Exception: {err or res['exceptionDetails']}")
        result_dict = res.get("result", {})
        if "value" in result_dict:
            return result_dict["value"]
        return None

    async def wait_for(self, expr, timeout=15.0, interval=0.2):
        start = time.time()
        while time.time() - start < timeout:
            val = await self.evaluate(expr)
            if val:
                return val
            await asyncio.sleep(interval)
        raise TimeoutError(f"Condition '{expr}' timed out after {timeout}s")


async def run_full_browser_verification():
    results = {}
    server_proc = None
    chrome_proc = None

    browser_path = find_browser_executable()
    app_port = find_free_port()
    cdp_port = find_free_port()
    app_url = f"http://127.0.0.1:{app_port}"

    print(f"Browser Binary: {browser_path}")
    print(f"Target App Port: {app_port} | CDP Port: {cdp_port}")

    if os.path.exists(PROFILE_DIR):
        shutil.rmtree(PROFILE_DIR, ignore_errors=True)

    try:
        print(f"=== Starting Uvicorn Server on port {app_port} ===")
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(app_port)],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        ready = False
        for attempt in range(25):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"{app_url}/health", timeout=1) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode())
                        print(f"Server is ready! Health: {data.get('status')} | Models loaded: {data.get('model_loaded')}")
                        ready = True
                        break
            except Exception:
                pass

        if not ready:
            raise RuntimeError("FastAPI server failed to start within 12 seconds.")

        print(f"=== Launching Browser in Headless Mode ===")
        chrome_proc = subprocess.Popen([
            browser_path,
            "--headless=new",
            f"--remote-debugging-port={cdp_port}",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={PROFILE_DIR}"
        ])

        time.sleep(2)
        with urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json/version") as r:
            version_meta = json.loads(r.read().decode())
        browser_ws = version_meta["webSocketDebuggerUrl"]
        print(f"Connected to Browser CDP WebSocket: {browser_ws}")

        browser = BrowserController(browser_ws)
        await browser.connect()
        await browser.attach_page(app_url)

        print(f"\n--- WORKFLOW 1: Navigating to Homepage ({app_url}) ---")
        await browser.wait_for("document.readyState === 'complete' && document.querySelector('h1') !== null && document.getElementById('domain-select') !== null")
        await asyncio.sleep(1.0)

        title = await browser.evaluate("document.title")
        h1_text = await browser.evaluate("document.querySelector('h1').innerText")
        sub_brand = await browser.evaluate("document.querySelector('.sub-brand').innerText")
        active_tab = await browser.evaluate("document.querySelector('.nav-tab.active').innerText")
        model_status = await browser.evaluate("document.getElementById('model-status-text').innerText")

        print(f"Page Title: {title}")
        print(f"Header: {h1_text}")
        print(f"Subtitle: {sub_brand}")
        print(f"Active Tab: {active_tab.strip()}")
        print(f"Model Status Badge: {model_status}")

        assert "General Text Deception Detection" in title
        assert "General Text Deception Detection" in h1_text
        assert "Dual-Engine System" in sub_brand
        assert model_status == "Dual Engines Active"
        results["Workflow 1: Homepage Rendering"] = "PASSED"

        print(f"\n--- WORKFLOW 2: Hotel Review Domain Classification ---")
        await browser.evaluate("document.getElementById('domain-select').value = 'hotel_review'; onDomainChange();")
        await asyncio.sleep(0.3)
        caveat_text = await browser.evaluate("document.getElementById('domain-caveat-box').innerText")
        pill_text = await browser.evaluate("document.getElementById('domain-validation-pill').innerText")
        print(f"Domain Pill: {pill_text}")
        print(f"Domain Caveat: {caveat_text}")
        assert "88.75% test accuracy" in caveat_text
        assert "Validated Benchmark" in pill_text

        # Click quick sample: Hotel (Truthful)
        await browser.evaluate("loadSampleGeneral('hotel_truthful')")
        char_count = await browser.evaluate("document.getElementById('general-char-count').innerText")
        print(f"Input character count: {char_count}")

        # Click analyze button
        await browser.evaluate("document.getElementById('analyze-general-btn').click()")
        await browser.wait_for("!document.getElementById('general-results-content').classList.contains('hidden')")

        verdict = await browser.evaluate("document.getElementById('general-verdict-badge').innerText")
        score = await browser.evaluate("document.getElementById('general-score-val').innerText")
        conf = await browser.evaluate("document.getElementById('general-confidence-val').innerText")
        prob = await browser.evaluate("document.getElementById('general-prob-val').innerText")
        prob_line = await browser.evaluate("document.getElementById('general-prob-line').innerText")
        heat_count = await browser.evaluate("document.querySelectorAll('#general-highlight-box span').length")
        truth_cues = await browser.evaluate("document.querySelectorAll('#general-truthful-cues li').length")

        print(f"Hotel Review (Truthful) Verdict: {verdict}")
        print(f"Decision Score: {score} {conf}")
        print(f"Calibrated Deception Prob: {prob}")
        print(f"Probability Label Line: {prob_line}")
        print(f"Heatmap token count: {heat_count} | Truthful cues count: {truth_cues}")

        assert verdict.upper() == "TRUTHFUL"
        assert score.startswith("-")
        assert "Platt Calibrated Sigmoid" in prob_line
        assert "Benchmark Resemblance" in prob_line
        assert heat_count > 0
        assert truth_cues > 0

        # Test Hotel (Deceptive) sample
        print("Testing Hotel Review (Deceptive) Sample...")
        await browser.evaluate("loadSampleGeneral('hotel_deceptive')")
        await browser.evaluate("document.getElementById('analyze-general-btn').click()")
        await asyncio.sleep(1.0)

        dec_verdict = await browser.evaluate("document.getElementById('general-verdict-badge').innerText")
        dec_score = await browser.evaluate("document.getElementById('general-score-val').innerText")
        dec_prob = await browser.evaluate("document.getElementById('general-prob-val').innerText")
        print(f"Hotel Review (Deceptive) Verdict: {dec_verdict} | Score: {dec_score} | Prob: {dec_prob}")

        assert dec_verdict.upper() == "DECEPTIVE"
        assert dec_score.startswith("+")
        results["Workflow 2: Hotel Review Domain"] = "PASSED"

        print(f"\n--- WORKFLOW 3: Personal Statement Domain Classification ---")
        await browser.evaluate("document.getElementById('domain-select').value = 'personal_statement'; onDomainChange();")
        await asyncio.sleep(0.3)
        ps_caveat = await browser.evaluate("document.getElementById('domain-caveat-box').innerText")
        ps_pill = await browser.evaluate("document.getElementById('domain-validation-pill').innerText")
        print(f"PS Domain Pill: {ps_pill}")
        print(f"PS Domain Caveat: {ps_caveat}")
        assert "62.55% test accuracy" in ps_caveat
        assert "103 held-out contributors" in ps_caveat
        assert "not universal lie detection" in ps_caveat

        # Enter personal statement text
        statement_text = "I grew up in Michigan and have two older brothers. My mother was an elementary school teacher and my father worked in construction."
        await browser.evaluate(f"document.getElementById('general-text-input').value = {json.dumps(statement_text)}; document.getElementById('general-text-input').dispatchEvent(new Event('input'));")
        await browser.evaluate("document.getElementById('analyze-general-btn').click()")
        await asyncio.sleep(1.2)

        ps_verdict = await browser.evaluate("document.getElementById('general-verdict-badge').innerText")
        ps_score = await browser.evaluate("document.getElementById('general-score-val').innerText")
        ps_prob = await browser.evaluate("document.getElementById('general-prob-val').innerText")
        ps_result_caveat = await browser.evaluate("document.getElementById('general-result-caveat').innerText")
        ps_prob_line = await browser.evaluate("document.getElementById('general-prob-line').innerText")
        ps_highlights = await browser.evaluate("document.querySelectorAll('#general-highlight-box span').length")

        print(f"PS Verdict: {ps_verdict} | Score: {ps_score} | Prob: {ps_prob}")
        print(f"PS Result Caveat Banner: {ps_result_caveat}")
        print(f"PS Probability Label: {ps_prob_line}")
        print(f"PS Token Highlights count: {ps_highlights}")

        assert ps_verdict.upper() in ["TRUTHFUL", "DECEPTIVE"]
        assert "62.55% test accuracy" in ps_result_caveat
        assert "103 held-out contributors" in ps_result_caveat
        assert "not universal" in ps_result_caveat
        assert "Benchmark Resemblance" in ps_prob_line
        assert ps_highlights > 0
        results["Workflow 3: Personal Statement Domain"] = "PASSED"

        print(f"\n--- WORKFLOW 4: Combined Dual-Engine Synthesis ---")
        await browser.evaluate("switchTab('combined')")
        await asyncio.sleep(0.3)
        comb_active = await browser.evaluate("document.getElementById('section-combined').classList.contains('active')")
        assert comb_active is True

        # Part 4A: Without profile selected
        print("Testing Combined Analysis WITHOUT profile...")
        await browser.evaluate("""
            document.getElementById('combined-domain-select').value = 'personal_statement';
            document.getElementById('combined-profile-select').value = '';
            document.getElementById('combined-text-input').value = 'I have worked at the company for over five years as a senior engineer.';
            document.getElementById('combined-text-input').dispatchEvent(new Event('input'));
        """)
        await browser.evaluate("document.getElementById('run-combined-btn').click()")
        await browser.wait_for("!document.getElementById('combined-results-view').classList.contains('hidden')")

        synth_title = await browser.evaluate("document.getElementById('combined-concordance-title').innerText")
        comb_a_class = await browser.evaluate("document.getElementById('comb-a-class').innerText")
        comb_a_prob = await browser.evaluate("document.getElementById('comb-a-prob').innerText")
        comb_b_assessment = await browser.evaluate("document.getElementById('comb-b-assessment').innerText")
        comb_b_summary = await browser.evaluate("document.getElementById('comb-b-summary').innerText")
        method_rule = await browser.evaluate("document.querySelector('.synthesis-rule-alert').innerText")

        print(f"Synthesis Concordance: {synth_title}")
        print(f"Engine A: {comb_a_class} ({comb_a_prob})")
        print(f"Engine B Assessment (No profile): {comb_b_assessment}")
        print(f"Engine B Summary: {comb_b_summary}")
        print(f"Methodological Rule Alert: {method_rule}")

        assert comb_a_class.upper() in ["TRUTHFUL", "DECEPTIVE"]
        assert comb_b_assessment in ["No Profile Selected", "Baseline Incomplete"]
        assert "never combined mathematically" in method_rule

        # Part 4B: With profile selected (Alex Vance)
        print("Testing Combined Analysis WITH profile...")
        options_count = await browser.evaluate("document.getElementById('combined-profile-select').options.length")
        print(f"Combined profile select options count: {options_count}")
        if options_count > 1:
            await browser.evaluate("(() => { const s = document.getElementById('combined-profile-select'); s.selectedIndex = 1; s.value = s.options[1].value; })()")
            await browser.evaluate("document.getElementById('run-combined-btn').click()")
            await browser.wait_for("document.getElementById('comb-b-alignment').innerText !== 'Not Evaluated'", timeout=10.0)
            
            comb_b_align = await browser.evaluate("document.getElementById('comb-b-alignment').innerText")
            comb_b_samples = await browser.evaluate("document.getElementById('comb-b-samples').innerText")
            comb_b_rel = await browser.evaluate("document.getElementById('comb-b-reliability').innerText")
            comb_b_eval = await browser.evaluate("document.getElementById('comb-b-assessment').innerText")
            print(f"Engine B with Profile: Alignment={comb_b_align} | Assessment={comb_b_eval} | Samples={comb_b_samples} | Reliability={comb_b_rel}")
            assert "%" in comb_b_align
            assert "samples" in comb_b_samples
        results["Workflow 4: Combined Dual-Engine Synthesis"] = "PASSED"

        print(f"\n--- WORKFLOW 5: Personal Stylometry (Engine B) ---")
        await browser.evaluate("switchTab('engine-b')")
        await asyncio.sleep(0.5)

        active_prof_name = await browser.evaluate("document.getElementById('active-profile-name').innerText")
        active_prof_samples = await browser.evaluate("document.getElementById('active-profile-samples').innerText")
        active_prof_badge = await browser.evaluate("document.getElementById('active-profile-badge').innerText")
        print(f"Active Profile: {active_prof_name} | Samples: {active_prof_samples} | Rating: {active_prof_badge}")

        # Consistent Sample
        print("Testing Consistent Sample on Engine B...")
        await browser.evaluate("loadSamplePersonal('matching')")
        await browser.evaluate("document.getElementById('compare-personal-btn').click()")
        await browser.wait_for("!document.getElementById('personal-results-content').classList.contains('hidden')")

        match_align = await browser.evaluate("document.getElementById('alignment-index-val').innerText")
        match_title = await browser.evaluate("document.getElementById('alignment-assessment-title').innerText")
        stable_count = await browser.evaluate("document.querySelectorAll('#stable-features-chips span').length")
        disclaimer = await browser.evaluate("document.querySelector('#section-engine-b .scientific-disclaimer').innerText")
        print(f"Matching sample Alignment: {match_align} | Assessment: {match_title} | Stable Features: {stable_count}")
        print(f"Scientific Disclaimer: {disclaimer}")
        assert "Consistent" in match_title or "Moderate" in match_title
        assert "Stylometric divergence measures changes in writing habits, NOT deception" in disclaimer

        # Divergent Sample
        print("Testing Divergent Sample on Engine B...")
        await browser.evaluate("loadSamplePersonal('divergent')")
        await browser.evaluate("document.getElementById('compare-personal-btn').click()")
        await asyncio.sleep(0.8)

        div_align = await browser.evaluate("document.getElementById('alignment-index-val').innerText")
        div_title = await browser.evaluate("document.getElementById('alignment-assessment-title').innerText")
        shifted_rows = await browser.evaluate("document.querySelectorAll('#shifted-features-tbody tr').length")
        alt_exps = await browser.evaluate("document.querySelectorAll('#alt-explanations-list li').length")
        print(f"Divergent Alignment: {div_align} | Assessment: {div_title} | Shifted Rows: {shifted_rows} | Alt Explanations: {alt_exps}")
        assert shifted_rows > 0
        assert alt_exps > 0
        results["Workflow 5: Personal Stylometry"] = "PASSED"

        print(f"\n--- WORKFLOW 6: Input Validation, Empty Inputs & Error Handling ---")
        browser.dialogs.clear()

        # 6A: Empty input on Engine A
        await browser.evaluate("switchTab('engine-a')")
        await browser.evaluate("clearGeneralText()")
        await browser.evaluate("document.getElementById('analyze-general-btn').click()")
        await asyncio.sleep(0.5)
        print(f"Captured Dialogs after Engine A empty submit: {browser.dialogs}")
        assert any("Please enter text to classify" in d for d in browser.dialogs)
        browser.dialogs.clear()

        # 6B: Empty input on Engine B
        await browser.evaluate("switchTab('engine-b')")
        await browser.evaluate("clearPersonalText()")
        await browser.evaluate("document.getElementById('compare-personal-btn').click()")
        await asyncio.sleep(0.5)
        print(f"Captured Dialogs after Engine B empty submit: {browser.dialogs}")
        assert any("Please enter the author's current text to compare" in d for d in browser.dialogs)
        browser.dialogs.clear()

        # 6C: Empty input on Combined
        await browser.evaluate("switchTab('combined')")
        await browser.evaluate("document.getElementById('combined-text-input').value = ''; document.getElementById('combined-text-input').dispatchEvent(new Event('input'));")
        await browser.evaluate("document.getElementById('run-combined-btn').click()")
        await asyncio.sleep(0.5)
        print(f"Captured Dialogs after Combined empty submit: {browser.dialogs}")
        assert any("Please enter text for dual-engine synthesis" in d for d in browser.dialogs)
        results["Workflow 6: Input Validation & Error Handling"] = "PASSED"

        print(f"\n--- WORKFLOW 7: Metrics Display & Deception Claims Audit ---")
        await browser.evaluate("switchTab('metrics')")
        await asyncio.sleep(1.0)
        metrics_active = await browser.evaluate("document.getElementById('section-metrics').classList.contains('active')")
        assert metrics_active is True

        acc_val = await browser.evaluate("document.getElementById('metric-acc-val').innerText")
        cal_acc_val = await browser.evaluate("document.getElementById('metric-cal-acc-val').innerText")
        print(f"Metrics Card - Linear SVM: {acc_val} | Calibrated SVM: {cal_acc_val}")
        assert "90.62%" in acc_val
        assert "88.75%" in cal_acc_val

        full_metrics_text = await browser.evaluate("document.getElementById('section-metrics').innerText")
        print(f"Auditing Metrics text claims in DOM...")
        assert "88.75% Test Acc" in full_metrics_text
        assert "62.55% Test Acc" in full_metrics_text
        assert "103 Held-Out Authors" in full_metrics_text
        assert "Empirical Transfer Failure" in full_metrics_text
        assert "48.32% Test Acc" in full_metrics_text
        assert "49.69% Test Acc" in full_metrics_text
        assert "Phase 10: Privacy, Security & Data Governance" in full_metrics_text
        assert "Local Storage Isolation" in full_metrics_text
        assert "Zero External Transmission" in full_metrics_text

        # Scientific disclaimer on Engine A
        engine_a_disclaimer = await browser.evaluate("document.querySelector('.scientific-disclaimer').innerText")
        print(f"Scientific Disclaimer on Engine A: {engine_a_disclaimer}")
        assert "does NOT prove whether an author is lying" in engine_a_disclaimer
        results["Workflow 7: Metrics Display & Deception Claims Audit"] = "PASSED"

        print("\n=======================================================")
        print("ALL 7 WORKFLOWS VERIFIED IN REAL INSTALLED BROWSER VIA CDP!")
        print("=======================================================")
        for k, v in results.items():
            print(f"  {k}: {v}")

    finally:
        if chrome_proc:
            try:
                chrome_proc.kill()
            except Exception:
                pass
        if server_proc:
            try:
                server_proc.kill()
            except Exception:
                pass
        time.sleep(1)
        if os.path.exists(PROFILE_DIR):
            shutil.rmtree(PROFILE_DIR, ignore_errors=True)

if __name__ == "__main__":
    asyncio.run(run_full_browser_verification())
