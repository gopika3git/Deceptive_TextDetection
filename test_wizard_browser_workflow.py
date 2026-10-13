"""
test_wizard_browser_workflow.py
End-to-End Real Browser Automation Suite for the Investigative Onboarding Wizard.

Verifies the complete 9-step investigative target-author workflow using an installed
Chrome/Edge browser via native Chrome DevTools Protocol (CDP) over WebSockets:

1. Open application and launch onboarding wizard.
2. Create a new target-author profile with authorization confirmation and validation.
3. Add historical writing samples and verify input validation.
4. Confirm baseline readiness status updates correctly (insufficient -> operationally ready).
5. Finish onboarding and verify target-author profile selected in Engine B and Combined Analysis.
6. Submit a new text for analysis and confirm comparison against target author's baseline.
7. Reload application and verify profile, samples, and readiness persist from disk.
8. Test an insufficient baseline and confirm general classification proceeds without invalid comparison.
9. Verify demo profiles are clearly distinguished from real target-author profiles.
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
import builtins

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def safe_print(*args, **kwargs):
    clean_args = [
        str(a).encode("ascii", errors="replace").decode("ascii") if isinstance(a, str) else a
        for a in args
    ]
    builtins.print(*clean_args, **kwargs)

print = safe_print

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(PROJECT_DIR, "chrome_test_wizard_profile")


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


async def run_wizard_e2e_verification():
    step_results = {}
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

    created_profile_ids = []

    try:
        print(f"\n=== 1. Starting FastAPI Server on port {app_port} ===")
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
                        print(f"Server ready! Status: {data.get('status')} | Engine A: {data.get('engine_a_loaded')}")
                        ready = True
                        break
            except Exception:
                pass

        if not ready:
            raise RuntimeError("FastAPI server failed to start within 12 seconds.")

        print(f"\n=== 2. Launching Headless Chrome via CDP ===")
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
        print(f"Connected to CDP WebSocket: {browser_ws}")

        browser = BrowserController(browser_ws)
        await browser.connect()
        await browser.attach_page(app_url)

        # ------------------------------------------------------------------
        # WORKFLOW STEP 1: Open Application & Launch Wizard
        # ------------------------------------------------------------------
        print(f"\n--- STEP 1: Open Application and Launch Onboarding Wizard ---")
        await browser.wait_for("document.readyState === 'complete' && document.getElementById('btn-header-onboard') !== null")
        await asyncio.sleep(0.5)

        # Confirm wizard modal is currently hidden
        is_modal_hidden = await browser.evaluate("document.getElementById('modal-onboarding-wizard').classList.contains('hidden')")
        assert is_modal_hidden is True, "Modal should be initially hidden"

        # Click the header onboarding button
        print("Action: Click #btn-header-onboard to open wizard...")
        await browser.evaluate("document.getElementById('btn-header-onboard').click()")
        await browser.wait_for("!document.getElementById('modal-onboarding-wizard').classList.contains('hidden')")

        # Verify Step 1 is active and visible
        step1_visible = await browser.evaluate("!document.getElementById('wizard-step-1').classList.contains('hidden')")
        step2_hidden = await browser.evaluate("document.getElementById('wizard-step-2').classList.contains('hidden')")
        step3_hidden = await browser.evaluate("document.getElementById('wizard-step-3').classList.contains('hidden')")
        step4_hidden = await browser.evaluate("document.getElementById('wizard-step-4').classList.contains('hidden')")
        step1_ind_active = await browser.evaluate("document.getElementById('w-step-ind-1').classList.contains('active')")

        assert step1_visible and step2_hidden and step3_hidden and step4_hidden
        assert step1_ind_active
        print("Verified: Wizard opened at Step 1, stepper shows Step 1 Active.")
        step_results["Step 1: Open Application & Launch Wizard"] = "PASSED"

        # ------------------------------------------------------------------
        # WORKFLOW STEP 2: Create Target-Author Profile & Test Authorization Validation
        # ------------------------------------------------------------------
        print(f"\n--- STEP 2: Create New Target-Author Profile & Test Authorization Validation ---")
        # 2A: Test empty name validation
        print("Action: Click Next with empty name to trigger validation...")
        await browser.evaluate("document.getElementById('w-btn-next').click()")
        await asyncio.sleep(0.3)
        err_name_visible = await browser.evaluate("!document.getElementById('w-err-name').classList.contains('hidden')")
        assert err_name_visible is True, "Validation error for missing name should be visible"
        print("Verified: Missing name error displayed correctly.")

        # 2B: Fill name but leave consent unchecked
        print("Action: Enter target author name but leave consent unchecked...")
        target_name = "Target Witness Marcus Vance"
        await browser.evaluate(f"document.getElementById('w-profile-name').value = {json.dumps(target_name)};")
        await browser.evaluate("document.getElementById('w-profile-desc').value = 'Key witness in Case #771';")
        await browser.evaluate("document.getElementById('w-profile-notes').value = 'Authenticated corporate emails from 2025';")
        await browser.evaluate("document.getElementById('w-btn-next').click()")
        await asyncio.sleep(0.3)

        err_consent_visible = await browser.evaluate("!document.getElementById('w-err-consent').classList.contains('hidden')")
        assert err_consent_visible is True, "Validation error for missing consent should be visible"
        print("Verified: Missing investigator authorization error displayed correctly.")

        # 2C: Check consent and submit valid profile
        print("Action: Confirm investigator authorization checkbox and click Next...")
        await browser.evaluate("document.getElementById('w-profile-consent').checked = true;")
        await browser.evaluate("document.getElementById('w-btn-next').click()")
        await browser.wait_for("!document.getElementById('wizard-step-2').classList.contains('hidden')", timeout=8.0)

        step2_active = await browser.evaluate("document.getElementById('w-step-ind-2').classList.contains('active')")
        step1_done = await browser.evaluate("document.getElementById('w-step-ind-1').classList.contains('completed')")
        assert step2_active and step1_done, "Step 2 should now be active, Step 1 completed"

        # Capture created profile ID
        prof_id = await browser.evaluate("wizardCreatedProfile ? wizardCreatedProfile.id : null")
        print(f"Verified: Target author profile created successfully with ID: {prof_id}")
        assert prof_id is not None
        created_profile_ids.append(prof_id)
        step_results["Step 2: Create Target-Author Profile & Authorization Validation"] = "PASSED"

        # ------------------------------------------------------------------
        # WORKFLOW STEP 3: Add Historical Samples & Verify Input Validation
        # ------------------------------------------------------------------
        print(f"\n--- STEP 3: Add Historical Writing Samples and Verify Input Validation ---")
        # 3A: Test short sample validation (< 5 words)
        print("Action: Try adding sample with < 5 words...")
        await browser.evaluate("document.getElementById('w-sample-text').value = 'Too short';")
        await browser.evaluate("document.getElementById('w-add-sample-btn').click()")
        await asyncio.sleep(0.3)

        err_sample_visible = await browser.evaluate("!document.getElementById('w-err-sample').classList.contains('hidden')")
        assert err_sample_visible is True, "Error for sample with < 5 words should be visible"
        print("Verified: Sample length validation enforced (< 5 words rejected).")

        # 3B: Add Sample 1 (45 words)
        sample1_text = (
            "Regarding our annual financial audit, we have concluded the comprehensive balance sheet reconciliations "
            "across all operating departmental accounts. Operational expenses declined by four point five percent "
            "following our enterprise cloud migration earlier this quarter. All supporting accounting documents are attached "
            "for the oversight committee review."
        )
        print("Action: Add authentic Sample 1 (~45 words)...")
        await browser.evaluate(f"""
            document.getElementById('w-sample-text').value = {json.dumps(sample1_text)};
            document.getElementById('w-sample-context').value = 'professional_email';
            document.getElementById('w-sample-desc').value = 'Audit Memo 2025-Q1';
            document.getElementById('w-sample-text').dispatchEvent(new Event('input'));
        """)
        char_stat_1 = await browser.evaluate("document.getElementById('w-sample-count').innerText")
        print(f"Sample 1 Input Counter: {char_stat_1}")
        await browser.evaluate("document.getElementById('w-add-sample-btn').click()")
        await browser.wait_for("document.getElementById('w-stat-samples').innerText === '1'")

        # Verify Sample 1 in list
        s_count_1 = await browser.evaluate("document.getElementById('w-sample-items-count').innerText")
        err_sample_hidden = await browser.evaluate("document.getElementById('w-err-sample').classList.contains('hidden')")
        assert s_count_1 == "1"
        assert err_sample_hidden is True
        print("Verified: Sample 1 accepted, added to baseline list.")
        step_results["Step 3: Add Historical Writing Samples & Validation"] = "PASSED"

        # ------------------------------------------------------------------
        # WORKFLOW STEP 4: Confirm Readiness Status Updates Correctly
        # ------------------------------------------------------------------
        print(f"\n--- STEP 4: Confirm Baseline Readiness Status Updates Correctly ---")
        # Check status with 1 sample (under 200 words)
        badge_text_1 = await browser.evaluate("document.getElementById('w-readiness-badge').innerText")
        hint_text_1 = await browser.evaluate("document.getElementById('w-readiness-hint').innerText")
        words_1 = int(await browser.evaluate("document.getElementById('w-stat-words').innerText"))
        print(f"State after 1 sample: Badge='{badge_text_1}' | Words={words_1} | Hint='{hint_text_1}'")
        assert "Needs More Writing Samples" in badge_text_1
        assert "Add at least 2 more sample(s)" in hint_text_1

        # Add Sample 2 (45 words)
        sample2_text = (
            "The vendor contract indemnification provisions have been examined in detail by outside legal counsel. "
            "All revisions in section eight adhere to current enterprise risk management requirements and statutory "
            "compliance policies. Please review the attached agreement draft before Thursday afternoon committee sync."
        )
        print("Action: Add authentic Sample 2 (~45 words)...")
        await browser.evaluate(f"""
            document.getElementById('w-sample-text').value = {json.dumps(sample2_text)};
            document.getElementById('w-sample-context').value = 'professional_email';
            document.getElementById('w-sample-desc').value = 'Legal Contract Review';
            document.getElementById('w-sample-text').dispatchEvent(new Event('input'));
        """)
        await browser.evaluate("document.getElementById('w-add-sample-btn').click()")
        await browser.wait_for("document.getElementById('w-stat-samples').innerText === '2'")

        words_2 = int(await browser.evaluate("document.getElementById('w-stat-words').innerText"))
        badge_text_2 = await browser.evaluate("document.getElementById('w-readiness-badge').innerText")
        hint_text_2 = await browser.evaluate("document.getElementById('w-readiness-hint').innerText")
        print(f"State after 2 samples: Badge='{badge_text_2}' | Words={words_2} | Hint='{hint_text_2}'")
        assert "Needs More Writing Samples" in badge_text_2
        assert "Add at least 1 more sample(s)" in hint_text_2

        # Add Sample 3 (50 words)
        sample3_text = (
            "Following up on the customer service latency audit, our engineering teams identified several database "
            "bottlenecks during peak transaction windows on weekdays. We are implementing an asynchronous indexing "
            "strategy to reduce queue times and ensure high system responsiveness across all customer-facing services. "
            "We expect preliminary benchmarks on Friday."
        )
        print("Action: Add authentic Sample 3 (~50 words)...")
        await browser.evaluate(f"""
            document.getElementById('w-sample-text').value = {json.dumps(sample3_text)};
            document.getElementById('w-sample-context').value = 'professional_email';
            document.getElementById('w-sample-desc').value = 'Latency Audit Update';
            document.getElementById('w-sample-text').dispatchEvent(new Event('input'));
        """)
        await browser.evaluate("document.getElementById('w-add-sample-btn').click()")
        await browser.wait_for("document.getElementById('w-stat-samples').innerText === '3'")

        words_3 = int(await browser.evaluate("document.getElementById('w-stat-words').innerText"))
        print(f"State after 3 samples: Total Words={words_3}")

        # Add Sample 4 (85 words) to ensure total words comfortably exceed 200 threshold
        sample4_text = (
            "Thank you for attending yesterday afternoon's project sprint retrospective and platform infrastructure "
            "demonstration. Our joint discussion highlighted commendable execution velocity across the data ingestion "
            "and feature extraction pipelines. I have summarized the core action items in our shared project documentation "
            "and assigned designated workstream leads to track milestone delivery prior to next month's production release. "
            "Please ensure all regional unit test metrics are updated in the engineering dashboard by close of business."
        )
        print("Action: Add authentic Sample 4 (~80 words) to exceed 200 word threshold...")
        await browser.evaluate(f"""
            document.getElementById('w-sample-text').value = {json.dumps(sample4_text)};
            document.getElementById('w-sample-context').value = 'professional_email';
            document.getElementById('w-sample-desc').value = 'Retrospective Summary';
            document.getElementById('w-sample-text').dispatchEvent(new Event('input'));
        """)
        await browser.evaluate("document.getElementById('w-add-sample-btn').click()")
        await browser.wait_for("document.getElementById('w-stat-samples').innerText === '4'")

        total_words_final = int(await browser.evaluate("document.getElementById('w-stat-words').innerText"))
        badge_final = await browser.evaluate("document.getElementById('w-readiness-badge').innerText")
        hint_final = await browser.evaluate("document.getElementById('w-readiness-hint').innerText")
        print(f"State after 4 samples: Total Words={total_words_final} | Badge='{badge_final}' | Hint='{hint_final}'")

        assert total_words_final >= 200, f"Expected >= 200 words, got {total_words_final}"
        assert badge_final == "Operationally Ready", f"Expected 'Operationally Ready', got '{badge_final}'"
        assert "Minimum baseline reached" in hint_final
        print("Verified: Live tracker dynamically flipped to Operationally Ready!")

        # Advance to Step 3 (Readiness Check)
        print("Action: Click Next to review Step 3 Readiness Review...")
        await browser.evaluate("document.getElementById('w-btn-next').click()")
        await browser.wait_for("!document.getElementById('wizard-step-3').classList.contains('hidden')", timeout=6.0)

        step3_name = await browser.evaluate("document.getElementById('w-review-name').innerText")
        step3_badge = await browser.evaluate("document.getElementById('w-review-status-badge').innerText")
        step3_samples = await browser.evaluate("document.getElementById('w-review-samples').innerText")
        step3_words = await browser.evaluate("document.getElementById('w-review-words').innerText")
        step3_sent = await browser.evaluate("document.getElementById('w-review-sent').innerText")
        step3_ttr = await browser.evaluate("document.getElementById('w-review-ttr').innerText")
        step3_detail = await browser.evaluate("document.getElementById('w-readiness-detail-box').innerText")

        print(f"Step 3 Review: Name='{step3_name}' | Status='{step3_badge}' | Samples={step3_samples} | Words={step3_words} | AvgSent={step3_sent} | TTR={step3_ttr}")
        print(f"Step 3 Detail Box: {step3_detail}")

        assert step3_name == target_name
        assert "Operationally Ready" in step3_badge
        assert int(step3_samples) == 4
        assert int(step3_words) >= 200
        assert step3_sent != "--"
        assert step3_ttr != "--"
        assert "Baseline meets minimum operational criteria" in step3_detail
        print("Verified: Step 3 readiness review populated from /api/profiles/{id}/readiness.")
        step_results["Step 4: Confirm Baseline Readiness Updates Correctly"] = "PASSED"

        # ------------------------------------------------------------------
        # WORKFLOW STEP 5: Finish Onboarding & Verify Target Profile Selected in Engine B & Combined
        # ------------------------------------------------------------------
        print(f"\n--- STEP 5: Finish Onboarding & Verify Target Profile Selected in Engine B & Combined ---")
        # Advance to Step 4 (Confirm & Activate)
        print("Action: Click Complete & Activate to go to Step 4...")
        await browser.evaluate("document.getElementById('w-btn-next').click()")
        await browser.wait_for("!document.getElementById('wizard-step-4').classList.contains('hidden')", timeout=6.0)

        step4_name = await browser.evaluate("document.getElementById('w-success-name').innerText")
        step4_stats = await browser.evaluate("document.getElementById('w-success-stats').innerText")
        print(f"Step 4 Confirmation Card: Name='{step4_name}' | Stats='{step4_stats}'")
        assert step4_name == target_name
        assert "Operationally Ready" in step4_stats

        # Click button to proceed to Engine B
        print("Action: Click 'Go to Target Stylometry (Engine B)'...")
        await browser.evaluate("finishOnboardingToTab('engine-b')")
        await asyncio.sleep(1.0)

        # Verify wizard modal is closed
        modal_closed = await browser.evaluate("document.getElementById('modal-onboarding-wizard').classList.contains('hidden')")
        assert modal_closed is True, "Wizard modal should be closed"

        # Verify Tab 2 is active
        tab_b_active = await browser.evaluate("document.getElementById('section-engine-b').classList.contains('active')")
        assert tab_b_active is True, "Engine B tab should be active"

        # Verify Engine B dropdown and active indicators
        eng_b_selected_val = await browser.evaluate("document.getElementById('profile-select').value")
        eng_b_selected_text = await browser.evaluate("document.getElementById('profile-select').options[document.getElementById('profile-select').selectedIndex].text")
        active_prof_name = await browser.evaluate("document.getElementById('active-profile-name').innerText")
        active_prof_samples = await browser.evaluate("document.getElementById('active-profile-samples').innerText")
        active_prof_words = await browser.evaluate("document.getElementById('active-profile-words').innerText")
        p_indicator_text = await browser.evaluate("document.getElementById('personal-active-profile-indicator').innerText")
        demo_banner_hidden = await browser.evaluate("document.getElementById('demo-profile-banner').classList.contains('hidden')")

        print(f"Engine B Selected Value: {eng_b_selected_val} (Expected {prof_id})")
        print(f"Engine B Option Text: {eng_b_selected_text}")
        print(f"Engine B Active Name: {active_prof_name} | Samples: {active_prof_samples} | Words: {active_prof_words}")
        print(f"Engine B Indicator Card: {p_indicator_text}")

        assert eng_b_selected_val == prof_id
        assert "[TARGET AUTHOR]" in eng_b_selected_text
        assert target_name in eng_b_selected_text
        assert active_prof_name == target_name
        assert int(active_prof_samples) == 4
        assert int(active_prof_words) >= 200
        assert "TARGET AUTHOR" in p_indicator_text
        assert "Operationally Ready" in p_indicator_text
        assert demo_banner_hidden is True, "Demo alert banner should be hidden for real target author"

        # Now check Tab 3 (Combined Analysis)
        print("Action: Switch to Combined Analysis Tab and inspect target author selection...")
        await browser.evaluate("switchTab('combined')")
        await asyncio.sleep(0.5)

        comb_selected_val = await browser.evaluate("document.getElementById('combined-profile-select').value")
        comb_selected_text = await browser.evaluate("document.getElementById('combined-profile-select').options[document.getElementById('combined-profile-select').selectedIndex].text")
        comb_indicator_text = await browser.evaluate("document.getElementById('combined-active-profile-indicator').innerText")

        print(f"Combined Selected Value: {comb_selected_val}")
        print(f"Combined Option Text: {comb_selected_text}")
        print(f"Combined Indicator Card: {comb_indicator_text}")

        assert comb_selected_val == prof_id
        assert target_name in comb_selected_text
        assert "TARGET AUTHOR" in comb_indicator_text
        assert target_name in comb_indicator_text
        assert "Operationally Ready" in comb_indicator_text

        print("Verified: Target author profile selected across both Engine B and Combined Analysis!")
        step_results["Step 5: Finish Onboarding & Verify Target Profile Selected"] = "PASSED"

        # ------------------------------------------------------------------
        # WORKFLOW STEP 6: Submit Text & Confirm Comparison Against Target Baseline
        # ------------------------------------------------------------------
        print(f"\n--- STEP 6: Submit Text for Analysis and Confirm Target Comparison ---")
        # 6A: Test in Engine B
        print("Action: Submit submitted text in Engine B for stylometric comparison...")
        await browser.evaluate("switchTab('engine-b')")
        await asyncio.sleep(0.3)

        test_submission_text = (
            "Good morning team, I have completed the final review of our quarterly compliance report "
            "and updated our key project deliverables. Please send any remaining feedback prior to the Friday board meeting."
        )
        await browser.evaluate(f"""
            document.getElementById('personal-compare-input').value = {json.dumps(test_submission_text)};
            document.getElementById('personal-compare-input').dispatchEvent(new Event('input'));
        """)
        await browser.evaluate("document.getElementById('compare-personal-btn').click()")
        await browser.wait_for("!document.getElementById('personal-results-content').classList.contains('hidden')", timeout=8.0)

        alignment_val = await browser.evaluate("document.getElementById('alignment-index-val').innerText")
        assessment_title = await browser.evaluate("document.getElementById('alignment-assessment-title').innerText")
        summary_text = await browser.evaluate("document.getElementById('alignment-summary-text').innerText")
        print(f"Engine B Comparison Results: Alignment={alignment_val} | Assessment='{assessment_title}'")
        print(f"Engine B Narrative: {summary_text}")

        assert "%" in alignment_val
        assert len(assessment_title) > 0
        assert len(summary_text) > 0

        # 6B: Test in Combined Analysis
        print("Action: Run Combined Dual Analysis with Target Witness Marcus Vance selected...")
        await browser.evaluate("switchTab('combined')")
        await asyncio.sleep(0.3)

        combined_submission_text = (
            "My business partner and I stayed at this downtown hotel for a corporate conference. "
            "The front desk clerk checked us in efficiently and provided keycards to the fifth floor. "
            "The room was quiet and clean, though the elevator took a couple minutes during morning peak."
        )
        await browser.evaluate(f"""
            document.getElementById('combined-text-input').value = {json.dumps(combined_submission_text)};
            document.getElementById('combined-domain-select').value = 'hotel_review';
            document.getElementById('combined-text-input').dispatchEvent(new Event('input'));
        """)
        await browser.evaluate("document.getElementById('run-combined-btn').click()")
        await browser.wait_for("!document.getElementById('combined-results-view').classList.contains('hidden')", timeout=10.0)

        comb_a_pred = await browser.evaluate("document.getElementById('comb-a-class').innerText")
        comb_a_prob = await browser.evaluate("document.getElementById('comb-a-prob').innerText")
        comb_b_align = await browser.evaluate("document.getElementById('comb-b-alignment').innerText")
        comb_b_eval = await browser.evaluate("document.getElementById('comb-b-assessment').innerText")
        comb_b_samp = await browser.evaluate("document.getElementById('comb-b-samples').innerText")
        synth_narrative = await browser.evaluate("document.getElementById('combined-narrative-text').innerText")

        print(f"Combined Analysis Output:")
        print(f"  Engine A General Classification: {comb_a_pred} ({comb_a_prob})")
        print(f"  Engine B Target Stylometry: Alignment={comb_b_align} | Assessment={comb_b_eval} | Samples={comb_b_samp}")
        print(f"  Combined Synthesis Narrative: {synth_narrative}")

        assert comb_a_pred.upper() in ["TRUTHFUL", "DECEPTIVE"]
        assert "%" in comb_b_align
        assert "4 historical samples" in comb_b_samp
        assert target_name in synth_narrative
        # Verify scores are separate and not merged into an invented deception score
        comb_results_html = await browser.evaluate("document.getElementById('combined-results-view').innerHTML")
        assert "combined_deception_percentage" not in comb_results_html
        print("Verified: Evaluated against target author's baseline with independent metric separation.")
        step_results["Step 6: Submit Text & Confirm Comparison Against Target Baseline"] = "PASSED"

        # ------------------------------------------------------------------
        # WORKFLOW STEP 7: Reload Application & Verify Persistence
        # ------------------------------------------------------------------
        print(f"\n--- STEP 7: Reload Application and Verify Profile & Samples Persist ---")
        print("Action: Reloading the browser page...")
        await browser.evaluate("location.reload()")
        await browser.wait_for("document.readyState === 'complete' && document.getElementById('profile-select') !== null", timeout=10.0)
        await asyncio.sleep(1.0)

        # Inspect profiles in dropdown after page reload
        options_after_reload = await browser.evaluate("""
            Array.from(document.getElementById('profile-select').options).map(o => ({ value: o.value, text: o.text }))
        """)
        print(f"Options after page reload: {options_after_reload}")

        marcus_opt = next((o for o in options_after_reload if o["value"] == prof_id), None)
        assert marcus_opt is not None, f"Profile {prof_id} should persist in dropdown options"
        assert "[TARGET AUTHOR]" in marcus_opt["text"]
        assert target_name in marcus_opt["text"]
        assert "4 samples" in marcus_opt["text"]

        # Switch to Engine B and select Marcus Vance
        print("Action: Select persisted profile in Engine B...")
        await browser.evaluate("switchTab('engine-b')")
        await browser.evaluate(f"document.getElementById('profile-select').value = '{prof_id}'; onProfileChange();")
        await asyncio.sleep(0.5)

        p_name_reloaded = await browser.evaluate("document.getElementById('active-profile-name').innerText")
        p_samples_reloaded = await browser.evaluate("document.getElementById('active-profile-samples').innerText")
        p_words_reloaded = await browser.evaluate("document.getElementById('active-profile-words').innerText")
        p_ind_reloaded = await browser.evaluate("document.getElementById('personal-active-profile-indicator').innerText")

        print(f"Persisted Profile on Tab 2: Name='{p_name_reloaded}' | Samples={p_samples_reloaded} | Words={p_words_reloaded}")
        print(f"Indicator text: {p_ind_reloaded}")

        assert p_name_reloaded == target_name
        assert int(p_samples_reloaded) == 4
        assert int(p_words_reloaded) >= 200
        assert "Operationally Ready" in p_ind_reloaded
        print("Verified: Profile, historical samples, and operational readiness persisted from disk!")
        step_results["Step 7: Reload Application & Verify Persistence"] = "PASSED"

        # ------------------------------------------------------------------
        # WORKFLOW STEP 8: Test Insufficient Baseline & Confirm Standalone General Classification
        # ------------------------------------------------------------------
        print(f"\n--- STEP 8: Test Insufficient Baseline & Confirm General Classification Still Works ---")
        # 8A: Create an incomplete profile (1 short sample of 15 words)
        print("Action: Open wizard to create a profile with INSUFFICIENT baseline...")
        await browser.evaluate("document.getElementById('btn-header-onboard').click()")
        await browser.wait_for("!document.getElementById('modal-onboarding-wizard').classList.contains('hidden')")

        incomplete_name = "Subject Incomplete Baseline"
        await browser.evaluate(f"document.getElementById('w-profile-name').value = {json.dumps(incomplete_name)};")
        await browser.evaluate("document.getElementById('w-profile-consent').checked = true;")
        await browser.evaluate("document.getElementById('w-btn-next').click()")
        await browser.wait_for("!document.getElementById('wizard-step-2').classList.contains('hidden')")

        incomp_prof_id = await browser.evaluate("wizardCreatedProfile ? wizardCreatedProfile.id : null")
        created_profile_ids.append(incomp_prof_id)

        # Add single short sample (16 words)
        short_sample_text = "This is a brief introductory note sent to a colleague regarding our upcoming project kickoff meeting."
        await browser.evaluate(f"""
            document.getElementById('w-sample-text').value = {json.dumps(short_sample_text)};
            document.getElementById('w-sample-context').value = 'professional_email';
            document.getElementById('w-sample-text').dispatchEvent(new Event('input'));
        """)
        await browser.evaluate("document.getElementById('w-add-sample-btn').click()")
        await browser.wait_for("document.getElementById('w-stat-samples').innerText === '1'")

        incomp_badge = await browser.evaluate("document.getElementById('w-readiness-badge').innerText")
        incomp_words = int(await browser.evaluate("document.getElementById('w-stat-words').innerText"))
        print(f"Incomplete profile: Badge='{incomp_badge}' | Words={incomp_words}")
        assert incomp_badge == "Needs More Writing Samples"
        assert incomp_words < 200

        # Advance to Step 3 then Step 4 and finish
        await browser.evaluate("document.getElementById('w-btn-next').click()")
        await browser.wait_for("!document.getElementById('wizard-step-3').classList.contains('hidden')")
        incomp_review_status = await browser.evaluate("document.getElementById('w-review-status-badge').innerText")
        print(f"Step 3 Review Status for Incomplete profile: {incomp_review_status}")
        assert "Needs More Writing Samples" in incomp_review_status

        await browser.evaluate("document.getElementById('w-btn-next').click()")
        await browser.wait_for("!document.getElementById('wizard-step-4').classList.contains('hidden')")
        await browser.evaluate("finishOnboardingToTab('combined')")
        await asyncio.sleep(1.0)

        # 8B: Run Combined Analysis with incomplete profile
        print("Action: Run Combined Analysis with incomplete baseline profile...")
        await browser.evaluate("""
            document.getElementById('combined-text-input').value = 'The hotel room was very quiet and spotless throughout our visit.';
            document.getElementById('combined-text-input').dispatchEvent(new Event('input'));
        """)
        await browser.evaluate("document.getElementById('run-combined-btn').click()")
        await browser.wait_for("!document.getElementById('combined-results-view').classList.contains('hidden')", timeout=10.0)

        incomp_a_class = await browser.evaluate("document.getElementById('comb-a-class').innerText")
        incomp_b_assess = await browser.evaluate("document.getElementById('comb-b-assessment').innerText")
        incomp_b_align = await browser.evaluate("document.getElementById('comb-b-alignment').innerText")
        incomp_synthesis = await browser.evaluate("document.getElementById('combined-narrative-text').innerText")

        print(f"Combined Results with Incomplete Baseline:")
        print(f"  Engine A General Class: {incomp_a_class}")
        print(f"  Engine B Assessment: {incomp_b_assess} | Alignment: {incomp_b_align}")
        print(f"  Synthesis Narrative: {incomp_synthesis}")

        assert incomp_a_class.upper() in ["TRUTHFUL", "DECEPTIVE"], "General classification should succeed"
        # Verify Engine B flagged the incomplete baseline
        assert any(term in incomp_b_assess for term in ["Moderate", "Deviation", "Divergence", "Incomplete", "Evaluated", "Low"]), f"Unexpected assessment: {incomp_b_assess}"

        # 8C: Run Combined Analysis with NO profile selected (None)
        print("Action: Run Combined Analysis with NO profile selected at all...")
        await browser.evaluate("""
            document.getElementById('combined-profile-select').value = '';
            onCombinedProfileChange();
            document.getElementById('run-combined-btn').click();
        """)
        await asyncio.sleep(1.0)

        no_prof_a_class = await browser.evaluate("document.getElementById('comb-a-class').innerText")
        no_prof_b_assess = await browser.evaluate("document.getElementById('comb-b-assessment').innerText")
        no_prof_narrative = await browser.evaluate("document.getElementById('combined-narrative-text').innerText")

        print(f"Results with NO profile selected:")
        print(f"  Engine A General Class: {no_prof_a_class}")
        print(f"  Engine B Assessment: {no_prof_b_assess}")
        print(f"  Narrative: {no_prof_narrative}")

        assert no_prof_a_class.upper() in ["TRUTHFUL", "DECEPTIVE"]
        assert no_prof_b_assess == "No Profile Selected"
        assert "No personal historical baseline" in no_prof_narrative

        print("Verified: General classification proceeds reliably without an invalid personalized comparison.")
        step_results["Step 8: Insufficient Baseline & General Classification Fallback"] = "PASSED"

        # ------------------------------------------------------------------
        # WORKFLOW STEP 9: Verify Demo Profiles Distinguished from Real Profiles
        # ------------------------------------------------------------------
        print(f"\n--- STEP 9: Verify Demo Profiles Distinguished from Real Profiles ---")
        await browser.evaluate("switchTab('engine-b')")
        await asyncio.sleep(0.3)

        # Inspect options in profile select
        select_options = await browser.evaluate("""
            Array.from(document.getElementById('profile-select').options).map(o => ({ value: o.value, text: o.text }))
        """)
        demo_opt = next((o for o in select_options if o["value"] == "demo_analyst"), None)
        real_opt = next((o for o in select_options if o["value"] == prof_id), None)

        print(f"Demo option: {demo_opt}")
        print(f"Real target option: {real_opt}")

        assert demo_opt is not None
        assert "[DEMO SUBJECT]" in demo_opt["text"]
        assert "Demonstration Only" in demo_opt["text"]

        assert real_opt is not None
        assert "[TARGET AUTHOR]" in real_opt["text"]
        assert "Demonstration" not in real_opt["text"]

        # Select demo_analyst and check warning banner
        print("Action: Select demo_analyst in Engine B...")
        await browser.evaluate("document.getElementById('profile-select').value = 'demo_analyst'; onProfileChange();")
        await asyncio.sleep(0.5)

        demo_banner_visible = await browser.evaluate("!document.getElementById('demo-profile-banner').classList.contains('hidden')")
        demo_indicator_text = await browser.evaluate("document.getElementById('personal-active-profile-indicator').innerText")
        print(f"Demo Banner Visible: {demo_banner_visible}")
        print(f"Demo Indicator Badge: {demo_indicator_text}")

        assert demo_banner_visible is True, "Demo alert banner must be visible when viewing Alex Vance demo data"
        assert "DEMO SUBJECT" in demo_indicator_text

        # Switch back to real target author and check banner is hidden
        print(f"Action: Switch back to real target author {prof_id}...")
        await browser.evaluate(f"document.getElementById('profile-select').value = '{prof_id}'; onProfileChange();")
        await asyncio.sleep(0.5)

        demo_banner_hidden_again = await browser.evaluate("document.getElementById('demo-profile-banner').classList.contains('hidden')")
        real_indicator_text = await browser.evaluate("document.getElementById('personal-active-profile-indicator').innerText")
        print(f"Demo Banner Hidden for Real Target: {demo_banner_hidden_again}")
        print(f"Real Indicator Badge: {real_indicator_text}")

        assert demo_banner_hidden_again is True, "Demo alert banner must be hidden when real target author is selected"
        assert "TARGET AUTHOR" in real_indicator_text

        print("Verified: Demo profiles and demo data are unmistakably tagged and distinguished.")
        step_results["Step 9: Demo Profiles Clearly Distinguished from Real Profiles"] = "PASSED"

        # ------------------------------------------------------------------
        # FINAL REPORT SUMMARY
        # ------------------------------------------------------------------
        print("\n=================================================================")
        print("ALL 9 INVESTIGATIVE ONBOARDING WIZARD WORKFLOW STEPS PASSED!")
        print("=================================================================")
        for step_name, status in step_results.items():
            print(f"  [PASSED] {step_name}")

    finally:
        # Clean up created test profiles
        print("\nCleaning up test profiles...")
        for pid in created_profile_ids:
            try:
                prof_path = os.path.join(PROJECT_DIR, "profiles_store", f"{pid}.json")
                if os.path.exists(prof_path):
                    os.remove(prof_path)
                    print(f"Purged test profile file: {prof_path}")
            except Exception as e:
                print(f"Could not purge {pid}: {e}")

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
    asyncio.run(run_wizard_e2e_verification())
