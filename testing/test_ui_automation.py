import asyncio
import os
import time
from pathlib import Path
from playwright.async_api import async_playwright

BASE_URL = "http://localhost:5173"
SCREENSHOTS_DIR = Path("screenshots")
REPORT_PATH = Path("TEST_REPORT.md")

SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
RAVDESS_BASE = Path("../data/raw/ravdess/Actor_01")

emotions = [
    {"name": "neutral", "code": "01", "text_en": "The table is made of wood."},
    {"name": "calm", "code": "02", "text_en": "Everything is peaceful and quiet."},
    {"name": "happy", "code": "03", "text_en": "I am so excited and joyful!"},
    {"name": "sad", "code": "04", "text_en": "I feel very sad and lonely."},
    {"name": "angry", "code": "05", "text_en": "I am absolutely furious!"},
    {"name": "fearful", "code": "06", "text_en": "I am terrified of what might happen."},
    {"name": "disgust", "code": "07", "text_en": "This is disgusting and repulsive."},
    {"name": "surprised", "code": "08", "text_en": "Wow! That is completely unexpected!"},
]

multilingual = [
    {"lang": "hi", "text": "अरे! क्या यह तुम्हारी औकात है मेरे सामने आने की?", "name": "angry", "label": "Hindi"},
    {"lang": "mr", "text": "मी खूप आनंदात आहे", "name": "happy", "label": "Marathi"}
]

report_rows = []
ml_rows = []

def add_row(emotion, input_type, content, expected, predicted, conf, status, shot_name):
    link = f"[screenshot](screenshots/{shot_name})" if shot_name else "None"
    report_rows.append(f"| {emotion} | {input_type} | {content} | {expected} | {predicted} | {conf} | {status} | {link} |")

async def clear_session(page):
    clear_btn = page.locator("text=Clear Session").first
    if await clear_btn.is_visible():
        await clear_btn.click(force=True)
    await page.wait_for_timeout(1000)

async def trigger_analyze(page, e_name, shot_name, mode):
    await page.get_by_role("button", name="Analyze Emotion").click(force=True)
    try:
        # Wait for the results panel to appear uniquely identifying "Final Emotion:"
        await page.wait_for_selector("text=Final Emotion:", timeout=15000)
        await page.wait_for_timeout(1000)
        await page.screenshot(path=SCREENSHOTS_DIR / shot_name, full_page=True)
        return True, "Done"
    except Exception as ex:
        await page.screenshot(path=SCREENSHOTS_DIR / f"error_{shot_name}", full_page=True)
        return False, f"Fail: {str(ex).splitlines()[0]}"

async def run_scenario(page, e, mode="text"):
    await clear_session(page)
    e_name = e["name"]
    shot_name = f"test_{e_name}_{mode}.png"
    
    if mode == "text":
        text = e["text_en"]
        await page.fill("textarea", text)
        ok, stat = await trigger_analyze(page, e_name, shot_name, mode)
        add_row(e_name, mode, text, e_name, "CHECK_IMG", "CHECK_IMG", stat, shot_name if ok else f"error_{shot_name}")
        
    elif mode == "audio":
        code = e["code"]
        file_path = str(RAVDESS_BASE / f"03-01-{code}-01-01-01-01.wav")
        if not os.path.exists(file_path):
            add_row(e_name, mode, "N/A", e_name, "ERROR", "-", "Fail: Not Found", "")
            return
        await page.locator("input[type=file]").set_input_files(file_path)
        ok, stat = await trigger_analyze(page, e_name, shot_name, mode)
        msg = stat if ok else stat
        add_row(e_name, mode, "Audio Upload", e_name, "CHECK_IMG", "CHECK_IMG", msg, shot_name if ok else f"error_{shot_name}")

async def run_ml(page, m):
    await clear_session(page)
    e_name = m["name"]
    lang = m["lang"]
    label = m["label"]
    text = m["text"]
    
    await page.fill("textarea", text)
    # Select language Radix UI dropdown
    await page.get_by_role("combobox").click(force=True)
    await page.wait_for_timeout(500)
    await page.get_by_role("option", name=label).click(force=True)
    await page.wait_for_timeout(500)
    
    shot_name = f"test_{e_name}_{lang}_multilingual.png"
    ok, stat = await trigger_analyze(page, e_name, shot_name, f"ml_{lang}")
    
    s_link = f"[screenshot](screenshots/{shot_name})" if ok else "Fail"
    ml_rows.append(f"| {label} | {text[:15]}... | {e_name} | CHECK_IMG | CHECK_IMG | {s_link} |")


async def run_transcribe(page):
    await clear_session(page)
    file_path = str(RAVDESS_BASE / "03-01-03-01-01-01-01.wav")
    if not os.path.exists(file_path): return
    await page.locator("input[type=file]").set_input_files(file_path)
    
    btn = page.get_by_role("button", name="Auto-transcribe", exact=False)
    if not await btn.is_visible():
        btn = page.locator("text=Auto-transcribe").first
    await btn.click(force=True)
    
    await page.wait_for_timeout(8000) # Give whisper time
    shot_name = "test_auto_transcribe.png"
    await page.get_by_role("button", name="Analyze Emotion").click(force=True)
    
    try:
        await page.wait_for_selector("text=Final Emotion:", timeout=15000)
        await page.wait_for_timeout(1000)
        await page.screenshot(path=SCREENSHOTS_DIR / shot_name, full_page=True)
    except:
        pass

async def run_fusion(page):
    await clear_session(page)
    e_name_audio = "angry"
    e_name_text = "happy"
    
    # Text input
    await page.fill("textarea", "I am so excited and joyful!")
    
    # Audio input (Angry file)
    file_path = str(RAVDESS_BASE / "03-01-05-01-01-01-01.wav")
    if not os.path.exists(file_path): return
    await page.locator("input[type=file]").set_input_files(file_path)
    
    shot_name = f"test_fusion_{e_name_text}_{e_name_audio}.png"
    ok, stat = await trigger_analyze(page, "fusion", shot_name, "fusion")
    
    s_link = f"[screenshot](screenshots/{shot_name})" if ok else "Fail"
    report_rows.append(f"| Fusion | text+audio | Happy Text + Angry Audio | fusion | CHECK_IMG | CHECK_IMG | {stat} | {s_link} |")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto(BASE_URL)
        await page.wait_for_load_state("networkidle")
        
        for e in emotions:
            print(f"Testing {e['name']}")
            try: await run_scenario(page, e, "text")
            except Exception as x: print(x)
            try: await run_scenario(page, e, "audio")
            except Exception as x: print(x)
                
        for m in multilingual:
            print(f"Testing ML {m['lang']}")
            try: await run_ml(page, m)
            except Exception as x: print(x)
            
        print("Testing Auto-Transcribe")
        try: await run_transcribe(page)
        except Exception as x: print(x)
            
        print("Testing Fusion")
        try: await run_fusion(page)
        except Exception as x: print(x)
            
        await browser.close()
        
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("# EmpathyCo Test Report\n\n")
            f.write("## Summary\n")
            f.write(f"- Total test cases: {len(report_rows)} + {len(ml_rows)} multilingual + 1 transcription\n\n")
            f.write("## Detailed Results\n\n")
            f.write("| Emotion | Input Type | Input Content | Expected | Predicted | Confidence | Pass/Fail | Screenshot |\n")
            f.write("|---------|------------|---------------|----------|-----------|------------|-----------|-------------|\n")
            for r in report_rows: f.write(r + "\n")
            f.write("\n## Multilingual Results\n\n")
            f.write("| Language | Text | Expected | Predicted | Confidence | Screenshot |\n")
            f.write("|----------|------|----------|-----------|------------|------------|\n")
            for m in ml_rows: f.write(m + "\n")
            f.write("\n## Auto-transcribe Test\n")
            f.write("- Input audio: RAVDESS sample (Happy)\n")
            f.write("- Transcribed text: (See screenshot screenshots/test_auto_transcribe.png)\n")
            f.write("- Predicted emotion: (See screenshot)\n")

if __name__ == "__main__":
    asyncio.run(main())

