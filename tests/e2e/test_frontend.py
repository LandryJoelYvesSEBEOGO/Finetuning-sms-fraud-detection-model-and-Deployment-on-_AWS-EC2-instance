import re

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


def test_initial_state(page: Page) -> None:
    expect(page.locator("#status")).to_have_class(re.compile("status--ready"))
    expect(page.locator("#result")).to_have_attribute("data-state", "idle")
    expect(page.get_by_label("SMS message (English)")).to_be_visible()
    expect(page.locator("#examples button")).to_have_count(4)


def test_fraud_example(page: Page) -> None:
    page.get_by_role("button", name="Prize claim").click()
    expect(page.locator("#result")).to_have_attribute("data-state", "fraud")
    expect(page.locator("#verdictText")).to_have_text("Likely fraud")
    expect(page.locator("#scoreValue")).to_have_text("97%")
    expect(page.locator("#thresholdValue")).to_have_text("50%")
    expect(page.locator("#modelValue")).to_have_text("e2e@0000000+fake")
    expect(page.locator("#meter")).to_have_attribute("aria-valuenow", "97")


def test_typed_message_with_keyboard_shortcut(page: Page) -> None:
    page.get_by_label("SMS message (English)").fill("See you at the station at 6")
    expect(page.locator("#counter")).to_have_text("27 / 2000")
    page.keyboard.press("Control+Enter")
    expect(page.locator("#result")).to_have_attribute("data-state", "legit")
    expect(page.locator("#verdictText")).to_have_text("Looks legitimate")


def test_empty_message_shows_an_error(page: Page) -> None:
    page.get_by_role("button", name="Analyze").click()
    expect(page.locator("#result")).to_have_attribute("data-state", "error")
    expect(page.locator("#errorText")).to_have_text("Please enter a message to analyze.")


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (422, "The message must be between 1 and 2000 characters."),
        (503, "The model is starting up. Try again in a few seconds."),
        (500, "The analysis service returned an error."),
    ],
)
def test_api_errors_are_explained(page: Page, status: int, expected: str) -> None:
    page.route("**/v1/predict", lambda route: route.fulfill(status=status, body="{}"))
    page.get_by_label("SMS message (English)").fill("hello")
    page.get_by_role("button", name="Analyze").click()
    expect(page.locator("#errorText")).to_have_text(expected)


def test_network_failure_then_retry(page: Page) -> None:
    page.route("**/v1/predict", lambda route: route.abort())
    page.get_by_label("SMS message (English)").fill("URGENT call now")
    page.get_by_role("button", name="Analyze").click()
    expect(page.locator("#errorText")).to_have_text("Could not reach the analysis service.")
    page.unroute("**/v1/predict")
    page.get_by_role("button", name="Try again").click()
    expect(page.locator("#result")).to_have_attribute("data-state", "fraud")


def test_clear_resets_the_form(page: Page) -> None:
    page.get_by_role("button", name="Lunch plan").click()
    expect(page.locator("#result")).to_have_attribute("data-state", "legit")
    page.get_by_role("button", name="Clear").click()
    expect(page.locator("#result")).to_have_attribute("data-state", "idle")
    expect(page.get_by_label("SMS message (English)")).to_have_value("")


def test_language_toggle_translates_and_persists(page: Page) -> None:
    page.get_by_role("button", name="Prize claim").click()
    expect(page.locator("#verdictText")).to_have_text("Likely fraud")
    page.get_by_role("button", name="Français").click()
    expect(page.locator("html")).to_have_attribute("lang", "fr")
    expect(page.locator("h1")).to_have_text("Détecteur de fraude SMS")
    # The current verdict is re-rendered in the new language.
    expect(page.locator("#verdictText")).to_have_text("Fraude probable")
    page.reload()
    expect(page.locator("h1")).to_have_text("Détecteur de fraude SMS")


def test_mobile_layout_has_no_horizontal_scroll(browser, base_url: str) -> None:
    context = browser.new_context(viewport={"width": 360, "height": 740}, color_scheme="dark")
    page = context.new_page()
    page.goto(base_url)
    page.get_by_role("button", name="Bank alert").click()
    expect(page.locator("#result")).to_have_attribute("data-state", "fraud")
    assert page.evaluate("document.documentElement.scrollWidth") <= 360
    context.close()


def test_api_routes_still_win_over_static_files(page: Page, base_url: str) -> None:
    response = page.request.get(f"{base_url}/health")
    assert response.json() == {"status": "ok"}


def test_page_is_served_with_the_production_csp(page: Page, base_url: str) -> None:
    csp = page.request.get(f"{base_url}/").headers["content-security-policy"]
    assert "script-src 'self'" in csp
    # The page fixture fails the test on any console error, CSP violations included.
    page.get_by_role("button", name="Prize claim").click()
    expect(page.locator("#result")).to_have_attribute("data-state", "fraud")


def test_docs_are_not_blocked_by_the_csp(page: Page, base_url: str) -> None:
    assert "content-security-policy" not in page.request.get(f"{base_url}/docs").headers
