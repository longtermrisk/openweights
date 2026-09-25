"""Exercise the built jobs UI with paginated API fixtures (no real credentials)."""

import json
import os
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright

from dashboard_browser_smoke import dashboard


def main():
    jobs = [
        {
            "id": f"job-{n:04}",
            "type": "custom",
            "status": (
                "pending" if n < 200 else "in_progress" if n < 400 else "completed"
            ),
            "model": "needle" if n == 1199 else "example",
            "created_at": "2026-09-25T00:00:00Z",
        }
        for n in range(1200)
    ]
    with dashboard(None) as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.getenv("OW_CHROMIUM_PATH"), args=["--no-sandbox"]
        )
        try:
            page = browser.new_page()
            errors, requests = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.add_init_script("""
                localStorage.setItem('openweights_jwt', 'fixture');
                localStorage.setItem('openweights_jwt_expires_at', String(Math.floor(Date.now()/1000) + 3600));
            """)
            page.route(
                "**/organizations/",
                lambda route: route.fulfill(
                    json=[
                        {
                            "id": "org",
                            "name": "Test",
                            "created_at": "2026-09-25T00:00:00Z",
                        }
                    ]
                ),
            )

            def respond(route):
                params = parse_qs(urlparse(route.request.url).query)
                requests.append(params)
                matching = [job for job in jobs if job["status"] in params["status"]]
                search = params.get("search", [""])[0].lower()
                matching = [
                    job for job in matching if search in json.dumps(job).lower()
                ]
                offset, limit = int(params["offset"][0]), int(params["limit"][0])
                route.fulfill(
                    json={
                        "items": matching[offset : offset + limit],
                        "total": len(matching),
                    }
                )

            page.route("**/organizations/org/jobs/page?*", respond)
            page.goto(url + "/org/jobs")
            expect(
                page.get_by_role("heading", name="Pending (200)", exact=True)
            ).to_be_visible()
            expect(
                page.get_by_role("heading", name="Finished (800)", exact=True)
            ).to_be_visible()
            expect(
                page.get_by_role("link", name="job-0000", exact=True)
            ).to_be_visible()
            page.get_by_role("button", name="Go to next page").first.click()
            expect(
                page.get_by_role("link", name="job-0010", exact=True)
            ).to_be_visible()
            page.get_by_role("button", name="list view", exact=True).click()
            expect(page.get_by_role("row")).to_have_count(11)
            expect(page.get_by_text("1–10 of 1200", exact=True)).to_be_visible()
            page.get_by_label("Search", exact=True).fill("needle")
            expect(
                page.get_by_role("rowheader", name="job-1199", exact=True)
            ).to_be_visible()
            expect(page.get_by_role("row")).to_have_count(2)
            expect(page.get_by_text("1–1 of 1", exact=True)).to_be_visible()
            assert all(int(request["limit"][0]) == 10 for request in requests)
            assert not errors, errors
            print("Jobs browser pagination, totals, list view and search passed")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
