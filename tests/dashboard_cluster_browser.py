"""Exercise the organization switcher and the Cluster tab with API fixtures."""

import os

from playwright.sync_api import expect, sync_playwright

from dashboard_browser_smoke import dashboard

ORGS = [
    {"id": "org-a", "name": "Alpha", "created_at": "2026-10-01T00:00:00Z"},
    {"id": "org-b", "name": "Beta", "created_at": "2026-10-01T00:00:00Z"},
]
LOG = "\n".join(
    [
        "2026-10-06 13:06:14,343 Failed to start worker on 1x A100 80GB; cooling down",
    ]
    + [
        line
        for second in range(10, 50, 15)
        for line in (
            f"2026-10-06 16:41:{second},471 [org=beta (org-b)] workers: 0/16 (none), pending jobs: 1",
            f"2026-10-06 16:41:{second},472 Cannot start worker batch: cooldown ~2h{second}m left",
        )
    ]
)


def main():
    with dashboard(None) as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.getenv("OW_CHROMIUM_PATH"), args=["--no-sandbox"]
        )
        try:
            page = browser.new_page()
            errors, job_page_orgs = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.add_init_script("""
                localStorage.setItem('openweights_jwt', 'fixture');
                localStorage.setItem('openweights_jwt_expires_at', String(Math.floor(Date.now()/1000) + 3600));
            """)
            page.route("**/organizations/", lambda route: route.fulfill(json=ORGS))

            def jobs_page(route):
                job_page_orgs.append(route.request.url.split("/organizations/")[1].split("/")[0])
                route.fulfill(json={"items": [], "total": 0})

            page.route("**/organizations/*/jobs/page?*", jobs_page)
            page.route(
                "**/organizations/org-b/cluster/logs",
                lambda route: route.fulfill(body=LOG, content_type="text/plain"),
            )

            # Switching orgs must move the URL (the source of truth) and stay there.
            page.goto(url + "/org-a/jobs")
            expect(page.locator("header").get_by_role("combobox")).to_have_text("Alpha")
            page.locator("header").get_by_role("combobox").click()
            page.get_by_role("option", name="Beta").click()
            expect(page).to_have_url(url + "/org-b/jobs")
            page.wait_for_timeout(500)
            expect(page).to_have_url(url + "/org-b/jobs")
            expect(page.locator("header").get_by_role("combobox")).to_have_text("Beta")
            assert job_page_orgs[-1] == "org-b", job_page_orgs

            page.get_by_role("link", name="Cluster", exact=True).click()
            expect(page).to_have_url(url + "/org-b/cluster")
            expect(page.get_by_text("workers: 0/16 (none), pending jobs: 1").first).to_be_visible()
            expect(page.get_by_text("(×3, first at 2026-10-06 16:41:10)")).to_have_count(2)
            expect(page.get_by_text("Failed to start worker on 1x A100 80GB; cooling down")).to_be_visible()
            page.get_by_label("Collapse repeats").uncheck()
            expect(page.get_by_text("Cannot start worker batch", exact=False)).to_have_count(3)

            # Switching from the Cluster tab keeps the section.
            page.locator("header").get_by_role("combobox").click()
            page.get_by_role("option", name="Alpha").click()
            expect(page).to_have_url(url + "/org-a/cluster")
            assert not errors, errors
            print("Org switcher and Cluster tab passed")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
