import { expect, test, type Page, type Route } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

async function holdRequest(page: Page, url: string, method: string) {
  let capture!: (route: Route) => void;
  const request = new Promise<Route>((resolve) => {
    capture = resolve;
  });
  await page.route(url, (route) => {
    if (route.request().method() === method) capture(route);
    else return route.fallback();
  });
  return { request };
}

async function openAccountSettings(page: Page) {
  await page.goto("/workspace/chats/new");
  const sidebar = page.locator("[data-sidebar='sidebar']");
  await sidebar.getByRole("button", { name: /Settings and more/ }).click();
  await page.getByRole("menuitem", { name: "Settings", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Settings", exact: true });
  await dialog.getByRole("button", { name: "Account", exact: true }).click();
  return dialog;
}

test.describe("approved action feedback", () => {
  for (const succeed of [true, false]) {
    test(`password update displays pending feedback and recovers after ${succeed ? "success" : "failure"}`, async ({
      page,
    }) => {
      mockLangGraphAPI(page);
      const { request } = await holdRequest(
        page,
        "**/api/v1/auth/change-password",
        "POST",
      );
      const dialog = await openAccountSettings(page);
      const current = dialog.getByLabel("Current password", { exact: true });
      const password = dialog.getByLabel("New password", { exact: true });
      const confirmation = dialog.getByLabel("Confirm new password", {
        exact: true,
      });
      await current.fill("old-password");
      await password.fill("new-password");
      await confirmation.fill("new-password");
      await dialog
        .getByRole("button", { name: "Update Password", exact: true })
        .click();
      const route = await request;

      const pending = dialog.getByRole("button", {
        name: "Updating...",
        exact: true,
      });
      await expect(pending).toBeDisabled();
      await expect(pending.locator("svg.animate-spin")).toBeVisible();
      expect(route.request().postDataJSON()).toEqual({
        current_password: "old-password",
        new_password: "new-password",
      });
      await route.fulfill({
        status: succeed ? 200 : 400,
        contentType: "application/json",
        body: JSON.stringify(
          succeed ? {} : { detail: "Wrong current password" },
        ),
      });

      await expect(
        dialog.getByRole("button", { name: "Update Password", exact: true }),
      ).toBeEnabled();
      await expect(dialog.locator("button svg.animate-spin")).toHaveCount(0);
      if (succeed) {
        await expect(current).toHaveValue("");
        await expect(password).toHaveValue("");
        await expect(confirmation).toHaveValue("");
        await expect(
          dialog.getByText("Password changed successfully", { exact: true }),
        ).toBeVisible();
      } else {
        await expect(
          dialog.getByText("Wrong current password", { exact: true }),
        ).toBeVisible();
        await expect(current).toHaveValue("old-password");
      }
    });

    test(`agent deletion displays pending feedback and recovers after ${succeed ? "success" : "failure"}`, async ({
      page,
    }) => {
      mockLangGraphAPI(page, {
        agents: [{ name: "test-agent", description: "Test agent" }],
      });
      const { request } = await holdRequest(
        page,
        "**/api/agents/test-agent",
        "DELETE",
      );
      await page.goto("/workspace/agents");
      await page.getByTitle("Delete", { exact: true }).click();
      const dialog = page.getByRole("dialog", {
        name: "Delete",
        exact: true,
      });
      await dialog.getByRole("button", { name: "Delete", exact: true }).click();
      const route = await request;

      const pending = dialog.getByRole("button", {
        name: "Loading...",
        exact: true,
      });
      await expect(pending).toBeDisabled();
      await expect(pending.locator("svg.animate-spin")).toBeVisible();
      await expect(
        dialog.getByRole("button", { name: "Cancel", exact: true }),
      ).toBeDisabled();
      await route.fulfill({
        status: succeed ? 204 : 500,
        contentType: "application/json",
        body: succeed ? "" : JSON.stringify({ detail: "Delete failed" }),
      });

      if (succeed) {
        await expect(dialog).not.toBeVisible();
        await expect(
          page.getByText("Agent deleted", { exact: true }),
        ).toBeVisible();
      } else {
        await expect(
          dialog.getByRole("button", { name: "Delete", exact: true }),
        ).toBeEnabled();
        await expect(
          dialog.getByRole("button", { name: "Cancel", exact: true }),
        ).toBeEnabled();
        await expect(dialog.locator("svg.animate-spin")).toHaveCount(0);
      }
    });
  }

  test("named back action returns to the agent gallery", async ({ page }) => {
    mockLangGraphAPI(page);
    await page.goto("/workspace/agents/new");
    await page
      .getByRole("button", { name: "Back to Gallery", exact: true })
      .click();
    await expect(page).toHaveURL(/\/workspace\/agents$/);
  });
});
