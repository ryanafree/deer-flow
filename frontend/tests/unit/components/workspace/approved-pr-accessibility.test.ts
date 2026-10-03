import { describe, expect, it, rs } from "@rstest/core";
import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

rs.mock("next/navigation", () => ({
  useRouter: () => ({ push: rs.fn() }),
}));
rs.mock("@/core/auth/AuthProvider", () => ({
  useAuth: () => ({
    user: { email: "test@example.com", system_role: "admin" },
    logout: rs.fn(),
  }),
}));
rs.mock("@/core/threads/hooks", () => ({
  useThreadStream: () => ({
    thread: { messages: [], values: {}, isLoading: false },
    sendMessage: rs.fn(),
  }),
}));

import NewAgentPage from "@/app/workspace/agents/new/page";
import { MessageListItem } from "@/components/workspace/messages/message-list-item";
import { AccountSettingsPage } from "@/components/workspace/settings/account-settings-page";
import { I18nContext } from "@/core/i18n/context";
import { enUS } from "@/core/i18n/locales/en-US";
import { zhCN } from "@/core/i18n/locales/zh-CN";

function render(element: ReactElement, locale: "en-US" | "zh-CN") {
  return renderToStaticMarkup(
    createElement(
      I18nContext.Provider,
      { value: { locale, setLocale: () => undefined } },
      element,
    ),
  );
}

describe("approved accessible controls", () => {
  for (const [locale, t] of [
    ["en-US", enUS],
    ["zh-CN", zhCN],
  ] as const) {
    it(`connects visible password labels to required inputs in ${locale}`, () => {
      const html = render(createElement(AccountSettingsPage), locale);
      for (const [id, label] of [
        ["currentPassword", t.settings.account.currentPassword],
        ["newPassword", t.settings.account.newPassword],
        ["confirmNewPassword", t.settings.account.confirmNewPassword],
      ]) {
        expect(html).toContain(`for="${id}"`);
        expect(html).toContain(`>${label}</label>`);
        expect(html).toMatch(
          new RegExp(`<input[^>]*id="${id}"[^>]*required=""`),
        );
      }
      expect(html.match(/minLength="8"/g)).toHaveLength(2);
    });

    it(`names the agent-gallery back action in ${locale}`, () => {
      const html = render(createElement(NewAgentPage), locale);
      expect(html).toContain(`aria-label="${t.agents.backToGallery}"`);
    });

    it(`names both feedback actions and preserves the selected rating in ${locale}`, () => {
      const html = render(
        createElement(MessageListItem, {
          message: { type: "ai", id: "ai-1", content: "Answer" },
          threadId: "thread-1",
          runId: "run-1",
          feedback: { feedback_id: "feedback-1", rating: -1, comment: null },
        }),
        locale,
      );
      expect(html).toContain(`aria-label="${t.conversation.goodResponse}"`);
      expect(html).toContain(`aria-label="${t.conversation.badResponse}"`);
      expect(html).toMatch(/lucide-thumbs-down[^>]*fill-current/);
      expect(html).not.toMatch(/lucide-thumbs-up[^>]*fill-current/);
    });
  }
});
