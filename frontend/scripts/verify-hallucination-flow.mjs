import { chromium } from "playwright-core";

const baseUrl = process.env.BASE_URL ?? "http://localhost:3000";
const chromePath =
  process.env.CHROME_PATH ??
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";

const browser = await chromium.launch({
  executablePath: chromePath,
  headless: true,
  args: ["--no-sandbox"],
});

const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const consoleErrors = [];
const checkRequests = [];

page.on("console", (message) => {
  if (message.type() === "error") {
    consoleErrors.push(message.text());
  }
});
page.on("pageerror", (error) => consoleErrors.push(error.message));
page.on("request", (request) => {
  if (request.url().endsWith("/api/hallucination/check")) {
    checkRequests.push(request.postData());
  }
});

try {
  console.log("1/8 opening Ask Documents");
  await page.goto(`${baseUrl}/ask`, { waitUntil: "networkidle" });
  if ((await page.getByLabel(/top_k/i).count()) !== 0) {
    throw new Error("Ask Documents still exposes a per-request top_k control");
  }
  await page.getByLabel("Question").fill("What is the purpose of ISO 27001?");
  console.log("2/8 submitting a real Ask request");
  const answerResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/rag/query") && response.request().method() === "POST",
    { timeout: 600000 },
  );
  await page.getByRole("button", { name: "Ask documents" }).click();
  const response = await answerResponse;
  if (!response.ok()) {
    throw new Error(`Ask request failed: ${response.status()} ${await response.text()}`);
  }
  const result = await response.json();
  if (!result.result_id) {
    throw new Error("Ask response did not include result_id");
  }

  await page.getByRole("heading", { name: "Answer" }).waitFor({ timeout: 600000 });
  const answerText = (await page.locator(".answer-text").first().textContent())?.trim();
  if (!answerText) {
    throw new Error("Ask page did not render the generated answer");
  }
  if ((await page.getByTestId("retrieved-evidence-list").count()) === 0) {
    throw new Error("Ask page did not render retrieved evidence");
  }
  if (result.verification?.status !== "pending") {
    throw new Error(`Ask request unexpectedly ran NLI: ${result.verification?.status}`);
  }
  if ((await page.getByTestId("verification-claim").count()) !== 0) {
    throw new Error("NLI claims rendered before the user requested verification");
  }
  const verifyButton = page.getByRole("button", {
    name: "Verify Answer Claims",
  });
  await verifyButton.waitFor();

  console.log("3/8 verifying the answer on demand");
  const verifyResponsePromise = page.waitForResponse(
    (candidate) =>
      candidate.url().includes("/api/verification/results/") &&
      candidate.request().method() === "POST",
    { timeout: 600000 },
  );
  await verifyButton.click();
  const verifyResponse = await verifyResponsePromise;
  if (!verifyResponse.ok()) {
    throw new Error(
      `Verification request failed: ${verifyResponse.status()} ${await verifyResponse.text()}`,
    );
  }
  const verified = await verifyResponse.json();
  if (verified.result_id !== result.result_id) {
    throw new Error("Verification response used a different result_id");
  }
  if (verified.verification?.status !== "verified" && verified.verification?.status !== "partial") {
    throw new Error(`Verification did not complete: ${verified.verification?.status}`);
  }
  await page.getByTestId("verification-claim").first().waitFor({
    timeout: 120000,
  });
  if ((await page.getByRole("button", { name: "Verify Answer Claims" }).count()) !== 0) {
    throw new Error("Verify Answer Claims is still offered after verification");
  }

  console.log("4/8 opening compact Hallucination Checks");
  await page.getByRole("link", { name: "Hallucination Checks" }).click();
  await page.waitForURL(`${baseUrl}/verification`);
  await page.getByText("Ask result ID").waitFor();
  await page
    .getByRole("definition")
    .filter({ hasText: result.question })
    .first()
    .waitFor();
  if ((await page.getByText("Retrieved evidence", { exact: true }).count()) !== 0) {
    throw new Error("Retrieved evidence is still shown on Hallucination Checks");
  }
  if ((await page.getByText("Existing NLI claim verification", { exact: true }).count()) !== 0) {
    throw new Error("Detailed NLI results are still shown on Hallucination Checks");
  }

  console.log("5/8 compact Hallucination Checks metadata rendered");
  const button = page.getByRole("button", { name: "Check for Hallucinations" });
  await button.waitFor();
  if (await button.isDisabled()) {
    throw new Error("Check for Hallucinations is disabled for a valid result");
  }

  const checkResponsePromise = page.waitForResponse(
    (candidate) =>
      candidate.url().endsWith("/api/hallucination/check") &&
      candidate.request().method() === "POST",
    { timeout: 180000 },
  );
  await button.evaluate((element) => {
    element.click();
    element.click();
  });
  const checkResponse = await checkResponsePromise;
  if (!checkResponse.ok()) {
    throw new Error(
      `Hallucination request failed: ${checkResponse.status()} ${await checkResponse.text()}`,
    );
  }
  const check = await checkResponse.json();
  if (check.result_id !== result.result_id) {
    throw new Error("Hallucination response used a different result_id");
  }
  if (checkRequests.length !== 1) {
    throw new Error(`Rapid clicks sent ${checkRequests.length} requests`);
  }

  console.log("6/8 detector request completed");
  await page.getByRole("heading", { name: "Groundedness summary" }).waitFor();
  if ((await page.getByText(check.claims[0]?.claim ?? "", { exact: true }).count()) === 0) {
    throw new Error("Hallucination claim was not rendered");
  }

  await page.screenshot({
    path: "C:\\Users\\tharu\\AppData\\Local\\Temp\\opencode\\hallucination-desktop.png",
    fullPage: true,
  });

  console.log("7/8 checking refresh and back-navigation persistence");
  await page.reload({ waitUntil: "networkidle" });
  await page.getByText("Ask result ID").waitFor();
  await page
    .getByRole("definition")
    .filter({ hasText: result.question })
    .first()
    .waitFor();

  await page.getByRole("link", { name: "Ask Documents" }).click();
  await page.waitForURL(`${baseUrl}/ask`);
  await page.getByRole("heading", { name: "Answer" }).waitFor({ timeout: 60000 });
  const restoredAnswer = (await page.locator(".answer-text").first().textContent())?.trim();
  if (restoredAnswer !== answerText) {
    throw new Error("Ask page lost the shared result after navigation");
  }

  await page.getByRole("link", { name: "Hallucination Checks" }).click();
  await page.waitForURL(`${baseUrl}/verification`);
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(250);
  const overflow = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    theme: document.documentElement.dataset.theme,
  }));
  if (overflow.theme !== "dark") {
    throw new Error("Dark theme was not applied");
  }
  if (overflow.scrollWidth > overflow.clientWidth + 1) {
    throw new Error(
      `Mobile layout overflows horizontally: ${overflow.scrollWidth}/${overflow.clientWidth}`,
    );
  }
  await page.screenshot({
    path: "C:\\Users\\tharu\\AppData\\Local\\Temp\\opencode\\hallucination-mobile-dark.png",
    fullPage: true,
  });

  console.log("8/8 checking mobile layout and clear action");
  await page.getByRole("button", { name: "Clear shared result" }).click();
  await page.getByText("No Ask result to check").waitFor();
  const storedId = await page.evaluate(() =>
    window.sessionStorage.getItem("rag.latestResultId"),
  );
  if (storedId !== null) {
    throw new Error("Clearing the shared result left a stored identifier");
  }

  if (consoleErrors.length > 0) {
    throw new Error(`Browser console errors: ${consoleErrors.join(" | ")}`);
  }

  console.log(
    JSON.stringify(
      {
        resultId: result.result_id,
        answer: answerText,
        sources: result.sources.length,
        askNliStatus: result.verification?.status,
        verifiedNliStatus: verified.verification?.status,
        nliClaims: verified.verification?.details?.claims?.length ?? 0,
        detectorClaims: check.claims.length,
        detectorSummary: check.summary,
        checkRequests: checkRequests.length,
        refreshRestored: true,
        backNavigationRestored: true,
        mobileOverflow: overflow,
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
}
