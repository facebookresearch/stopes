const puppeteer = require("puppeteer");

jest.setTimeout(40000);

describe("File Viewer", () => {
  let browser;
  let page;

  beforeAll(async () => {
    browser = await puppeteer.launch();
    page = await browser.newPage();
  });

  afterAll(async () => {
    await browser.close();
  });

  beforeEach(async () => {
    await page.goto("http://localhost:3000/");
  });

  test("should display the filename input, fetch button and help button", async () => {
    const filenameInput = await page.waitForSelector(
      ".form-control.form-control-sm"
    );
    const fetchButton = await page.waitForSelector(".btn.btn-primary", {
      text: "Fetch!",
    });
    const helpButton = await page.waitForSelector(
      'button[aria-controls="help-text"]'
    );

    expect(filenameInput).not.toBeNull();
    expect(fetchButton).not.toBeNull();
    expect(helpButton).not.toBeNull();
  });

  test("should display the help text when the button is clicked", async () => {
    // Click the help button
    await page.click('button[aria-controls="help-text"]');

    // Wait for the help text to appear
    await page.waitForSelector("#help-text", { visible: true });

    // Assert that the help text is visible
    const helpTextVisible = await page.$eval("#help-text", (element) => {
      return getComputedStyle(element).display !== "none";
    });
    expect(helpTextVisible).toBe(true);
  });
});
