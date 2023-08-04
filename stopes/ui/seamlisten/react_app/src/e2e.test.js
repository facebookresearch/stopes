const puppeteer = require('puppeteer');

jest.setTimeout(40000);

describe('File Viewer', () => {
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
    await page.goto('http://localhost:3000/'); 
  });

  test('should display the filename input, fetch button and help button', async () => {
    const filenameInput = await page.waitForSelector('.form-control.form-control-sm');
    const fetchButton = await page.waitForSelector('.btn.btn-primary', { text: 'Fetch!' });
    const helpButton = await page.waitForSelector('button[aria-controls="help-text"]');

    expect(filenameInput).not.toBeNull();
    expect(fetchButton).not.toBeNull();
    expect(helpButton).not.toBeNull();

  });

  test('should display the help text when the button is clicked', async () => {
    // Click the help button
    await page.click('button[aria-controls="help-text"]');

    // Wait for the help text to appear
    await page.waitForSelector('#help-text', { visible: true });

    // Assert that the help text is visible
    const helpTextVisible = await page.$eval('#help-text', (element) => {
      return getComputedStyle(element).display !== 'none';
    });
    expect(helpTextVisible).toBe(true);
  });

  test("should update the filename input with pasted data", async () => {
    const filenameInput = await page.waitForSelector(
      ".form-control.form-control-sm"
    );
    const pastedFilename = "/default/path/";

    await page.evaluate(
      //a function to simulate a paste action
      (input, value) => {
        const event = new Event("paste", {
          bubbles: true,
          cancelable: true,
          composed: true,
        });
        event.clipboardData = new DataTransfer();
        event.clipboardData.setData("text/plain", value);
        input.dispatchEvent(event);
      },
      // arguments for our function are passed here
      filenameInput, // input
      pastedFilename //value
    );

    // Wait for a moment to allow the paste event handler to run
    await new Promise((r) => setTimeout(r, 100));

    // check whether the inputvalue matches the pastedFileName
    const inputValue = await page.$eval(
      ".form-control.form-control-sm",
      (el) => el.value
    );
    expect(inputValue).toBe(pastedFilename);
  });
});
