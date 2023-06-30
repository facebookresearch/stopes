import puppeteer from "puppeteer"


jest.setTimeout(40000); 

describe('Homepage', () => {
  let browser;
  let page;

  beforeAll(async () => {
    browser = await puppeteer.launch();
    page = await browser.newPage();
    await page.goto('https://facebookresearch.github.io/stopes/'); // Replace with the URL of your application
  });

  afterAll(async () => {
    await browser.close();
  });

  test('Banner should be displayed', async () => {
    const banner = await page.$('header.sbanner');
    expect(banner).not.toBeNull();
  });

  test('Features section should be rendered', async () => {
    const featureCards = await page.$$('div > .col.sfeatures .card.card--full-height');
    expect(featureCards.length).toBe(3);
  });

  test('Content sections should be rendered', async () => {
    const contentSections = await page.$$('section > .ssection');
    expect(contentSections.length).toBe(3);
  });
});

