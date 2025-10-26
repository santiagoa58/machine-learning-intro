import { test, expect } from '@playwright/test';

test.describe('Visual Comparison - ML Learning Path vs Compass Template', () => {
  test('Homepage - ML Learning Path', async ({ page }) => {
    await page.goto('http://localhost:3001');
    await page.waitForLoadState('networkidle');

    // Take full page screenshot
    await page.screenshot({
      path: 'e2e/screenshots/ml-learning-path-homepage-full.png',
      fullPage: true,
    });

    // Take viewport screenshot for above-the-fold comparison
    await page.screenshot({
      path: 'e2e/screenshots/ml-learning-path-homepage-viewport.png',
    });

    // Verify page loaded correctly
    await expect(page.getByRole('heading', { name: 'Machine Learning Introduction' })).toBeVisible();
  });

  test('Homepage - Compass Template', async ({ page }) => {
    await page.goto('http://localhost:3002');
    await page.waitForLoadState('networkidle');

    // Take full page screenshot
    await page.screenshot({
      path: 'e2e/screenshots/compass-template-homepage-full.png',
      fullPage: true,
    });

    // Take viewport screenshot
    await page.screenshot({
      path: 'e2e/screenshots/compass-template-homepage-viewport.png',
    });
  });

  test('Docs Page - ML Learning Path', async ({ page }) => {
    await page.goto('http://localhost:3001/docs/readme');
    await page.waitForLoadState('networkidle');

    // Take full page screenshot
    await page.screenshot({
      path: 'e2e/screenshots/ml-learning-path-docs-full.png',
      fullPage: true,
    });

    // Take viewport screenshot
    await page.screenshot({
      path: 'e2e/screenshots/ml-learning-path-docs-viewport.png',
    });

    // Verify prose content is rendered
    await expect(page.locator('article.prose')).toBeVisible();
  });

  test('Theme Elements - ML Learning Path', async ({ page }) => {
    await page.goto('http://localhost:3001');
    await page.waitForLoadState('networkidle');

    // Screenshot of card components
    const cards = page.locator('section.bg-white').first();
    await cards.screenshot({
      path: 'e2e/screenshots/ml-learning-path-card-component.png',
    });

    // Screenshot of button if visible
    const button = page.locator('a[href^="http"]').first();
    if (await button.isVisible()) {
      await button.screenshot({
        path: 'e2e/screenshots/ml-learning-path-link-component.png',
      });
    }
  });

  test('Theme Elements - Compass Template', async ({ page }) => {
    await page.goto('http://localhost:3002');
    await page.waitForLoadState('networkidle');

    // Wait for any images/content to load
    await page.waitForTimeout(1000);

    // Screenshot of a button component
    const startButton = page.getByRole('link', { name: /start/i }).first();
    if (await startButton.isVisible()) {
      await startButton.screenshot({
        path: 'e2e/screenshots/compass-template-button-component.png',
      });
    }
  });

  test('Dark Mode - ML Learning Path', async ({ page, context }) => {
    // Enable dark mode via class
    await page.goto('http://localhost:3001');
    await page.waitForLoadState('networkidle');

    // Add dark class to html element
    await page.evaluate(() => {
      document.documentElement.classList.add('dark');
    });

    // Wait for dark mode to apply
    await page.waitForTimeout(500);

    // Take screenshot
    await page.screenshot({
      path: 'e2e/screenshots/ml-learning-path-homepage-dark.png',
      fullPage: true,
    });
  });

  test('Dark Mode - Compass Template', async ({ page }) => {
    await page.goto('http://localhost:3002');
    await page.waitForLoadState('networkidle');

    // Add dark class
    await page.evaluate(() => {
      document.documentElement.classList.add('dark');
    });

    await page.waitForTimeout(500);

    // Take screenshot
    await page.screenshot({
      path: 'e2e/screenshots/compass-template-homepage-dark.png',
      fullPage: true,
    });
  });
});
